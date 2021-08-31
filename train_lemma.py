import os
import re
import pickle
import string
import spacy
import scispacy
import gensim
import logging
import multiprocessing
from smart_open import open
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.spatial import distance
from BIOSSESDataset import BIOSSESDataset

nlp = spacy.load('en_core_sci_sm', disable=['tok2vec','parser','ner']) 

def get_txt_paths(starting_path, file_paths):
    for entry in os.scandir(starting_path):
      
        if not entry.name.startswith('.') and \
          not entry.name == 'drive' and \
          not entry.name == 'sample_data' and \
          entry.is_dir(follow_symlinks=False):
            get_txt_paths(entry.path, file_paths)

        if not entry.name.startswith('.') and entry.is_file():
            file_paths.append(entry.path)


class PMCSubsetCorpus:
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self.cite_pattern = r'(\s+\(*(([À-ÿA-Za-z\s\-.,;&])+\s\(*(\d{4}[a-z]*)+\)*)+)|(\s+\[[\d\s+,;&\[\]]+\])'
        self.idx = 0

    def __iter__(self):
        for filepath in self.filepaths[:100]:
            with open(filepath, 'r', encoding='ascii', errors='ignore') as f:
                body = re.sub(self.cite_pattern, '', f.read())
                body = re.search('==== Body(.*)==== Refs', body, re.DOTALL)

                if body is not None:
                  body = " ".join(body.group(1).split())
                  paper = [token.lemma_ for token in nlp(body) if not token.is_stop]
                  yield TaggedDocument(paper, [self.idx])
                  self.idx += 1

# Get proper file paths for all articles used in training
filepaths = []
current_path = os.getcwd() + '/data'
print(current_path)
get_txt_paths(current_path, filepaths)

# Set up logging for doc2vec training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Set up corpus iterator for streaming data in
corpus = PMCSubsetCorpus(filepaths)

# Train doc2vec
model = gensim.models.doc2vec.Doc2Vec(vector_size=96, min_count=5, workers=multiprocessing.cpu_count())
model.build_vocab(corpus, progress_per=1000)
model.train(corpus, total_examples=model.corpus_count, epochs=10)

# Create directory to save model
SAVE_DIR = './d2v_OAPMC_il/'
if not os.path.isdir(SAVE_DIR):
  os.mkdir(SAVE_DIR)

model.save(SAVE_DIR + 'model.root')

# Preprocessing function for benchmarking
def preprocess_sentence(text):
    return [token.lemma_ for token in nlp(text) if not token.is_stop]

# Get BIOSSES Dataset for benchmarking
dataset = BIOSSESDataset()
df = dataset.get_dataframe()

# Turn BIOSSES sentences into tokens
sent1_tokens = df['Sentence 1'].apply(preprocess_sentence)
sent2_tokens = df['Sentence 2'].apply(preprocess_sentence)

# Generate similarity scores from the benchmark using doc2vec
def sim_from_toks(model, tokens1, tokens2):
    model.random.seed(0)
    return 1 - distance.cosine(model.infer_vector(tokens1), model.infer_vector(tokens2))

sim_scores = list(map(lambda x, y: sim_from_toks(model, x, y), sent1_tokens, sent2_tokens))

# Dump similarity scores using pickle
with open(SAVE_DIR + 'sim_scores_il.txt', 'w') as f:
  for score in sim_scores:
    f.write(str(score) + "\n")
