import os
import re
import pickle
import string
import gensim
import logging
import nltk
import multiprocessing
from smart_open import open
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.spatial import distance
from BIOSSESDataset import BIOSSESDataset

STOPWORDS = set(nltk.corpus.stopwords.words('english'))

def get_txt_paths(starting_path, file_paths):
    for entry in os.scandir(starting_path):
      
        if not entry.name.startswith('.') and \
          not entry.name == 'drive' and \
          not entry.name == 'sample_data' and \
          entry.is_dir(follow_symlinks=False):
            get_txt_paths(entry.path, file_paths)

        if not entry.name.startswith('.') and entry.is_file():
            file_paths.append(entry.path)


def get_tagged_docs(file_paths):
    doc_set = set()
    idx = 0

    for file_path in file_paths:
        with open(file_path, 'r', encoding='ascii', errors='ignore') as txt:
            raw_txt = txt.read()
            body = re.search('==== Body(.*)==== Refs', raw_txt, re.DOTALL)

            if body != None:
                body = body.group(1).split()
                paper = tuple([token for token in body if not token in STOPWORDS])
                doc = TaggedDocument(paper, tuple([idx]))
                doc_set.add(doc)
                idx += 1
    
    if len(doc_set) != 0:
        return list(doc_set)
            

# Get proper file paths for all articles used in training
filepaths = []
current_path = os.getcwd() + '/data'
print(current_path)
get_txt_paths(current_path, filepaths)

# Set up logging for doc2vec training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Set up corpus iterator for streaming data in
corpus = get_tagged_docs(filepaths)

# Train doc2vec
model = gensim.models.doc2vec.Doc2Vec(vector_size=96, min_count=5, workers=multiprocessing.cpu_count())
model.build_vocab(corpus, progress_per=1000)
model.train(corpus, total_examples=model.corpus_count, epochs=10)

# Create directory to save model
SAVE_DIR = './d2v_OAPMC_n/'
if not os.path.isdir(SAVE_DIR):
  os.mkdir(SAVE_DIR)

model.save(SAVE_DIR + 'model.root')

# Preprocessing function for benchmarking
def preprocess_sentence(text):
    return [token for token in text.split() if not token in STOPWORDS]

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

# Dump similarity scores using open
with open(SAVE_DIR + 'sim_scores_n.txt', 'w') as f:
  for score in sim_scores:
    f.write(str(score) + "\n")
