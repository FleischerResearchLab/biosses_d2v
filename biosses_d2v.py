import os
import re
import pickle
import string
import logging
import multiprocessing
from subprocess import run
from numpy import array
from docx import Document
from nltk.corpus import stopwords
from smart_open import open
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.spatial import distance


class BIOSSESDataset:
    """
    Dataset class for the BIOSSES Benchmark.
    Source: https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html.
    """
    
    def __init__(self):
        BIOSSES_URL = "https://tabilab.cmpe.boun.edu.tr/BIOSSES/Downloads/BIOSSES-Dataset.rar"
        if not os.path.isfile("BIOSSES-Dataset/Annotation-Pairs.docx") \
          or not os.path.isfile("BIOSSES-Dataset/Annotator-Scores.docx"):
            run(["wget", BIOSSES_URL])
            run(["unrar", "x", "BIOSSES-Dataset.rar"])

 
    def get_sentences(self):
        doc = Document(os.path.abspath("BIOSSES-Dataset/Annotation-Pairs.docx"))
        table = doc.tables[0]
        sent_arr = array([[cell.text.strip() for cell in row.cells] for row in table.rows])
        return sent_arr[1:,1:]

    
    def get_scores(self):
        doc = Document(os.path.abspath("BIOSSES-Dataset/Annotator-Scores.docx"))
        table = doc.tables[0]
        score_arr = array([[cell.text.strip() for cell in row.cells] for row in table.rows], dtype="i1")
        return score_arr[1:,1:]


class PMCSubsetCorpus:
    def __init__(self, stopwords=[], sub_pattern=None, lemma=False):
        self.sub_pattern = sub_pattern #r"(\s+\(*(([À-ÿA-Za-z\s\-.,;&])+\s\(*(\d{4}[a-z]*)+\)*)+)|(\s+\[[\d\s+,;&\[\]]+\])"
        self.stopwords = stopwords
        self.corpus_paths = []
        if lemma:
            self.nlp = spacy.load('en_core_sci_sm', disable=['tok2vec','parser','ner']) 


    def __iter__(self):
        doc_id = 0 
        for paper_path in self.corpus_paths[:100]:
            tagged_doc = self._preprocess_doc(paper_path, doc_id)
            if tagged_doc is not None:
                yield tagged_doc
                doc_id += 1


    def _append_corpus_paths(self, start_path, path_arr):
        for entry in os.scandir(start_path):
            if not entry.name.startswith('.') \
              and not entry.name == 'drive' \
              and not entry.name == 'sample_data' \
              and entry.is_dir(follow_symlinks=False):
                  self._append_corpus_paths(entry.path, path_arr)

            if not entry.name.startswith('.') and entry.is_file():
                path_arr.append(entry.path)

                
    def _preprocess_doc(self, path, id):
        with open(path, 'r', encoding='ascii', errors='ignore') as f:
            paper = f.read()
            if self.sub_pattern is not None:
                paper = re.sub(self.sub_pattern, '', f.read())
            body = re.search('==== Body(.*)==== Refs', paper, re.DOTALL)

            if body is not None:
                doc = body.group(1).split()
                if self.nlp is not None:
                    doc = " ".join(doc)
                    doc = [token.lemma_ for token in self.nlp(doc) if not token in self.stopwords]
                else:
                    doc = [token for token in doc if not token in self.stopwords]
                return TaggedDocument(doc, [id])


    def load(self, subsets=["0-9A-B"]):
        PMCSC_URL = "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/"
        ZIP_NAMES = []

        for subset in subsets:
            ZIP_NAMES.append("non_comm_use." + subset + ".txt.tar.gz")

        for zname in ZIP_NAMES:
            fdir = zname.replace(".txt.tar.gz", "")
            fdir_path = os.path.abspath(fdir)

            if not os.path.isdir(fdir_path):
                run(["mkdir", fdir])
                if not os.path.isfile(os.path.abspath(zname)):
                    run(["wget", PMCSC_URL + zname])
                run(["tar", "-xf", zname, "-C", fdir])
            
            self._append_corpus_paths(fdir_path, self.corpus_paths)

    
    def get_list(self):
        doc_list = []
        doc_id = 0

        for paper_path in self.corpus_paths[:100]:
            tagged_doc = self._preprocess_doc(paper_path, doc_id)
            if tagged_doc is not None:
                doc_list.append(tagged_doc)
                print(tagged_doc)
                doc_id += 1
        return doc_list
        

def main():
    train_corpus = PMCSubsetCorpus(stopwords=set(stopwords.words('english')))
    train_corpus.load()
    biodata = BIOSSESDataset()
    biosent = biodata.get_sentences()
    print(biosent[59])
    print(train_corpus.corpus_paths[10000])
    train_corpus = train_corpus.get_list()

    # Train doc2vec
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Doc2Vec(vector_size=96, min_count=5, workers=multiprocessing.cpu_count())
    model.build_vocab(train_corpus, progress_per=1000)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=1)


if __name__ == "__main__":
    main()