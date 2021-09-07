import os
import re
import pickle
import spacy
import string
import logging
import argparse
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
        score_arr = array([[cell.text.strip() for cell in row.cells] \
                            for row in table.rows], dtype="i1")
        return score_arr[1:,1:]


class PMCSubsetCorpus:
    def __init__(self, stopwords=[], sub_pattern=None, lemma=False):
        self.sub_pattern = sub_pattern #r"(\s+\(*(([À-ÿA-Za-z\s\-.,;&])+\s\(*(\d{4}[a-z]*)+\)*)+)|(\s+\[[\d\s+,;&\[\]]+\])"
        self.stopwords = stopwords
        self.corpus_paths = []
        self.nlp = None
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
                doc = body.group(1).lower().split()
                print(self.nlp)
                if self.nlp is not None:
                    doc = " ".join(doc)
                    doc = [token.lemma_ for token in self.nlp(doc) \
                            if not token in self.stopwords]
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


def run_doc2vec(lemma=False, logger=False, progress_per=1000, **kwargs):
    if logger:
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", 
                            level=logging.INFO)
    
    corpus = PMCSubsetCorpus(lemma=lemma)
    corpus.load()
    corpus = corpus.get_list()

    for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))
    model = Doc2Vec(**kwargs)
    model.build_vocab(corpus, progress_per=progress_per)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    #return model


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--vector_size",
        type=int,
        default=56,
        help="Dimensionality of Doc2Vec embeddings."
    )
    parser.add_argument(
        "-w", "--window",
        type=int,
        default=5,
        help="Maximum distance between current and predicted word during training."
    )
    parser.add_argument(
        "-a", "--alpha",
        type=float,
        default=0.025,
        help="Initial learning rate."
    )    
    parser.add_argument(
        "-m", "--min_count",
        type=int,
        default=5,
        help="Minimum frequency of accepted words."
    )
    parser.add_argument(
        "-l", "--lemma",
        action="store_true",
        help="Whether tokens are turned into lemmas for training."
    )
    parser.add_argument(
        "-W", "--workers",
        type=int,
        default=3,
        help="Number of worker threads used to train the model."
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=10,
        help="Number of iterations trained over the corpus."
    )
    parser.add_argument(
        "-L", "--logger",
        action="store_true",
        help="Whether to show training progress."
    )
    parser.add_argument(
        "-p", "--progress_per",
        type=int,
        default=1000,
        help="Number of words processed before showing progress."
    )
    args, unknown = parser.parse_known_args()
    run_doc2vec(vector_size=args.vector_size, 
                window=args.window, 
                alpha=args.alpha,
                min_count=args.min_count, 
                lemma=args.lemma, 
                workers=args.workers,
                epochs=args.epochs, 
                logger=args.logger, 
                progress_per=args.progress_per)


if __name__ == "__main__":
    main()