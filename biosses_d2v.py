import os
import re
import logging
from subprocess import run
from numpy import array
from docx import Document
from nltk.corpus import stopwords
from smart_open import open
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


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
    def __init__(self, lemma=False, stopwords=[], sub_pattern=None):
        self.sub_pattern = sub_pattern 
        # Could try r"(\s+\(*(([À-ÿA-Za-z\s\-.,;&])+\s\(*(\d{4}[a-z]*)+\)*)+)|(\s+\[[\d\s+,;&\[\]]+\])"
        # for getting rid of majority of in-text citations
        self.stopwords = stopwords
        if lemma:
            import spacy        
            self.nlp = spacy.load(name="en_core_sci_sm", 
                                  disable=["parser","ner"])
        

    def __iter__(self):
        return self._iterator()


    def __call__(self, size):
        return self._iterator(size)


    def _iterator(self, size=float("inf")):
        doc_id = 0 
        for paper_path in self.corpus_paths[:size]:
            tagged_doc = self._preprocess_doc(paper_path, doc_id)
            if tagged_doc is not None:
                yield tagged_doc
                doc_id += 1

                
    def _preprocess_doc(self, path, id):
        with open(path, "r", encoding="ascii", errors="ignore") as f:
            paper = f.read()
            if self.sub_pattern is not None:
                paper = re.sub(self.sub_pattern, "", f.read())
            body = re.search("==== Body(.*)==== Refs", paper, re.DOTALL)

            if body is not None:
                doc = body.group(1).lower().split()
                if hasattr(self, "nlp"):
                    doc = " ".join(doc)
                    doc = [token.lemma_ for token in self.nlp(doc) \
                           if not token in self.stopwords]
                else:
                    doc = [token for token in doc \
                           if not token in self.stopwords]
                return TaggedDocument(doc, [id])


    def _append_corpus_paths(self, start_path, path_arr):
        for entry in os.scandir(start_path):
            if not entry.name.startswith(".") \
              and not entry.name == "drive" \
              and not entry.name == "sample_data" \
              and entry.is_dir(follow_symlinks=False):
                  self._append_corpus_paths(entry.path, path_arr)

            if not entry.name.startswith(".") and entry.is_file():
                path_arr.append(entry.path)


    def load(self, subsets=["0-9A-B"]):
        self.corpus_paths = []
        ZIP_NAMES = []
        PMCSC_URL = "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/"

        for subset in subsets:
            ZIP_NAMES.append("non_comm_use." + subset + ".txt.tar.gz")

        for zname in ZIP_NAMES:
            fdir = zname.replace(".txt.tar.gz", "")
            fdir_path = os.path.abspath(fdir)

            if not os.path.isdir(fdir_path):
                os.mkdir(fdir)
                if not os.path.isfile(os.path.abspath(zname)):
                    run(["wget", PMCSC_URL + zname])
                run(["tar", "-xf", zname, "-C", fdir])
            
            self._append_corpus_paths(fdir_path, self.corpus_paths)


class Doc2VecRunner:
    def __init__(self, corpus, **kwargs):
        self.model = Doc2Vec(**kwargs)
        self.corpus = corpus

    def run(self, 
            use_logger=False, 
            log_dir=None, 
            log_fname=None, 
            progress_per=None):
        if use_logger:
            if log_dir is None:
                log_dir = "model"
            if log_fname is None:
                log_fname = "model.log"
            if not os.path.isdir(log_dir):
                os.mkdir(log_dir)
            logpath = os.path.join(log_dir, log_fname)
            logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",
                                level=logging.INFO,
                                handlers=[
                                    logging.FileHandler(logpath, mode="a"),
                                    logging.StreamHandler()
                                ])    

        if progress_per is not None:
            self.model.build_vocab(self.corpus, progress_per=progress_per)
        else:
            self.model.build_vocab(self.corpus)
        self.model.train(self.corpus, 
                         total_examples=self.model.corpus_count, 
                         epochs=self.model.epochs)


    def save_model(self, model_dir=None, model_fname=None):
        if model_dir is None:
            model_dir = "model"
        if model_fname is None:
            model_fname = "model.gsm"
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        savepath = os.path.join(model_dir, model_fname)
        self.model.save(savepath)

  
    def get_model(self):
        return self.model


def run_doc2vec(iterator, lemma, use_logger, save_model, progress_per, **kwargs):
    corpus = PMCSubsetCorpus(lemma=lemma)
    corpus.load()
    if iterator:
        corpus = corpus(10)
    else:
        corpus = list(corpus(10))

    model_dir = "biod2v_{}lemma_{}iter".format("!" if not lemma else "", 
                                               "!" if not iterator else "")
    runner = Doc2VecRunner(corpus, **kwargs)
    runner.run(use_logger, model_dir, progress_per=progress_per)
    if save_model:
        runner.save_model(model_dir)


def main():

    import argparse
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
        help="Maximum distance between the current and predicted word \
              during training."
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
        "-p", "--progress_per",
        type=int,
        default=1000,
        help="Number of words processed before showing progress."
    )
    parser.add_argument(
        "-i", "--iterator",
        action="store_true",
        help="Whether data is streamed in using an iterator or \
              stored in memory in a list object."
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
        "-L", "--use_logger",
        action="store_true",
        help="Whether to show training progress."
    )
    parser.add_argument(
        "-s", "--save_model",
        action="store_true",
        help="Whether to save the model weights."
    )    
    args, _ = parser.parse_known_args()
    run_doc2vec(iterator=args.iterator,
                lemma=args.lemma, 
                use_logger=args.use_logger, 
                save_model=args.save_model,
                progress_per=args.progress_per,
                vector_size=args.vector_size, 
                window=args.window, 
                alpha=args.alpha,
                min_count=args.min_count, 
                workers=args.workers,
                epochs=args.epochs)


if __name__ == "__main__":
    main()