import os
import re
from subprocess import run
from smart_open import open
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class BIOSSESDataset:
    """
    Dataset class for the BIOSSES Benchmark.
    The BIOSSES dataset includes 100 biomedical sentence pairs from
    the TAC 2014 Biomedical Summarization Track (https://tac.nist.gov/2014/BiomedSumm/) 
    whose similarity was judged by 5 different human experts according to 
    the SemEval 2012 Task 6 Guideline on a scale of 0-4.
    This class can prepare the BIOSSES dataset, both the sentence pairs
    and similarity scores, in array form.
    Source: https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html.
    """
    
    def __init__(self):
        """Downloads BIOSSES data if current directory does not contain it."""

        if not os.path.isfile("BIOSSES-Dataset/Annotation-Pairs.docx") \
           or not os.path.isfile("BIOSSES-Dataset/Annotator-Scores.docx"):
            run(["wget", "https://tabilab.cmpe.boun.edu.tr/\
                          BIOSSES/Downloads/BIOSSES-Dataset.rar"])
            run(["unrar", "x", "BIOSSES-Dataset.rar"])


    def _get_array_from_doc(self, path):
        """Loads a table from a .doc/docx file into an array.
      
        Args: 
          path: Filepath of the document.

        Returns:
          Array obtained from the file. 
        """

        from docx import Document
        from numpy import array

        doc = Document(os.path.abspath(path))
        table = doc.tables[0]
        arr = array([[cell.text.strip() for cell in row.cells] \
                           for row in table.rows])
        return arr

 
    def get_sentence_pairs(self):
        """Gets the BIOSSES sentence pairs for benchmarking."""

        sent_arr = self._get_array_from_doc("BIOSSES-Dataset/Annotation-Pairs.docx")
        return sent_arr[1:,1:]

    
    def get_annotator_scores(self):
        """Gets the BIOSSES similarity scores by expert annotators."""

        score_arr = self._get_array_from_doc("BIOSSES-Dataset/Annotator-Scores.docx")
        return score_arr[1:,1:].astype("i1")


class PMCOASubsetCorpus:
    """
    Corpus iterator for the PMC Open Access (PMCOA) Subset in plain text form.
    PMCOA Subset is a corpus of select PubMed Central articles made free for 
    research purposes and is accessible via FTP server (ftp://ftp.ncbi.nlm.nih.gov/pub/pmc).
    The iterator generates such text data through the oa_bulk directory, from
    both usage groups: commercial and non-commercial use.
    Source: https://www.ncbi.nlm.nih.gov/pmc/tools/ftp/.
    """
    def __init__(self, 
                 packages=["0-9A-B"], 
                 size=float("inf"), 
                 lemma=False, 
                 stopwords=[], 
                 sub_pattern=None):
        """Loads user-defined corpus packages.
        User can determine the number of articles from all or any combination 
        of the following packages: 0-9A-B, C-H, I-N, O-Z.
        Preprocessing is optional in the following ways:
          - lemmatization
          - stopword removal
          - regex substitution
        """

        self.packages = packages
        self.size = size
        self.lemma = lemma
        self.stopwords = stopwords
        self.sub_pattern = sub_pattern
        # Could try r"(\s+\(*(([À-ÿA-Za-z\s\-.,;&])+\s\(*(\d{4}[a-z]*)+\)*)+)|(\s+\[[\d\s+,;&\[\]]+\])"
        # for getting rid of majority of in-text citations
        self.corpus_paths = self._load()


    def __iter__(self):
        """Enables users to treat this object as an iterable."""

        return self._iterator()


    def _iterator(self):
        """Creates a generator of preprocessed articles from the PMCOA Subset."""

        doc_id = 0
        while doc_id < self.size:
            tagged_doc = self._preprocess_doc(self.corpus_paths[doc_id], doc_id)
            if tagged_doc is not None:
                yield tagged_doc
                doc_id += 1

                
    def _preprocess_doc(self, path, id):
        """Tokenizes text from a file.
        
        Args:
          path: Filepath of the article to be preprocessed.
          id: Article ID number required for training Doc2Vec.
        
        Returns:
          Tokens in the form of a TaggedDocument,
          following Doc2Vec's training protocol.
        """

        with open(path, "r", encoding="ascii", errors="ignore") as f:
            paper = f.read()
            if self.sub_pattern is not None:
                paper = re.sub(self.sub_pattern, "", f.read())
            body = re.search("==== Body(.*)==== Refs", paper, re.DOTALL)

            if body is not None:
                doc = body.group(1).lower().split()
                if self.lemma:
                    doc = " ".join(doc)
                    doc = [token.lemma_ for token in self.nlp(doc) \
                           if not token in self.stopwords]
                else:
                    doc = [token for token in doc \
                           if not token in self.stopwords]
                return TaggedDocument(doc, [id])


    def _append_corpus_paths(self, start_path, path_arr):
        """Recursively scans and appends paths of all .txt files 
        from a directory and its subdirectories, if any, to a provided array. 

        Args:
          start_path: Directory path to be scanned for .txt files.
          path_arr: Array to append all the .txt paths.
        """

        for entry in os.scandir(start_path):
            if not entry.name.startswith(".") \
               and entry.is_dir(follow_symlinks=False):
                self._append_corpus_paths(entry.path, path_arr)

            if entry.name.endswith(".txt"):
                path_arr.append(entry.path)


    def _load(self):
        """Downloads the zipped corpus folders and unzips
        them if they are not already available in the current directory.
        Then loads individual filepaths of all .txt files into 
        an internal array, necessary for the iterator later.
        """

        if self.lemma:
            import spacy        
            self.nlp = spacy.load(name="en_core_sci_sm", 
                                  disable=["parser","ner"])
        corpus_paths = []

        for package in self.packages:
            zipnames = ["non_comm_use." + package + ".txt.tar.gz",
                        "comm_use." + package + ".txt.tar.gz"]
            
            for zname in zipnames:
                dirname = zname.replace(".txt.tar.gz", "")
                if not os.path.isdir(dirname):
                    os.mkdir(dirname)
                    if not os.path.isfile(zname):
                        run(["wget", 
                            "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/" 
                            + zname])
                    run(["tar", "-xf", zname, "-C", dirname])
                self._append_corpus_paths(dirname, corpus_paths)
            
        return corpus_paths


class Doc2VecRunner:
    """
    Runner to train Doc2Vec from the Gensim library on a corpus.
    This is just an abstraction to streamline Doc2Vec's configurations.
    It enables training logs and model saving if need be. 
    Doc2Vec's documentation: https://radimrehurek.com/gensim/models/doc2vec.html.
    """

    def __init__(self, corpus=None, **kwargs):
        """Initializes internal model and corpus"""

        self.model = Doc2Vec(**kwargs)
        self.corpus = corpus


    def run(self, 
            use_logger=False, 
            log_dir=None, 
            progress_per=None):
        """Trains the Doc2Vec model, with optional progress logging"""

        if use_logger:
            import logging
            log_dir = "model" if log_dir is None else log_dir
            log_fname = "{}.log".format(log_dir)
            if not os.path.isdir(log_dir):
                os.mkdir(log_dir)
            logpath = os.path.join(log_dir, log_fname)
            logging.basicConfig(format="%(asctime)s : \
                                        %(levelname)s : \
                                        %(message)s",
                                level=logging.INFO,
                                handlers=[
                                    logging.FileHandler(logpath, 
                                                        mode="a"),
                                    logging.StreamHandler()
                                ])    

        if progress_per is not None:
            self.model.build_vocab(self.corpus, 
                                   progress_per=progress_per)
        else:
            self.model.build_vocab(self.corpus)
        self.model.train(self.corpus, 
                         total_examples=self.model.corpus_count, 
                         epochs=self.model.epochs)


    def save_model(self, model_dir=None):
        """Saves model's internal weights to directory"""

        model_dir = "model" if model_dir is None else model_dir
        model_fname = "{}.gsm".format(model_dir)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        savepath = os.path.join(model_dir, model_fname)
        self.model.save(savepath)

  
    def get_model(self):
        return self.model


def run_doc2vec(iterator, 
                lemma, 
                use_logger, 
                save_model, 
                package_names, 
                corpus_size, 
                progress_per, 
                **kwargs):
    """Program to run based on CLI arguments.
    Helps train a configurable Doc2Vec model on a configurable corpus.
    If the model is saved, the default saving directory is determined 
    based on whether lemmatization or data iteration is enabled.
    """

    corpus = PMCOASubsetCorpus(packages=package_names, size=1000, lemma=lemma)
    corpus = list(corpus) if not iterator else corpus

    model_dir = "biod2v_{}lemma_{}iter".format("!" if not lemma else "", 
                                               "!" if not iterator else "")
    runner = Doc2VecRunner(corpus, **kwargs)
    runner.run(use_logger, model_dir, progress_per=progress_per)
    if save_model:
        runner.save_model(model_dir)


def main():
    """
    Parsing arguments to power Doc2Vec with the PMCOA Subset corpus
    if user wants to use this module in the CLI.
    """    
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--iterator",
        action="store_true",
        help="Whether data is streamed in using an iterator or \
              stored in memory in a list object."
    )
    parser.add_argument(
        "-l", "--lemma",
        action="store_true",
        help="Whether tokens are turned into lemmas for training."
    )
    parser.add_argument(
        "-L", "--use-logger",
        action="store_true",
        help="Whether to show training progress."
    )
    parser.add_argument(
        "-S", "--save-model",
        action="store_true",
        help="Whether to save the model weights."
    )    
    parser.add_argument(
        "-n", "--package-names",
        type=str,
        default=["0-9A-B"],
        help="Names of PMC Open Access packages used for training."
    )
    parser.add_argument(
        "-s", "--corpus-size",
        type=float,
        default=float("inf"),
        help="Size (number of articles) of the training corpus."
    )
    parser.add_argument(
        "-p", "--progress-per",
        type=int,
        default=1000,
        help="Number of words processed before showing progress."
    )
    parser.add_argument(
        "-v", "--vector-size",
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
        "-m", "--min-count",
        type=int,
        default=5,
        help="Minimum frequency of accepted words."
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
    args, _ = parser.parse_known_args()
    run_doc2vec(iterator=args.iterator,
                lemma=args.lemma, 
                use_logger=args.use_logger, 
                save_model=args.save_model,
                package_names=args.package_names,
                corpus_size=args.corpus_size,
                progress_per=args.progress_per,
                vector_size=args.vector_size, 
                window=args.window, 
                alpha=args.alpha,
                min_count=args.min_count, 
                workers=args.workers,
                epochs=args.epochs)


if __name__ == "__main__":
    main()