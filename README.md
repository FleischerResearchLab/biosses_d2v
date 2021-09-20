# biosses-doc2vec: Benchmarking the BIOSSES dataset streamlined!

## What is BIOSSES?

BIOSSES is short for [Biomedical Semantic Similarity Estimation System](https://tabilab.cmpe.boun.edu.tr/BIOSSES/), a series of methods to assess similarity between biomedical sentences proposed by [Soğancıoğlu et al. (2017)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5870675/). Each method in the original BIOSSES produces their own similarity scores and are then benchmarked in terms of the Pearson correlation metric.

The benchmark dataset is a collection of 100 biomedical sentence pairs picked from the [TAC 2014 Biomedical Summarization Track Dataset](https://tac.nist.gov/2014/BiomedSumm/). Similarity between each sentence pair has been assigned integer scores by 5 human expert annotators and included along with the [sentences](https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html). 

## What does **biosses-doc2vec** do?

**biosses-doc2vec** implements the paragraph vector approach ([Le & Mikolov, 2014](https://arxiv.org/pdf/1405.4053.pdf)) to  benchmarking BIOSSES sentences. 

**biosses-doc2vec** uses the Doc2Vec model library from Gensim as it is the only popular open-source implementation of the paragraph vector model in Python as of now ([documentation](https://radimrehurek.com/gensim/models/doc2vec.html)).

**biosses-doc2vec** also implements the training corpus for Doc2Vec –– PubMed Central articles in the Open Access Subset part of which the original BIOSSES's paragraph vectors were trained on. Different levels of granularity are available via a FTP server to download these articles (see [this](https://www.ncbi.nlm.nih.gov/pmc/tools/ftp/)). 
  - **biosses-doc2vec** allows bulk downloads (i.e. not by individual articles) of Commercial and Non-Commercial packages only. 
  
Finally, **`biosses-doc2vec.py`** can be used in the CLI to execute both the training and benchmarking in just one line.

## What are the classes in **biosses-doc2vec** for?

### **BIOSSESDataset**:

Enables downloading and converting biomedical sentence pair and annotator score tables into 2 separate DataFrames. 

Benchmarks a Doc2Vec model via `benchmark_with_d2v` with the Pearson correlation metric.

### **PMCOASubsetCorpus**:

Downloads a corpus that can be either part or all of the PubMed Central Open Access Subset. 
  - Since corpus bulk directories are named after the alphanumeric grouping of journal titles they contain, users can specify by passing to `packages` an iterable of any combination of the following groupings: `0-9A-B` **(default)**, `C-H`, `I-N`, `O-Z`.

  - Users can also choose the exact number of articles to be loaded into the resulting corpus, along with options of lemmatization and turning it into an iterator or a list in memory. 

### **Doc2VecRunner**:

An abstraction to streamline training Doc2Vec on a particular corpus. 
  - `**kwargs` refers to any parameters passed into the instantiation of Doc2Vec [here](https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec), **except** `documents` and `corpus_file`.
