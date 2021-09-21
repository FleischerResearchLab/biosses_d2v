# biosses-doc2vec: Benchmarking the BIOSSES dataset with Doc2Vec streamlined!

## What is BIOSSES?

BIOSSES is short for [Biomedical Semantic Similarity Estimation System](https://tabilab.cmpe.boun.edu.tr/BIOSSES/), a series of methods to assess similarity between biomedical sentences proposed by [Soğancıoğlu et al. (2017)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5870675/). Each method in the original BIOSSES produces their own similarity scores and are then benchmarked in terms of the Pearson correlation metric.

The benchmark dataset is a table of 100 biomedical sentence pairs picked from the [TAC 2014 Biomedical Summarization Track Dataset](https://tac.nist.gov/2014/BiomedSumm/). Each sentence pair has been assigned integer similarity scores by 5 human expert annotators which are included in another table along with the [sentences](https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html). 

## What does **biosses-doc2vec** do?

**biosses-doc2vec** implements the paragraph vector approach ([Le & Mikolov, 2014](https://arxiv.org/pdf/1405.4053.pdf)) to benchmarking BIOSSES sentences. 

**biosses-doc2vec** uses the Doc2Vec model library from Gensim as it is the only popular open-source implementation of the paragraph vector model in Python as of now ([documentation](https://radimrehurek.com/gensim/models/doc2vec.html)).

**biosses-doc2vec** also implements the training corpus for Doc2Vec, which is the PubMed Central Open Access (PMCOA) Subset of biomedical papers – part of which the original BIOSSES paragraph vectors were trained on. Different levels of granularity are available via a FTP server to download corpus text (see [this](https://www.ncbi.nlm.nih.gov/pmc/tools/ftp/)); **biosses-doc2vec** enforces bulk downloads (i.e. not by individual papers) of Commercial and Non-Commercial packages.
  > **biosses-doc2vec** treats each paper from the PMCOA Subset as a document with a vector to train.
  
Finally, **`biosses-doc2vec.py`** can be used in the CLI to execute both the training and benchmarking with just one line of code.

## What are the classes in **biosses-doc2vec** for?

### **PMCOASubsetCorpus**:

Downloads a corpus that can be either part or all of the PubMed Central Open Access Subset.

  > Corpus bulk directories to be downloaded are specified by the `packages` parameter. Since they are named after the alphanumeric grouping of journal titles they contain, an iterable of any combination of the following groupings is valid: `0-9A-B` **(default)**, `C-H`, `I-N`, `O-Z`.

Customizes characteristics of the corpus.

  > These include exact number of papers to be loaded into the resulting corpus; lemmatized or not; iterator or list in memory; stopwords to remove; regex pattern to be rid of.

Stores paths to downloaded papers internally.

### **BIOSSESDataset**:

Enables downloading and converting biomedical sentence pair and annotator score tables into 2 separate DataFrames via `get_sentence_df` and `get_score_df`. 

Benchmarks a Doc2Vec model via `benchmark_with_d2v` with the Pearson correlation metric.

### **Doc2VecRunner**:

An abstraction to streamline training Doc2Vec on a particular corpus. 
  > `**kwargs` refers to any parameters passed into the instantiation of Doc2Vec [here](https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec), **except** `documents` and `corpus_file`.

## Requirements:

It is recommended that a virtual environment be created and activated before installing any required libraries:

```python
python3 -m venv venv
source venv/bin/activate
```

Then install the requirements as follows:

```python
pip install -r requirements.txt
```

## CLI Demo:

![biosses-d2v demo](https://media.giphy.com/media/kWsp4ghLZYTtjvSxCh/source.gif?cid=790b76119fa15a6b2f4d386556779cd8dccef60873bc24b5&rid=source.gif&ct=g)'

## [Interactive Demo](./biosses_d2v_demo.ipynb)

## Areas for improvement:

1. Optimized parameters for the default CLI command.
2. Data structures to store training corpus.
3. Text preprocessing. 
  > Lemmatization engine (currenty using scispacy); stopword choices; regex patterns to remove unwanted features (could try r"(\s+\(*(([À-ÿA-Za-z\s\-.,;&])+\s\(*(\d{4}[a-z]*)+\)*)+)|(\s+\[[\d\s+,;&\[\]]+\])" for getting rid of majority of in-text citations), etc.
4. Should a `TaggedDocument` be another unit of text but an entire paper like now? Maybe try a single paragraph?
5. Other metrics to benchmark by.
6. Other corpora to train Doc2Vec on.
7. Other Doc2Vec implementations.
  