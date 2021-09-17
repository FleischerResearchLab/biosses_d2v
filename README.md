# biosses-doc2vec
## Introduction:
**biosses-doc2vec** is a streamlined module to benchmark the BIOSSES dataset with a configurable Doc2Vec (Paragraph Vector) model.

BIOSSES is short for [Biomedical Semantic Similarity Estimation System](https://tabilab.cmpe.boun.edu.tr/BIOSSES/), a series of methods, including supervised and ontology-based ones (utilizing WordNet and UMLS), to assess similarity between biomedical sentences proposed by [Soğancıoğlu et al. (2017)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5870675/). Each method in the original BIOSSES was devised such that similarity scores are computed and benchmarked with gold standard scores in terms of the Pearson correlation metric.

The dataset in question is a collection of 100 sentence pairs picked from the [TAC 2014 Biomedical Summarization Track Dataset](https://tac.nist.gov/2014/BiomedSumm/) related to biomedical literature only. Similarity between each sentence pair has been judged, assigned integer scores by 5 human expert annotators and also included along with the [sentences](https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html). 

The **biosses-doc2vec** module implements the paragraph vector ([Le & Mikolov, 2014](https://arxiv.org/pdf/1405.4053.pdf)) approach to the benchmark, which was mentioned in the BIOSSES paper. To date, Doc2Vec (Gensim) is the only popular open-source realization of the paragraph vector in Python and hence used as the sole model architecture in this module. Doc2Vec is made configurable with regard to parameters including but definitely not limited to vector size, minimum count of words, alpha rate, etc. (see [documentation](https://radimrehurek.com/gensim/models/doc2vec.html)) Since BIOSSES trained the paragraph vectors on a subset of a subset (called Open Access Subset) of publicly available PubMed Central text data (~4GB, ~37K articles), **biosses-doc2vec** does the same and more. This module allows users to download bulk Open Access packages of their own choosing and pool in any number of articles also of their own choosing. From there, users can opt to use the corpus, lemmatized or not, as an iterator or a list based on their memory and time constraints. There is also an option to run all the training and benchmarking with just one line in the CLI.  

## Classes:
### 1. BIOSSESDataset:
This class represents the BIOSSES dataset itself, starting with enabling users to download and handle biomedical sentence pairs and gold standard annotator scores as 2 separate DataFrames. 
The sentences and scores are stored in tables in Microsoft Word files so [**python-docx**](https://pypi.org/project/python-docx/) was used to parse data from those files. **BIOSSESDataset** also does the benchmarking if the following are supplied: (1) the model, (2) whether the corpus was lemmatized and (3) stopwords removed during training. 

