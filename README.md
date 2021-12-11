# Latent Dirichlet Allocation

Code for HKUST MATH 5472 Final Project

Python implementation of LDA based on [Latent Dirichlet Allocation](http://www.cs.columbia.edu/~blei/lda-c/)

### Usage

```python
from preprocess import *
from main import *
```

```python
corpus = preprocessing(M=200)
```

`preprocess.py` provides a text-preprocessing for American Press corpus, returns a list where each element represents a document coded by {0,1}. Preprocess the first *M* documents in AP corpus.

```python
alpha, beta = LDA.parameter_estimation(corpus, k=10, tol=1e-6, max_iter=100)
```

`LDA.parameter_estimation` performs variantial inference EM to estimate Dirichlet parameter alpha, and word probability beta. Number of topics *k* should be given.

### Examples

Check the notebooks  `sim_data.ipynb` and `ap_modeling.ipynb` to play the examples in report lda.pdf.

