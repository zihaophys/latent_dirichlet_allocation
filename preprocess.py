import numpy as np

import string
import re 


def preprocessing(M=100):

    with open('ap/ap.txt', 'r') as f:
        files = f.readlines()

    corpus = list(doc for doc in files if ('<DOC>\n' not in doc) 
                                        and ('</DOC' not in doc) 
                                        and ('TEXT>' not in doc))
    corpus = corpus[0:M]
    

    with open('ap/vocab_process.txt', 'r') as f:
        files = f.readlines()
    vocab = [word.strip() for word in files ]
    vocab = list(set(vocab))

    corpus_processed = []
    for doc in corpus:
        tmp = doc.strip().lower().translate(str.maketrans('', '', string.punctuation))
        tmp = re.sub('[0-9]+', '', tmp)
        doc_processed = [word for word in tmp.split() if word in vocab]
        doc_matrix = np.zeros((len(doc_processed), len(vocab)))
        for (id, word) in enumerate(doc_processed):
            doc_matrix[id, vocab.index(word)] = 1
        doc_matrix = doc_matrix.astype(int)
        corpus_processed.append(doc_matrix)

    return corpus_processed, vocab