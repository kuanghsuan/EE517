

import os
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np


with open("list1.txt") as textFile:
    lines = [line.split() for line in textFile]

##print(lines)

from gensim.models import Word2Vec
from gensim.models import KeyedVectors


# # Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# sentence = ["London", "is", "the", "capital"]
# vectors = [model[w] for w in sentence]
# vectors = np.array(vectors)
# dog = model['dog']
# vec = model[lines[0][0]]
#print(vectors.shape)
# print(dog[:10])
for line in lines:
    vectors = [model[w] for w in line]
    va, vb, vc = vectors[0:3]
    ground_truth = vectors[3]
    vd = vc + (vb-va)
    vd = np.array(vd, dtype='f')
    top3 = model.most_similar(positive=[vd], topn=3)
    print(line[3])
    print(top3)
    print('-----------')

