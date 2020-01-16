# coding=utf-8
import codecs
import os
import gensim
import numpy as np
from sklearn.cluster import MeanShift
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

STD_THRESHOLD = 5
BANDWIDTH = 5
VECTOR_SIZE = 300


def get_documents(path):
    for root, firs, files in os.walk(path):
        for filename in files:
            with open(os.path.join(root, filename), 'r', encoding='utf8', errors='ignore')as f:
                yield f.read().split()


def _std(matrix):
    matrix = np.mat(matrix)
    a = np.sum(matrix, axis=0)
    m, n = np.shape(matrix)
    e = a / m
    return np.float(np.sqrt(np.sum(np.dot(matrix[i] - e, (matrix[i] - e).T) for i in range(m)) / (n - 1)))


def save_model(filepath, directory):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(get_documents(directory))]
    model = Doc2Vec(documents, vector_size=VECTOR_SIZE, window=2, min_count=1, workers=4)
    table = {}
    mapper = {}
    for i in range(20):
        model.train(documents, epochs=model.iter, total_examples=model.corpus_count)
    for tagged_document in documents:
        tags = tagged_document.tags
        words = tagged_document.words
        for tag in tags:
            vec = model.docvecs[tag]
            for word in words:
                if word not in table:
                    table[word] = []
                table[word].append(vec)
    for k, v in table.items():
        ms = MeanShift(bandwidth=BANDWIDTH).fit(np.array(v))
        pairs = zip(v, ms.cluster_centers_, ms.labels_)
        cluster = {}
        for origin, center, label in pairs:
            if label not in cluster:
                cluster[label] = {
                    'center': center,
                    'neighbor': []
                }
            cluster[label]['neighbor'].append(origin)
        vectors = []
        for point in cluster.values():
            center = point['center']
            neighbor = point['neighbor']
            if _std(neighbor) < STD_THRESHOLD:
                vectors.append(center)
        for index, vector in enumerate(vectors, start=1):
            mapper[k + '_' + str(index)] = vector
    with open(filepath, 'w', encoding='utf8')as f:
        f.write('%d %d\n' % (len(mapper), VECTOR_SIZE))
        for word, item in mapper.items():
            f.write('%s %s\n' % (word, ' '.join(map(str, item))))
