import nltk
import re
import string
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from nltk.tokenize import word_tokenize as nltk_word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
from scipy.spatial.distance import cosine


def Similarity(v1, v2):
    score = 0.0
    if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
        score = ((1 - cosine(v1, v2)) + 1) / 2
    return score


def WordVectors(sentences, embedding_model):
    word_vectors = dict()
    for sent in sentences:
        words = nltk_word_tokenize(sent)
        for w in words:
            word_vectors.update({w: embedding_model.wv[w]})
    return word_vectors


def SentTokenize(text):
    sents = nltk_sent_tokenize(text)
    sents_filtered = []
    for s in sents:
        sents_filtered.append(s)
    return sents_filtered


def SentProcessing(text):
    stop_words = set(stopwords.words('english'))
    sentences = SentTokenize(text)
    sentences_cleaned = []
    for sent in sentences:
        words = nltk_word_tokenize(sent)
        words = [w for w in words if w not in string.punctuation]
        words = [w for w in words if not w.lower() in stop_words]
        words = [w.lower() for w in words]
        sentences_cleaned.append(" ".join(words))
    return sentences_cleaned


def GetTFIDF(sentences):
    vectorizer = CountVectorizer()
    sent_word_matrix = vectorizer.fit_transform(sentences)
    transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
    tfidf = transformer.fit_transform(sent_word_matrix)
    tfidf = tfidf.toarray()

    centroid_vector = tfidf.sum(0)
    centroid_vector = np.divide(centroid_vector, centroid_vector.max())

    feature_names = vectorizer.get_feature_names()

    relevant_vector_indices = np.where(centroid_vector > 0.3)[0]
    word_list = list(np.array(feature_names)[relevant_vector_indices])  # get centroid words
    return word_list


def EmbeddingRep(words, word_vectors, embedding_model):
    embedding_representation = np.zeros(embedding_model.vector_size, dtype="float32")  # vocabulary size of text
    word_vectors_keys = set(word_vectors.keys())  # alphabetical order or vocabulary in word form (keys of word vectors)
    count = 0
    for w in words:
        if w in word_vectors_keys:  # lookup against key to extract the correct corresponding word vector
            embedding_representation = embedding_representation + word_vectors[w]  # sum of loop
            count += 1
        if count != 0:
            embedding_representation = np.divide(embedding_representation, count)
        return embedding_representation
