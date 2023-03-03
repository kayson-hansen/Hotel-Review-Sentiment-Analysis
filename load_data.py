import csv
import numpy as np
import spacy


def load_dataset(filename):
    labels = []
    reviews = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == '40' or '50':
                labels.append(1)
                reviews.append(row[1])
            elif row[0] == '10' or '20':
                labels.append(0)
                reviews.append(row[1])

    return reviews, labels


def get_inputs_and_outputs(filename):
    review_texts, review_scores = load_dataset(filename)
    nlp = spacy.load('en_core_web_md')
    m = 10000
    n = len(nlp('hotel').vector)

    X = np.zeros((m, n))
    Y = np.zeros((m, 1))

    for i in range(m):
        tokens = nlp(review_texts[i])
        # use the mean of all the word embeddings as the total review embedding
        word_embeddings = []
        for token in tokens:
            if not token.is_stop:
                word_embeddings.append(token.vector)
        mean_embedding = np.mean(word_embeddings, axis=0)
        X[i] = mean_embedding
        Y[i] = review_scores[i]

    return X, Y
