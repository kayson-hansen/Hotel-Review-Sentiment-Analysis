import csv
import numpy as np
import spacy


def load_dataset(files):
    labels = []
    reviews = []
    for filename in files:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # only include 1 star, 2 star, 4 star, and 5 star reviews
                if row[0] == '40' or '50':
                    labels.append(1)
                    reviews.append(row[1])
                elif row[0] == '10' or '20':
                    labels.append(0)
                    reviews.append(row[1])
            f.close()
    return reviews, labels


def get_inputs_and_outputs(filenames):
    review_texts, review_scores = load_dataset(filenames)
    nlp = spacy.load('en_core_web_sm')
    m = len(review_scores)
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
        # can also use sum or max of word embeddings to find sentence embedding
        mean_embedding = np.mean(word_embeddings, axis=0)
        X[i] = mean_embedding
        Y[i] = review_scores[i]

    return X, Y
