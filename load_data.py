import csv
import numpy as np
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import string
from pathlib import Path


def load_dataset(files, multiclass):
    """ Given a list of files containing reviews and ratings, return whether each review
    had a positive sentiment (4 or 5 stars) or a negative sentiment (1 or 2 stars). 
    Discard the 3 star reviews. Called by get_mean_embedding_inputs and get_outputs.

    Args:
        files (List[String]): list of files containing input reviews and ratings
        multiclass (Bool): true if we want to return the rating out of 5, false if we want to
        return a positive or negative sentiment (1 or 0, respectively)

    Returns:
        (List, List): a list (reviews) containing the text for each 1, 2, 4, or 5 star review,
        and a list (labels) containing 1s or 0s corresponding to positive or negative sentiment
    """
    reviews = []
    labels = []
    for filename in files:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if multiclass:
                    labels.append(int(row[0])/10 - 1)
                    reviews.append(row[1])
                else:
                    # only include 1 star, 2 star, 4 star, and 5 star reviews
                    if row[0] == '40' or row[0] == '50':
                        labels.append(1)
                        reviews.append(row[1])
                    elif row[0] == '10' or row[0] == '20':
                        labels.append(0)
                        reviews.append(row[1])
            f.close()
    return reviews, labels


def get_mean_embedding_inputs(filenames, multiclass):
    """ From a set of files containing reviews and ratings, generate sentence embeddings by taking
    the mean of all the word embeddings in the reviews and store them in a numpy matrix. Called 
    in the generate_embeddings.py file.

    Args:
        filenames (List[String]): list of files containing input reviews and ratings
        multiclass (Bool): true if we want to return the rating out of 5, false if we want to
        return a positive or negative sentiment (1 or 0, respectively)

    Returns:
        (np.ndarray): vector of sentence embeddings, one for each input review, dimension num_inputs x num_features
    """
    reviews, ratings = load_dataset(filenames, multiclass)
    nlp = spacy.load('en_core_web_sm')
    m = len(reviews)
    n = len(nlp('hotel').vector)

    X = np.zeros((m, n))

    for i in range(m):
        # measure the progress over time
        if i % 1000 == 0:
            print(i)
        tokens = nlp(reviews[i])
        # use the mean of all the word embeddings as the total review embedding
        word_embeddings = []
        for token in tokens:
            if not token.is_stop:
                word_embeddings.append(token.vector)
        # can also use sum or max of word embeddings to find sentence embedding
        mean_embedding = np.mean(word_embeddings, axis=0)
        X[i] = mean_embedding

    return X


def get_doc2vec_inputs(filenames, multiclass):
    """ From a set of files containing reviews and ratings, generate sentence embeddings using
    doc2vec for each review and store the embeddings in a numpy matrix. Called in the 
    generate_embeddings.py file.

    Args:
        filenames (List[String]): list of files containing input reviews and ratings
        multiclass (Bool): true if we want to return the rating out of 5, false if we want to
        return a positive or negative sentiment (1 or 0, respectively)

    Returns:
        (np.ndarray): vector of sentence embeddings, one for each input review, dimension num_inputs x num_features
    """
    num_features = 50
    reviews, ratings = load_dataset(filenames, multiclass)
    m = len(reviews)
    n = num_features

    # strip reviews of punctuation and split them into lists of tokens
    examples = []
    train_corpus = []
    for i in range(len(reviews)):
        review = reviews[i].translate(
            str.maketrans('', '', string.punctuation))
        tokens = review.lower().split(' ')
        examples.append(tokens)
        train_corpus.append(TaggedDocument(tokens, [i]))

    # check if we've already trained and saved the model
    model_filename = 'doc2vec/doc2vec_model'
    if Path(model_filename).is_file():
        model = Doc2Vec.load(model_filename)
    # if we don't already have a trained model, build a vocab and train one
    else:
        model = Doc2Vec(vector_size=num_features, min_count=2, epochs=10)
        model.build_vocab(train_corpus)
        print("Finished building vocab!")

        model.train(train_corpus, total_examples=model.corpus_count,
                    epochs=model.epochs)
        print("Finished training model!")
        model.save(model_filename)

    X = np.zeros((m, n))
    for i in range(m):
        # measure the progress over time
        if i % 1000 == 0:
            print(i)
        X[i] = model.infer_vector(examples[i])
    return X


def get_outputs(filenames, multiclass):
    """ From a list of filenames containing reviews and ratings, store whether each
    review contains positive or negative sentiment in a numpy array. Called by 
    create_inputs_and_outputs in the sentiment_analysis.py file.

    Args:
        filenames (List[String]): list of files containing input reviews and ratings
        multiclass (Bool): true if we want to return the rating out of 5, false if we want to
        return a positive or negative sentiment (1 or 0, respectively)

    Returns:
        (np.ndarray): vector of output sentiments, dimension num_inputs x 1
    """
    review_texts, review_scores = load_dataset(filenames, multiclass)
    m = len(review_scores)
    Y = np.zeros((m, 1))
    for i in range(m):
        Y[i] = review_scores[i]

    return Y
