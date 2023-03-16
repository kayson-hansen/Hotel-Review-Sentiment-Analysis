import csv
import numpy as np
from web_scraper import file_paths
import matplotlib.pyplot as plt
from sentiment_analysis import training_loop


def find_num_ratings_per_star(files):
    """Returns a dict of each star rating (1 to 5 stars) and a count of how many reviews have that many stars

    Args:
        files (List): List with each filename (assume they're csv files) to be read from

    Returns:
        Dict: dictionary of star ratings with how many reviews have each star rating
    """
    reviews = {}
    for filename in files:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # we divide by 10 becauase the csv files record 10 for 1 star reviews, 20 for 2 star reviews, and so on
                reviews[int(row[0])/10] = reviews.get(int(row[0])/10, 0) + 1
            f.close()
    return reviews


def find_average_rating(reviews):
    """Calculates the average rating given a count of how many reviews have each number of stars

    Args:
        reviews (Dict): dictionary of star ratings with how many reviews have each star rating

    Returns:
        Float: average review rating
    """
    sum_ratings = 0
    num_reviews = 0
    for rating in reviews:
        sum_ratings += rating * reviews[rating]
        num_reviews += reviews[rating]
    return sum_ratings / num_reviews


def find_standard_deviation(reviews):
    """Calculates the standard deviation of all the review ratings

    Args:
        reviews (Dict): dictionary of star ratings with how many reviews have each star rating

    Returns:
        Float: standard deviation of review ratings
    """
    ratings = []
    for rating in reviews:
        ratings += reviews[rating] * [rating]
    return np.std(ratings)


def find_basic_stats():
    reviews = find_num_ratings_per_star(file_paths)
    print("Number of reviews per star rating: ", reviews)
    print("Average rating: ", find_average_rating(reviews))
    print("Standard deviation: ", find_standard_deviation(reviews))

    plt.bar(reviews.keys(), reviews.values(), color='b')
    plt.title('Rating star distribution')
    plt.xlabel('Number of stars')
    plt.ylabel('Number of reviews')
    plt.show()


# model can be 'neural network' or 'logistic regression'
model = 'neural network'
# embedding can be 'doc2vec' or 'mean'
embedding = 'doc2vec'
# output can be 'softmax' or 'binary'
output = 'softmax'
# metric can be 'learning rate', 'num epochs', or 'batch size'
metric = 'num epochs'
# TODO: add regularization metric

# trains the model of your choice over different values of a given metric
if embedding == 'doc2vec' and output == 'softmax':
    input_file = 'Embeddings/MulticlassDoc2VecEmbeddings.npy'
    scores = training_loop(input_file, file_paths, model, softmax=True,
                           metric=metric)
elif embedding == 'doc2vec' and output == 'binary':
    input_file = 'Embeddings/Doc2VecEmbeddings.npy'
    scores = training_loop(input_file, file_paths, model, softmax=False,
                           metric=metric)
elif embedding == 'mean' and output == 'softmax':
    input_file = 'Embeddings/MulticlassMeanWordEmbeddings.npy'
    scores = training_loop(input_file, file_paths, model, softmax=True,
                           metric=metric)
else:
    input_file = 'Embeddings/MeanWordEmbeddings.npy'
    scores = training_loop(input_file, file_paths, model, softmax=False,
                           metric=metric)

if metric == 'learning rate' or metric == 'batch size':
    plt.xscale('log')
x = list(scores.keys())
y = list(scores.values())
plt.plot(x, y)
plt.xticks(x, x)
plt.title('Accuracy vs ' + metric)
plt.xlabel(metric)
plt.ylabel('Accuracy')
plt.show()
