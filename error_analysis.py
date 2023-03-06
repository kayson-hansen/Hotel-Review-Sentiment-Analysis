from web_scraper import file_paths
from load_data import load_dataset
from sentiment_analysis import evaluate_model


data = load_dataset([file_paths[0]])
# reviews to check is an array of tuples with the review text, predicted rating and actual rating
# of eaach misclassified review
misclassified_reviews = []
for i in range(50):
    index = wrong_predictions[i]
    review_text = data[index][1]
    misclassified_reviews.append((review_text, yhat[index], y_cv[index]))

filename = 'misclassified_reviews.txt'
with open(filename, 'a') as f:
    for review in misclassified_reviews:
        f.write(review[0])
    f.close()

filename = 'misclassified_reviews_and_ratings.txt'
with open(filename, 'a') as f:
    for review in misclassified_reviews:
        f.write(review[0] + ', Predicted rating: ' +
                review[1] + ', Actual rating: ' + review[2])
    f.close()
