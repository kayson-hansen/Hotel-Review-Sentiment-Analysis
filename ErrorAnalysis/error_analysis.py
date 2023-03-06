from web_scraper import file_paths
from load_data import load_dataset
from sentiment_analysis import create_inputs_and_outputs, train_neural_network_model


input_filename = "Embeddings/AllEmbeddings.npy"
X, Y = create_inputs_and_outputs(input_filename, file_paths, shuffle=False)
m = X.shape[0]

# splits are 80/10/10 train/cross-validation/test
m1 = int(m * 8/10)
m2 = int(m * 9/10)
x_train = X[:m1, :]
x_cv = X[m1:m2, :]
y_train = Y[:m1, :]
y_cv = Y[m1:m2, :]

model = train_neural_network_model(x_train, y_train)

# number of examples to perform manual error analysis on
n = 200

manual_x = x_train[:n]
manual_y = y_train[:n]
manual_pred = model.predict(manual_x)

# find the misclassified reviews in the first n reviews
reviews, labels = load_dataset([file_paths[0]])
misclassified_reviews = []
for i in range(n):
    y = manual_y[i]
    if manual_pred[i] >= 0.5:
        yhat = 1
    else:
        yhat = 0
    if y != yhat:
        misclassified_reviews.append((reviews[i], y, yhat))

# write the misclassified reviews, along with the incorrect prediction and the
# actual sentiment value, to a file
filename = 'misclassified_reviews_and_ratings.txt'
with open(filename, 'a') as f:
    for review in misclassified_reviews:
        f.write('Review: ' + review[0] + '\n, Actual rating: ' +
                str(review[1]) + ', Predicted rating: ' + str(review[2]) + '\n\n')
    f.close()
