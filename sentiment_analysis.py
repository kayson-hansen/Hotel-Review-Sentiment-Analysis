from load_data import get_inputs_and_outputs


file_paths = ["/users/kaysonhansen/cs129/HotelReviewData/VenetianHotelReviews.csv",
              "/users/kaysonhansen/cs129/HotelReviewData/MirageHotelReviews.csv",
              "/users/kaysonhansen/cs129/HotelReviewData/MandalayBayHotelReviews.csv"]

X, Y = get_inputs_and_outputs(file_paths)
m = X.shape[0]
n = X.shape[1]

# splits are 60/20/20 train/cross-validation/test
m1 = int(m * 3/5)
m2 = int(m * 4/5)
x_train = X[:m1, :]
x_cv = X[m1:m2, :]
x_test = X[m2:m, :]
y_train = Y[:m1, :]
y_cv = Y[m1:m2, :]
y_test = Y[m2:m, :]

print(m)
print(n)
