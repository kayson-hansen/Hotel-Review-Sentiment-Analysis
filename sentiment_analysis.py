from load_data import get_inputs_and_outputs
from web_scraper import file_paths
import pandas as pd

# combine all hotel reviews into one csv file
combined_csv = pd.concat([pd.read_csv(f) for f in file_paths])
combined_csv.to_csv("AllHotelReviews.csv", index=False, encoding='utf-8-sig')
filename = 'AllHotelReviews.csv'

X, Y = get_inputs_and_outputs(filename)
m = X.shape[0]
n = X.shape[1]

m1 = int(m * 3/5)
m2 = int(m * 4/5)
x_train = X[:m1, :]
x_cv = X[m1:m2, :]
x_test = X[m2:m, :]
y_train = Y[:m1, :]
y_cv = Y[m1:m2, :]
y_test = Y[m2:m, :]
