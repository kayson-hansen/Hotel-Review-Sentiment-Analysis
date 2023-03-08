# This file is meant to be run once to generate a file containing the numpy embeddings for all the reviews,
# so that you don't have to wait for them to be generated each time you want to test the neural network
from load_data import get_mean_embedding_inputs, get_doc2vec_inputs
import numpy as np


file_paths = [
    "/users/kaysonhansen/cs129/HotelReviewData/VenetianHotelReviews.csv",
    "/users/kaysonhansen/cs129/HotelReviewData/MirageHotelReviews.csv",
    "/users/kaysonhansen/cs129/HotelReviewData/MandalayBayHotelReviews.csv",
    "/users/kaysonhansen/cs129/HotelReviewData/TrumpInternationalHotelReviews.csv",
    "/users/kaysonhansen/cs129/HotelReviewData/LuxorHotelReviews.csv",
    "/users/kaysonhansen/cs129/HotelReviewData/TreasureIslandHotelReviews.csv",
    "/users/kaysonhansen/cs129/HotelReviewData/ParisHotelReviews.csv",
    "/users/kaysonhansen/cs129/HotelReviewData/CaesarsPalaceHotelReviews.csv",
    "/users/kaysonhansen/cs129/HotelReviewData/ARIAHotelReviews.csv",
    "/users/kaysonhansen/cs129/HotelReviewData/PlanetHollywoodHotelReviews.csv",
    "/users/kaysonhansen/cs129/HotelReviewData/PalazzoHotelReviews.csv",
    "/users/kaysonhansen/cs129/HotelReviewData/ParkMGMReviews.csv",
    "/users/kaysonhansen/cs129/HotelReviewData/VdaraHotelReviews.csv",
    "/users/kaysonhansen/cs129/HotelReviewData/ExcaliburHotelReviews.csv",
    "/users/kaysonhansen/cs129/HotelReviewData/WynnHotelReviews.csv"
]

X = get_doc2vec_inputs(file_paths, True)
# X = get_mean_embedding_inputs(file_paths, True)
np.save('Embeddings/MulticlassDoc2VecEmbeddings.npy', X)
# np.save('Embeddings/Doc2VecEmbeddings.npy', X)
# np.save('Embeddings/MulticlassMeanWordEmbeddings.npy', X)
