# This file is meant to be run once to generate a file containing the numpy embeddings for all the reviews,
# so that you don't have to wait for them to be generated each time you want to test the neural network
from load_data import get_inputs
import spacy
import numpy as np
import csv


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

X = get_inputs(file_paths)
np.save('all_embeddings.npy', X)
