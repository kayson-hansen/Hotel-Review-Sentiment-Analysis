from sentiment_analysis import evaluate_model
from web_scraper import file_paths


input_file = 'Embeddings/AllEmbeddings.npy'
evaluate_model(input_file, file_paths, 'neural network')
evaluate_model(input_file, file_paths, 'logistic regression')
