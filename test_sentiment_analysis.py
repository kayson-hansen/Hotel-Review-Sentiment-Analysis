from sentiment_analysis import evaluate_model
from web_scraper import file_paths


# choose the 3 parameters for your model here
# model can be 'neural network' or 'logistic regression'
model = 'neural network'
# embedding can be 'doc2vec' or 'mean'
embedding = 'doc2vec'
# output can be 'softmax' or 'binary'
output = 'softmax'
# oucput can be True or False
confusion_matrix = True

# trains and evaluates the model of your choice
if embedding == 'doc2vec' and output == 'softmax':
    input_file = 'Embeddings/MulticlassDoc2VecEmbeddings.npy'
    evaluate_model(input_file, file_paths, model, softmax=True,
                   confusion_matrix=confusion_matrix)
elif embedding == 'doc2vec' and output == 'binary':
    input_file = 'Embeddings/Doc2VecEmbeddings.npy'
    evaluate_model(input_file, file_paths, model, softmax=False,
                   confusion_matrix=confusion_matrix)
elif embedding == 'mean' and output == 'softmax':
    input_file = 'Embeddings/MulticlassMeanWordEmbeddings.npy'
    evaluate_model(input_file, file_paths, model, softmax=True,
                   confusion_matrix=confusion_matrix)
else:
    input_file = 'Embeddings/MeanWordEmbeddings.npy'
    evaluate_model(input_file, file_paths, model, softmax=False,
                   confusion_matrix=confusion_matrix)
