"""
File: task_a_evaluate_image_to_text_retrieval.py
Authors: Juan A. Rodriguez , Igor Ugarte, Francesc Net, David Serrano
Description:
    - This script is used to evaluate the image to text retrieval system for task a.
    - It uses the test set for retrieval using KNN
    - Quantitative and qualitative results are presented
"""

import json
import os.path
import pickle
from os import path

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from dataset import Img2TextDataset
from models import ImgEncoder, TextEncoder
from evaluation_metrics import mapk, plot_confusion_matrix, table_precision_recall

cuda = torch.cuda.is_available()


def extract_embeddings(dataloader, embedding_text_net,embedding_image_net, out_size=256, model_id=''):
    embedding_text_net.to('cuda')
    embedding_image_net.to('cuda')
    with torch.no_grad():        
        image_embeddings = np.zeros((len(dataloader.dataset), out_size))
        text_embeddings = np.zeros((len(dataloader.dataset) * 5, out_size))
        k = 0
        for images, text in dataloader:
            if cuda:
                images = images.cuda()
                text = text.cuda()

            im_emb, text_emb = embedding_image_net(images),embedding_text_net(text)
            image_embeddings[k:k + len(images)] = im_emb.data.cpu().numpy()
            text_embeddings[k:k + len(text) * 5] = text_emb.data.cpu().numpy().reshape(len(text) * 5, out_size)
            k += len(images)

    return image_embeddings, text_embeddings


def main():
    # Load the datasets
    dataset_dir = '/home/marcelo/Documents/Master_CV/M5/M5-T7-Project/week5/data/'
    img_features_file = '/home/marcelo/Documents/Master_CV/M5/M5-T7-Project/week5/data/vgg_feats.mat'
    text_features_file = '/home/marcelo/Documents/Master_CV/M5/M5-T7-Project/week5/data/fasttext_feats.npy'
    INPUT_SIZE = 300

    # Method selection        
    #model_id = base + '_' + image_features + '_' + text_aggregation + '_textagg_' + info

    PATH_MODEL = 'results/task_a/models/'
    PATH_RESULTS = 'results/'
    # Create folder if it does not exist
    if not path.exists(PATH_RESULTS):
        os.makedirs(PATH_RESULTS)

    # Load the test dataset
    test_dataset = Img2TextDataset(img_features_file, text_features_file,'test',dataset_dir)  # Create the test dataset

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=1)

    margin = 1.
    embedding_text_net = TextEncoder()#(embedding_size=300, output_size=out_size, sequence_modeling=None)
    embedding_image_net = ImgEncoder()#EmbeddingImageNet(input_size=4096, output_size=out_size)
    #model = TripletImageText(embedding_text_net, embedding_image_net, margin=margin)

    # Check if file exists
    if path.exists(PATH_MODEL + 'Image2Text' + '_text.pth'):
        print('Loading text model from the disk, {}'.format('Image2Text' + '_text.pth'))
        checkpoint = torch.load(PATH_MODEL + 'Image2Text' + '_text.pth')
        embedding_text_net.load_state_dict(checkpoint)
    else:
        print('COULD NOT LOAD {}'.format(PATH_MODEL + 'Image2Text' + '_text.pth'))
        #load image
    if path.exists(PATH_MODEL + 'Image2Text' + '_image.pth'):    
        print('Loading image model from the disk, {}'.format('Image2Text' + '_image.pth'))
        checkpoint = torch.load(PATH_MODEL + 'Image2Text' + '_image.pth')
        embedding_image_net.load_state_dict(checkpoint)
    else:
        print('COULD NOT LOAD {}'.format(PATH_MODEL + 'Image2Text' + '_image.pth'))

    # Obtain ground truth from the json file (test.json)
    #assume json test is in M5-T7-Project/week5/data
    with open(dataset_dir + 'val.json') as f:
        data = json.load(f)

    gt = {}  # Ground truth a a dictionary with the image filename as key and the list of text id as value
    dict_sentences = {}  # Dictionary with the text id as key and the sentence as value
    count = 0
    for item in data:
        gt[item['filename']] = [x['raw'] for x in item['sentences']]
        for sentence in item['sentences']:
            dict_sentences[count] = sentence['raw']
            count += 1

    # Extract embeddings
    image_embeddings, text_embeddings = extract_embeddings(test_loader, embedding_text_net,embedding_image_net,INPUT_SIZE)
    # Compute the labels for each embedding
    image_labels = [i for i in range(1, 1000 + 1)]
    text_labels = [j for j in range(1, 1000 + 1) for i in range(5)]  # Trick to obtain the same
    # number of labels, copying the same labels 5 (5 text embeddings)

    # Compute the nearest neighbors
    print('Computing the nearest neighbors...')
    k = 5  # Number of nearest neighbors

    # # load results if exists
    # if path.exists(PATH_RESULTS + model_id + '_knn.pkl'):
    #     print('Loading the nearest neighbors from the disk, {}'.format(model_id + '_knn.pkl'))
    #     distances, indices = pickle.load(open(PATH_RESULTS + model_id + '_knn.pkl', 'rb'))
    # else:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree').fit(text_embeddings, text_labels)

    # Make predictions
    distances, indices = knn.kneighbors(image_embeddings)
    # pickle.dump((distances, indices), open(PATH_MODEL + model_id + '_knn.pkl', 'wb'))

    # Compute mAPk
    image_labels_pred = []
    # We create a dict to map the index of a single text with its corresponding label (image)
    text_labels_pred_dict = {}
    for k_predictions in indices.tolist():
        # map indices with the corresponding labels
        k_labels_pred = [text_labels[i] for i in k_predictions]
        image_labels_pred.append(k_labels_pred)

    im_labels = [[i] for i in image_labels]
    map_k = mapk(im_labels, image_labels_pred, k=k)
    print(f'mAP@{k}: {map_k}')

    # Check if the correct label is in the knn
    # new_retrievals = []
    # for idx, retrieval in enumerate(image_labels_pred):
    #     if image_labels[idx] in retrieval:
    #         new_retrievals.append(image_labels[idx])
    #     else:
    #         new_retrievals.append([retrieval[0]])

    # # compute the confusion matrix
    # confusion_matrix = plot_confusion_matrix(im_labels, image_labels_pred, show=True)
    #
    # # compute the precision and recall
    # prec, recall = table_precision_recall(confusion_matrix, show=False)

    # print(f'map@k: {map_k}')
    # print(f'precision@k: {np.mean(prec)}')
    # print(f'recall@k: {np.mean(recall)}')

    # Compute the accuracy
    knn_accuracy = knn.score(image_embeddings, image_labels)
    print('KNN accuracy: {}%'.format(100*knn_accuracy))

    # Qualitative results
    num_samples = 10
    # Create random samples
    random_samples = np.random.choice(image_labels, num_samples, replace=False)
    # im_labels, image_labels_pred
    for sample in random_samples:
        print("Example:" + str(sample))
        print("--------------------------------")

        # Get image embedding from batch
        filename = list(gt)[sample]
        print("Ground truth: ")
        for t in gt[filename]:
            print(t)

        predictions = indices[sample]

        print("Predictions:")
        for pred in predictions:
            print(dict_sentences[pred])

        im = plt.imread(dataset_dir + 'flickr30k_images/flickr30k_images/' + filename)
        plt.imshow(im)
        plt.show()
        print("--------------------------------------------------------------------------------")
# Main
if __name__ == '__main__':
    main()
