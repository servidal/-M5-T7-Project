import os
import json

import torch
import numpy as np


import numpy as np
from transformers import BertTokenizer, BertModel


load_features = False
DATA_PATH = '../flickr30k'
OUT_PATH = "results/"
device_str = "cuda" if torch.cuda.is_available() else "cpu"


def BertEmbeding():
    with open(os.path.join(DATA_PATH, 'dataset.json')) as f:
        dataset = json.load(f)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    device = torch.device(device_str)

    bert_model = bert_model.to(device)

    features = []
    with torch.no_grad():
        bert_model.eval()
        for _, key in enumerate(dataset['images']):
            sentences = []
            for sentence in key['sentences']:
                #print("{}, {}".format(i,sentence['raw']))
                x = sentence['raw']
                x = bert_tokenizer(x, return_tensors="pt")
                x = x.to(device)
                x = bert_model(**x)['last_hidden_state']
                x = x.to("cpu")
                sentences.append(np.array(x.squeeze().numpy(), dtype="object"))
            features.append(sentences)

    #Save data
    os.path.join(OUT_PATH,"beart_features.npy")
    with open(os.path.join(OUT_PATH,"beart_features.npy"), "wb") as f:
        np.save(f, features)       
    print('Text features saved.')


BertEmbeding()