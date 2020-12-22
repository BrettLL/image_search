#! coding=utf-8
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from classes import class_dict
import numpy as np
import pickle 
import os, sys

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
)
])
resnet_model = torchvision.models.resnet50(
    pretrained=True, num_classes=1000)
resnet_model.eval()

class feaResNet(nn.Module):
    def __init__(self, original_model):
        super(feaResNet, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        return x

def extract_fea(img_path):
    with torch.no_grad():
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img_pil).float()
        img_tensor = img_tensor.unsqueeze_(0)
        features = feaResNet(resnet_model)(img_tensor).reshape(-1)
        return features

def classifier(img_path):
    with torch.no_grad():
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img_pil).float()
        img_tensor = img_tensor.unsqueeze_(0)
        fc_out = resnet_model(img_tensor)
        output = fc_out.detach().numpy()
        print('label: %s'%class_dict[output.argmax()])
        # print(np.argsort(output[0])[-5:])
        # for x in range(1,6):
        #     print(class_dict[np.argsort(output[0])[-1 * x]],np.sort(output[0])[-1 * x])
def compare_fea(fea, db_fea):
    from scipy.spatial import distance
    fea = fea.reshape((1,-1))
    dist = distance.cdist(fea, db_fea)
    sorted_dist = np.sort(dist)
    sorted_index = np.argsort(dist)
    # import pdb; pdb.set_trace()
    return sorted_dist[0], sorted_index[0]

def main():
    # creat db_fea
    db_dir = 'db_imgs'
    pkl_file = db_dir + '/db_fea.pkl'
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f: 
            db_list, db_fea = pickle.load(f)
    else:
        db_list = os.listdir(db_dir)
        db_fea = np.zeros((len(db_list), 2048))
        for index, path in enumerate(db_list):
            path = db_dir + '/' + path
            fea = extract_fea(path)
            db_fea[index,:] = fea
            if index % 10 == 0:
                print('loading {}/{}'.format(index, len(db_list)))
        with open(pkl_file, 'wb') as f: 
            pickle.dump([db_list, db_fea], f)
    # query img
    query_img = sys.argv[1]
    qry_fea = extract_fea(query_img)
    sorted_dist, sorted_index = compare_fea(qry_fea, db_fea)
    for i in range(5):
        print('name: {}, distance: {}'.format(db_list[sorted_index[i]],sorted_dist[i]))

if __name__ == '__main__':
    main()
