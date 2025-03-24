import os
import socket
import pickle
from PIL import Image, ImageOps, ImageEnhance
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

if socket.gethostname() == 'LTSSL-sKTPpP5Xl':
    data_dir = 'C:\\Users\\ams90\\PycharmProjects\\ConceptsBirds\\data'
elif socket.gethostname() == 'LAPTOP-NA88OLS1':
    data_dir = 'D:\\data\\caltecBirds\\CUB_200_2011'
else:
    data_dir = '/home/bwc/ams90/datasets/caltecBirds/CUB_200_2011'

class CaltechBirdsDataset(Dataset):
    def __init__(self, train=True, bounding=False, normalize=True, augments=[]):

        super(CaltechBirdsDataset).__init__()

        self.data_dir  = data_dir
        self.bounding  = bounding
        self.augments  = augments

        self.preprocessing = [T.ToTensor()]
        if normalize:
            self.preprocessing += [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

        try:
            with open(os.path.join(data_dir, 'data_dict.pkl'), 'rb') as data_dict_file:
                archive = pickle.load(data_dict_file)
                [attributes, concept_names, data_dict] = archive

        except:

            # Read metadata into pandas tables and join - to create a data dictionary
            train_test_split = pd.read_csv(os.path.join(data_dir, 'train_test_split.txt'),   sep=" ", index_col=[0], names=['image_id', 'trainset'])
            image_file_names = pd.read_csv(os.path.join(data_dir, 'images.txt'),             sep=" ", index_col=[0], names=['image_id', 'file_name'])
            class_labels     = pd.read_csv(os.path.join(data_dir, 'image_class_labels.txt'), sep=" ", index_col=[0], names=['image_id', 'class_id'])
            bounding_boxes   = pd.read_csv(os.path.join(data_dir, 'bounding_boxes.txt'),     sep=" ", index_col=[0], names=['image_id', 'x', 'y', 'w', 'h']).astype(int)
            class_labels['one_hot'] = class_labels['class_id'] - 1

            #Load the attributes datafile, and separate attributes in to concept names and concept values
            attributes       = pd.read_csv(os.path.join(data_dir, 'attributes.txt'),         sep=" ", index_col=[0], names=['attribute_id', 'attribute_name'])
            attributes['concept_name']  = attributes.apply(lambda x: x['attribute_name'][0: x['attribute_name'].find(':')], axis=1)
            attributes['concept_value'] = attributes.apply(lambda x: x['attribute_name'][x['attribute_name'].find(':')+2:], axis=1)
            concept_names = attributes['concept_name'].unique()

            #For each concept find the list of values. each list of values starts with 'Unknown' to copy with missing concept values
            values_dict = dict()
            for concept_name in concept_names:
                query_string = "concept_name==" + '\'' + concept_name + '\''
                concept_values = ['Unknown'] + attributes.query(query_string)['concept_value'].tolist()
                values_dict[concept_name] = concept_values

            #For each attribute give each concept value an id (ranging from 1 to n) for each concept
            for i, attribute in attributes.iterrows():
                concept_name = attribute['concept_name']
                concept_value = attribute['concept_value']
                concept_id = values_dict[concept_name].index(concept_value)
                attributes.at[i, 'concept_id'] = concept_id

            for concept_name in concept_names:
                attribute_name = concept_name + '::unknown'
                concept_value  = 'unknown'
                concept_id     = 0
                attributes = pd.concat((attributes, pd.DataFrame( [[attribute_name, concept_name, concept_value, concept_id]], columns=['attribute_name', 'concept_name', 'concept_value', 'concept_id'] )), axis=0)


            data_dict = pd.concat((train_test_split, image_file_names, class_labels, bounding_boxes, pd.DataFrame(columns=concept_names)), axis=1)
            data_cert = data_dict.copy()

            certainties            = pd.read_csv(os.path.join(data_dir, 'attributes', 'certainties.txt'),            sep=" ", index_col=[0],names=['certainty_id', 'certainty_name'])
            image_attribute_labels = pd.read_csv(os.path.join(data_dir, 'attributes', 'image_attribute_labels.txt'), sep=" ", names = ['image_id', 'attribute_id', 'present', 'certainty_id', 'time', 'd1', 'd2'])

            # the empty cells are willed with 0 to prevent pytorch casting the concept_id's to floats.
            data_dict = data_dict.fillna(0)

            for index, attribute_label in image_attribute_labels.iterrows():
                if attribute_label['present'] == 1:
                    image_id     = int(attribute_label['image_id'])
                    attribute_id = int(attribute_label['attribute_id'])
                    certainty_id = int(attribute_label['certainty_id'])
                    concept_name = attributes.loc[attribute_id]['concept_name']
                    concept_id   = int(attributes.loc[attribute_id]['concept_id'])

                    if certainty_id > data_cert.at[image_id, concept_name] or data_cert.at[image_id, concept_name] != data_cert.at[image_id, concept_name]:

                        data_dict.at[image_id, concept_name] = concept_id
                        data_cert.at[image_id, concept_name] = certainty_id

            archive = [attributes, concept_names, data_dict]

            with open(os.path.join(data_dir, 'data_dict.pkl'), 'wb') as data_dict_file:
                pickle.dump(archive, data_dict_file)

        self.filtered_data_dict = data_dict.loc[data_dict['trainset']==int(train)]

        #Read lookup data into pandas tables
        self.classes            = pd.read_csv(os.path.join(data_dir, 'classes.txt'),                    sep=" ", index_col=[0], names=['class_id', 'class_name'])
        #self.attributes         = pd.read_csv(os.path.join(data_dir, 'attributes.txt'),                 sep=" ", index_col=[0], names=['attribute_id', 'attribute_name'])
        self.attributes         = attributes
        self.concept_names      = concept_names



    def __len__(self):
        return len(self.filtered_data_dict)

    def __getitem__(self, idx):

        #transform = T.Compose([T.PILToTensor()])


        # idx is the positional index in the filtered (train or test) images
        data = self.filtered_data_dict.iloc[idx]

        file_path  = os.path.join(self.data_dir, 'images', data['file_name'])
        image      = Image.open(file_path)

        if self.bounding:
            x, y, w, h  = data['x'], data['y'], data['w'], data['h']
            left, right, top, bottom = x, x + w, y, y + h
            image = image.crop((left, top, right, bottom))

        image = image.resize((300, 300)).convert(mode='RGB')

        #tensor_image = transform(image)

        image = T.Compose(self.augments + self.preprocessing)(image)


        return data.to_dict(), image



if __name__ == '__main__':

    transforms = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(35) ])

    data = CaltechBirdsDataset(train=False, bounding=True, augments=transforms)

    birds = CaltechBirdsDataset(train=False, bounding=True)
    bird = birds[0][0]


    print(bird)
