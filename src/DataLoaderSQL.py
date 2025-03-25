import os
import socket
import pickle
import sqlite3
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
elif socket.gethostname() == 'andrew-ubuntu':
    data_dir ='/home/andrew/Data/CUB_200_2011'
else:
    data_dir = '/home/bwc/ams90/datasets/caltecBirds/CUB_200_2011'

class CaltechBirdsDataset(Dataset):
    def __init__(self, train=True, bounding=False, normalize=True, augments=[]):

        super(CaltechBirdsDataset).__init__()

        database_file = os.path.join(data_dir, 'birds.db')

        self.data_dir  = data_dir
        self.bounding  = bounding
        self.augments  = augments
        self.trainset  = int(train)

        self.preprocessing = [T.ToTensor()]
        if normalize:
            self.preprocessing += [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

        if not os.path.exists(database_file):
            raise Exception(f"Database file '{database_file}' does not exist.")

        self.conn = sqlite3.connect(database=database_file)

        cursor = self.conn.cursor()
        self.concepts = {concept_id: concept_name for (concept_id, concept_name) in cursor.execute("select concept_id,concept_name from concepts").fetchall()}
        cursor.close()

    def __len__(self):
        cursor = self.conn.cursor()
        (count,) = cursor.execute("""select count(*) from images where trainset = ?""", (self.trainset,)).fetchone()
        cursor.close()
        return count

    def __getitem__(self, idx):

        # idx is the positional index in the filtered (train or test) images
        cursor = self.conn.cursor()
        (image_id, filename, class_id, box_x, box_y, box_w, box_h,) = cursor.execute("""select image_id, filename, class_id, box_x, box_y, box_w, box_h from images where trainset = ? order by image_id  limit 1 offset  ? """, (self.trainset, idx)).fetchone()

        data_dict = dict()
        for concept_id, concept_name in self.concepts.items():
            concept_ids = cursor.execute("""select  a.value_id
                                                 from    attributes a,
                                                         image_attributes ia
                                                 where   a.concept_id = ?
                                                 and     ia.image_id  = ?
                                                 and     ia.present   = 1
                                                 and     ia.attribute_id = a.attribute_id""", (concept_id, image_id)).fetchall()
            data_dict[concept_name] = [concept for (concept,) in concept_ids]

        cursor.close()

        file_path  = os.path.join(self.data_dir, 'images', filename)
        image      = Image.open(file_path)

        if self.bounding:
            x, y, w, h  = box_x, box_y, box_w, box_h
            left, right, top, bottom = x, x + w, y, y + h
            image = image.crop((left, top, right, bottom))

        image = image.resize((300, 300)).convert(mode='RGB')


        image = T.Compose(self.augments + self.preprocessing)(image)

        return data_dict, image





if __name__ == '__main__':

    transforms = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(35) ])

    data = CaltechBirdsDataset(train=True, bounding=True, augments=transforms)
    birdloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

    for data, images in birdloader:
        break




