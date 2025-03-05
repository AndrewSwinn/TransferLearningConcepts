# Andrew Swinn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import argparse
from PIL import Image
from tempfile import TemporaryDirectory

from src.Logger import Logger
from src.DataLoader import CaltechBirdsDataset
from src.NeuralModels import ConceptNetwork

torch.multiprocessing.freeze_support()
cudnn.benchmark = True

# Read parameters
parser = argparse.ArgumentParser(epilog='Configure the search')
parser.add_argument('--concept_name', '-c', help='Concept Name', type=str, default='one_hot')
parser.add_argument('--validate',     '-v', help='Vaidate'     , type=bool, default=False)
args = parser.parse_args()


def train_model(model, concept, criterion, optimizer, scheduler, num_epochs=25, phases=['Train']):
    for epoch in range(num_epochs):

        # Each epoch can have a training and validation phase
        for phase in phases:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data_dict, inputs in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = data_dict[concept]
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)[concept]
                    values, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss +=  loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'Train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.log(f'Epoch: {epoch} Phase: {phase} Loss: {epoch_loss:.2f} Acc: {epoch_acc:.2f}')



    return model

if __name__ == '__main__':

    results_dir = os.path.join(os.getcwd(), 'results')
    for dir in [results_dir]:
        if not os.path.exists(dir): os.makedirs(dir)

    logger = Logger()

    phases = ['Train', 'Test'] if args.validate else ['Train']

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'Train': [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ],
        'Test': [
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ],
    }


    image_datasets = {x:CaltechBirdsDataset(train=(x=='Train'), bounding=True, augments=data_transforms[x]) for x in phases}
    dataloaders    = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=1) for x in phases}
    dataset_sizes  = {x: len(image_datasets[x]) for x in phases}


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.log(f"Using {device} device")
    logger.log('Concept: ' + args.concept_name)


    concept_names = {'one_hot': 200}
    for name in image_datasets['Train'].concept_names:
        concept_names[name] =  len(image_datasets['Train'].attributes.loc[(image_datasets['Train'].attributes['concept_name']==name)])

    concept_set  = {concept_name: neurons for concept_name, neurons in concept_names.items() if concept_name == args.concept_name}

    model_ft = ConceptNetwork(concept_names=concept_names, freeze=False)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7 , gamma=0.1)

    concept = args.concept_name

    model_ft = train_model(model_ft, concept, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=24, phases=phases)

    torch.save(model_ft.state_dict(), os.path.join(results_dir, 'Renset_' + concept + '.pth'))