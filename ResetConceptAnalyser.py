import os
import torch
import torch.nn as nn
import torchvision.models as models
from src.Logger import Logger
from src.NeuralModels import ConceptNetwork
from src.DataLoader import CaltechBirdsDataset

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger = Logger()

    phases = ['Train', 'Test']
    image_datasets = {x: CaltechBirdsDataset(train=(x == 'Train'), bounding=True) for x in phases}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=False, num_workers=1) for x in phases}
    dataset_sizes = {x: len(image_datasets[x]) for x in phases}

    concept_names = {'one_hot': 200}
    for name in image_datasets['Train'].concept_names:
        concept_names[name] = len(
            image_datasets['Train'].attributes.loc[(image_datasets['Train'].attributes['concept_name'] == name)])

    for model_dict in os.listdir(os.path.join(os.getcwd(),'results')):
        if model_dict[-11:-4] != 'results':
            correct, count = 0, 0
            concept_name = model_dict[8:-4]
            concept_set = {concept_name: concept_names[concept_name]}
            model_ft = ConceptNetwork(concept_names=concept_set, freeze=False)
            model_ft.load_state_dict(
                torch.load(os.path.join(os.getcwd(), 'results', model_dict), weights_only=False))
            model_ft.to(device)
            for data_dict, inputs in dataloaders['Test']:
                inputs = inputs.to(device)
                labels = data_dict[concept_name]
                labels = labels.to(device)

                with torch.set_grad_enabled(False):
                    outputs = model_ft(inputs)[concept_name]
                    values, preds = torch.max(outputs, 1)
                count += len(labels)
                correct += torch.sum(preds == labels).item()

            logger.log((concept_name, round(correct / count, 2)))