import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
#
class Birds(nn.Module):
    def __init__(self):

        super(Birds, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(3600, 1600)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1600, 400)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(400, 200)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.flat(out)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

    def f(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        return out

    def h(self,x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

class ConceptNetwork(nn.Module):
    def __init__(self, concept_names, freeze=True, softmax=False):
        super(ConceptNetwork, self).__init__()

        # Build the feature detector
        # Load pre-trained model, replace last fully connected layer with identity and freeze
        pretrained_model    = models.resnet18(weights='IMAGENET1K_V1')
        penultimate_neurons = pretrained_model.fc.in_features
        pretrained_model.fc = nn.Identity()
        for param in pretrained_model.parameters():
            param.requires_grad = not freeze
        self.feature_detector = pretrained_model

        #Build the concept linear layers
        self.concept_names  = concept_names
        self.concept_preact = nn.ModuleDict({concept: nn.Linear(penultimate_neurons, neurons) for concept, neurons in self.concept_names.items()})
        #self.concept_layers = nn.ModuleDict({concept: nn.Softmax(dim=0) for concept, neurons in self.concept_names.items()})
        #self.concept_layer = nn.Linear(penultimate_neurons, 200)


    def forward(self, x):

        out = self.feature_detector(x)
        #out = self.concept_layer(out)
        out = {concept_name: layer(out) for concept_name, layer in self.concept_preact.items()}
        #out = {concept_name: self.concept_layers[concept_name](out[concept_name]) for concept_name, layer in out.items()}
        return out


class ConceptNetworkMultiFeatures(nn.Module):
    def __init__(self, concept_names, freeze=True):
        super(ConceptNetworkMultiFeatures, self).__init__()

        self.concept_names = concept_names

        # Build the feature detectors
        # Load pre-trained model, replace last fully connected layer with identity and freeze
        pretrained_models   = {concept_name: models.resnet18(weights='IMAGENET1K_V1') for concept_name in concept_names.keys()}
        penultimate_neurons = {concept_name: pretrained_model.fc.in_features for concept_name, pretrained_model in pretrained_models.items()}

        for pretrained_model in pretrained_models.values():
            pretrained_model.fc = nn.Identity()
            for param in pretrained_model.parameters():
                param.requires_grad = not freeze

        self.feature_detectors = nn.ModuleDict(pretrained_models)

        self.concept_preact = nn.ModuleDict({concept_name: nn.Linear(penultimate_neurons[concept_name], neurons) for concept_name, neurons in self.concept_names.items()})

        #self.concept_layers = nn.ModuleDict({concept: nn.Softmax(dim=0) for concept, neurons in self.concept_names.items()})
        #self.concept_layer = nn.Linear(penultimate_neurons, 200)


    def forward(self, x):

        out = {concept_name: feature_detector(x) for concept_name, feature_detector in self.feature_detectors.items()}
        out = {concept_name: layer(out[concept_name]) for concept_name, layer in self.concept_preact.items()}

        return out