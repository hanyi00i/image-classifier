import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Image Classifier train.py')

    parser.add_argument('--data_dir', default='flowers')
    parser.add_argument('--save_dir', default='/checkpoint.pth')
    parser.add_argument('--arch', default='vgg16')
    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--hidden_units', default=4096, type=int)
    parser.add_argument('--output_features', default=102, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--gpu', help='Use GPU for training', default='gpu')

    return parser.parse_args()


def train_transform(train_dir):
    transform = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    train_set = datasets.ImageFolder(train_dir, transform=transform)
    return train_set


def valid_transform(valid_dir):
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    valid_set = datasets.ImageFolder(valid_dir, transform=transform)
    return valid_set


def train_loader(data, batch_size=64):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)


def valid_loader(data, batch_size=64):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)


def check_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def load_model(arch):
    exec('model = models.{}(pretrained=True)'.format(arch), globals())

    for param in model.parameters():
        param.requires_grad = False
    return model


def initialize_classifier(model, hidden_units, output_features):
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(4096, 256)),
                              ('relu', nn.ReLU()),
                              ('fc3', nn.Linear(256, output_features)),
                              ('output', nn.LogSoftmax(dim=1))
    ]))
    return classifier


def train_model(model, trainloader, validloader, device, optimizer, criterion, epochs, print_every, batch):
    running_loss = running_accuracy = 0
    validation_losses, training_losses = [], []

    # Defines the training process
    for e in range(epochs):
        batch = 0
        # Turns on training mode
        model.train()

        for images, labels in trainloader:
            batch += 1

            # Moves images and labels to the GPU
            images, labels = images.to(device), labels.to(device)
            # Pushes batch through network
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculates the metrics
            ps = torch.exp(outputs)
            top_ps, top_class = ps.topk(1, dim=1)
            matches = (top_class == labels.view(
                *top_class.shape)).type(torch.FloatTensor)
            accuracy = matches.mean()

            # Resets optimiser gradient and tracks metrics
            optimizer.zero_grad()
            running_loss += loss.item()
            running_accuracy += accuracy.item()

            # Runs the model on the validation set every 5 loops
            if batch % print_every == 0:
                # Sets the metrics
                validation_loss = 0
                validation_accuracy = 0
                model.eval()  # Turns on evaluation mode
                with torch.no_grad():  # Turns off calculation of gradients
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model.forward(images)
                        loss = criterion(outputs, labels)
                        ps = torch.exp(outputs)
                        top_ps, top_class = ps.topk(1, dim=1)
                        matches = (top_class == labels.view(
                            *top_class.shape)).type(torch.FloatTensor)
                        accuracy = matches.mean()

                        # Tracks validation metrics (test of the model's progress)
                        validation_loss += loss.item()
                        validation_accuracy += accuracy.item()

                # Tracks training metrics
                validation_losses.append(running_loss/print_every)
                training_losses.append(validation_loss/len(validloader))

                # Prints out metrics
                print(f'Epoch {e+1}/{epochs} , Batch {batch}',
                      f', Training Loss: {running_loss/print_every:.3f}',
                      f', Training Accuracy: {running_accuracy/print_every*100:.2f}%',
                      f', Validation Loss: {validation_loss/len(validloader):.3f}',
                      f', Validation Accuracy: {validation_accuracy/len(validloader)*100:.2f}%')

                # Resets the metrics and turns on training mode
                running_loss = running_accuracy = 0
                model.train()

    return model


def save_checkpoint(model, optimizer, class_to_idx, path, arch, hidden_units, output_features):
    model.class_to_idx = class_to_idx
    # Defines model's checkpoint
    checkpoint = {'state_dict': model.state_dict(),
                  'class_to_idx': class_to_idx,
                  'optimizer': optimizer.state_dict(),
                  }
    # Saves model in current directory
    torch.save(checkpoint, 'checkpoint.pth')


def main():
    args = arg_parse()

    data_dir = args.data_dir
    save_path = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    output_features = args.output_features
    epochs = args.epochs
    gpu = args.gpu

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    train_set = train_transform(train_dir)
    valid_set = valid_transform(valid_dir)

    trainloader = train_loader(train_set)
    validloader = valid_loader(valid_set)

    if args.gpu:
        device = check_device()

    model = load_model(arch)
    model.classifier = initialize_classifier(
        model, hidden_units, output_features)

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    model.to(device)

    print_every = 20
    batch = 0

    train_model(model, trainloader, validloader, device,
                optimizer, criterion, epochs, print_every, batch)
    save_checkpoint(model, optimizer, train_set.class_to_idx,
                    save_path, arch, hidden_units, output_features)


if __name__ == '__main__':
    main()
