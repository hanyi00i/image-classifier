import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
from PIL import Image
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Image Classifier predict.py')

    parser.add_argument(
        '--image_dir', default='flowers/test/47/image_04993.jpg')
    parser.add_argument('--load_dir', default='checkpoint.pth')
    parser.add_argument('--top_k', default=5, type=int)
    parser.add_argument(
        '--category_names', help='Use a mapping of categories to real names', default='cat_to_name.json')
    parser.add_argument('--gpu', default='gpu')

    return parser.parse_args()


def load_model(arch):
    exec('model = models.{}(pretrained=True)'.format(arch), globals())

    for param in model.parameters():
        param.requires_grad = False
    return model


def initialize_classifier(model, hidden_units, output_features):
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(4096, 256)),
                              ('relu', nn.ReLU()),
                              ('fc3', nn.Linear(256, 102)),
                              ('output', nn.LogSoftmax(dim=1))
    ]))
    return classifier


def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image):
    image_transforms = transforms.Compose([transforms.Resize(256),  # resize the image to 256 pixels on the shortest side
                                           # crop the center 224x224 pixels from the image
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),  # convert the image to a PyTorch tensor
                                           # normalize the image channels with mean and std values
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    # Processes a PIL image for use in a PyTorch model
    pil_image = Image.open(image)

    # Applies the specified transformations
    tensor_image = image_transforms(pil_image)
    return tensor_image


def predict(image_path, model, topk):

    # Set the model to evaluation mode
    model.eval()
    # Move the model to CPU
    model.cpu()
    # Process the image
    image = process_image(image_path)
    # Add a batch dimension
    image = image.unsqueeze(0)
    # Disable gradient calculation
    with torch.no_grad():
        # Forward pass through the model
        output = model(image)
        # Get the top k probabilities and labels
        probs, labels = torch.topk(output, topk)
        # Convert the probabilities to a tensor of actual probabilities
        probs = probs.exp()
        # Reverse the class to index mapping
        class_to_idx_rev = {
            model.class_to_idx[k]: k for k in model.class_to_idx}
        # Convert the labels from tensor to numpy array and get the corresponding classes
        classes = []
        for label in labels.numpy()[0]:
            classes.append(class_to_idx_rev[label])
        # Return the probabilities and classes as numpy arrays
        return probs.numpy()[0], classes


def main():
    args = arg_parse()

    image_dir = args.image_dir
    checkpoint_dir = args.load_dir
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu
    model = load_model('vgg16')
    model.classifier = initialize_classifier(model, 4096, 102)
    model = load_checkpoint(checkpoint_dir, model)

    image_tensor = process_image(args.image_dir)

    probs, classes = predict(image_dir, model, top_k)
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    flower_class = []
    for flower in classes:
        flower_name = cat_to_name[str(flower)]
        flower_class.append(flower_name)
    flower_name = flower_class[0]

    for i in range(top_k):
        print("Class flower of {} has a probability of {:.2f}%".format(
            flower_class[i], probs[i]*100))


if __name__ == '__main__':
    main()
