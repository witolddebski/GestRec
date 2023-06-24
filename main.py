import torch
import torch.nn as nn
import torchvision


if __name__ == '__main__':
    recognizer = torchvision.models.resnet34(progress=True)
    num_features = recognizer.fc.in_features
    recognizer.fc = nn.Linear(num_features, 20)
    recognizer.load_state_dict(torch.load('models/resnet34.zip', map_location=torch.device('cpu')))
    print(recognizer)


# TODO:
# run model on an image
# try training resnet with pretrained=True
