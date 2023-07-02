import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image


class Recognizer:
    """
    Perform overall recognition process
    """

    def __init__(self):
        self.detector = Detector()
        self.analyzer = Analyzer()

    def __call__(self, img):
        return self.analyzer(self.detector(img))


class Detector:
    """
    Detect and classify gestures from images
    """

    def __init__(self):
        self.model = torchvision.models.resnet34(progress=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 20)
        self.model.load_state_dict(torch.load('models/resnet34.zip', map_location=torch.device('cpu')))
        self.transformer = transforms.Compose([
            transforms.Resize([512, 512]),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.classes = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17',
                        '18', '19', '2', '3', '4', '5', '6', '7', '8', '9']
        self.model.eval()

    def __call__(self, img_raw):
        # optimize by calling ahead of time?
        img = self.transformer(img_raw)
        img = img.unsqueeze(0)

        with torch.inference_mode():
            outputs = self.model(img)
            _, pred = torch.max(outputs, 1)
        return self.classes[pred[0]]


class Analyzer:
    """
    Analyze Detector results across frames
    """

    def __call__(self, gesture):
        return gesture


if __name__ == '__main__':
    rec = Recognizer()
    image_path = "test_images/16.jpg"
    image = Image.open(image_path)

    print(rec(image))
    # print(os.listdir('test_images'))
# TODO:
# run model on an image
# try training resnet with pretrained=True
