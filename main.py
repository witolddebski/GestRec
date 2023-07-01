import torch
import torch.nn as nn
import torchvision


class Recognizer:
    """
    Perform overall recognition process
    """

    def __init__(self):
        self.detector = Detector()
        self.analyzer = Analyzer()


class Detector:
    """
    Detect and classify gestures from images
    """
    def __init__(self):
        self.model = torchvision.models.resnet34(progress=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 20)
        self.model.load_state_dict(torch.load('models/resnet34.zip', map_location=torch.device('cpu')))


class Analyzer:
    """
    Analyze Detector results across frames
    """
    def __call__(self, gesture: int):
        return gesture


if __name__ == '__main__':
    rec = Recognizer()
    print(rec.detector.model)


# TODO:
# run model on an image
# try training resnet with pretrained=True
