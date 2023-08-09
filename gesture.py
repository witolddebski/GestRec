"""
This module recognizes hand gestures based on palm images.
"""
import torch.jit
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import time
from PIL import Image
import torch.ao.quantization as quantize
import torchvision.models.mobilenetv3


class Recognizer:
    """
    Perform overall recognition process.
    """

    def __init__(self, model_name: str = 'mobilenet224', threshold: int = 3):
        """
        Create a Recognizer object. It initiates one of Detector and Analyzer, necessary for recognition.

        :param model_name: model of choice, passed to Detector. See Detector's constructor.
        :param threshold: threshold, passed to Analyzer. See Analyzer's constructor.
        """
        self.detector = Detector(model_name)
        self.analyzer = Analyzer(threshold=threshold)

    def __call__(self, img) -> int:
        """
        Perform inference by passing the image to Detector and then Analyzer.

        :param img: PIL image to be analyzed, passed to Detector. The image must be flipped horizontally, as when using
            a frontal camera.
        :return: class id of detected gesture.
        """
        return self.analyzer(self.detector(img))


class Detector:
    """
    Responsible for inferring a gesture from an image. Contains a model used in the recognition process.
    """
    def __init__(self, model_name: str, jit_trace: bool = True) -> None:
        """
        Create a Detector object. Model chosen by passing model_name is then fused and compiled to TorchScript.

        :param model_name: model of choice. Must be one of 'resnet34', 'mobilenet512' or 'mobilenet224'.
        :param jit_trace: bool specifying if compilation to TorchScript should be done via scripting or tracing.
        :raise ValueError: if anything beside the valid model names is used.
        """
        self.classes = [0, 1, 10, 11, 12, 13, 14, 15, 16, 17,
                        18, 19, 2, 3, 4, 5, 6, 7, 8, 9]

        if model_name == 'resnet34':
            self.model = torchvision.models.resnet34()
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))
            self.model.load_state_dict(torch.load('models/resnet34.zip', map_location=torch.device('cpu')))
            self.transformer = transforms.Compose([
                transforms.Resize([512, 512]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        elif model_name == 'mobilenet512':
            self.model = torchvision.models.mobilenet_v3_large()
            num_features = self.model.classifier._modules['3'].in_features
            self.model.classifier._modules['3'] = nn.Linear(num_features, len(self.classes), bias=True)
            self.model.load_state_dict(torch.load('models/mobilenet_v3_large512.zip', map_location=torch.device('cpu')))
            self.transformer = transforms.Compose([
                transforms.Resize([512, 512]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.model.eval()
            self.__fuse_mobilenet()

        elif model_name == 'mobilenet224':
            self.model = torchvision.models.mobilenet_v3_large()
            num_features = self.model.classifier._modules['3'].in_features
            self.model.classifier._modules['3'] = nn.Linear(num_features, len(self.classes), bias=True)
            self.model.load_state_dict(torch.load('models/mobilenet_v3_large224.zip', map_location=torch.device('cpu')))
            self.transformer = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.model.eval()
            self.__fuse_mobilenet()
        else:
            raise ValueError('model_name must be one of available models')

        self.model.eval()

        if jit_trace:
            sample_image = Image.open("test_images/16.jpg")
            sample_image = self.transformer(sample_image)
            sample_image = sample_image.unsqueeze(0)
            self.model = torch.jit.optimize_for_inference(torch.jit.trace(self.model, sample_image))
        else:
            self.model = torch.jit.optimize_for_inference(torch.jit.script(self.model))

    def __fuse_mobilenet(self):
        """
        Fuses self.model to provide better performance. Do not use on models other than MobilenetV3_large!

        :return: None, overwrites model stored in self.model.
        """
        for m in self.model.modules():
            if type(m) == torchvision.ops.Conv2dNormActivation:
                # noinspection PyTypeChecker
                if len(m) == 3 and type(m[2]) == nn.ReLU:
                    quantize.fuse_modules(m, [['0', '1', '2']], inplace=True)
                else:
                    quantize.fuse_modules(m, [['0', '1']], inplace=True)

    def __call__(self, img_raw) -> int:
        """
        Perform inference of the gesture in an image.

        :param img_raw: PIL image to be analyzed. The image must be flipped horizontally, as when using
            a frontal camera.

        :return: string with class number of detected gesture.
        """
        img = self.transformer(img_raw)
        img = img.unsqueeze(0)

        with torch.inference_mode():
            outputs = self.model(img)
            _, prediction = torch.max(outputs, 1)
        return self.classes[prediction[0]]


class Analyzer:
    """
    Analyze Detector results across frames. the Analyzer makes the results more robust by tracking the current gesture.
    """

    def __init__(self, threshold: int):
        """
        Create an Analyzer object. Initially, the current gesture is set to 19.

        :param threshold: controls how many consecutive frames should the gesture appear in before the analyzer starts.
            outputting it as a current result. Increasing this number will make the prediction more robust, but less
            responsive to gesture changes.
        """
        self.current_gest = 19
        self.counter = 0
        self.predicted_gest = 19
        self.threshold = threshold

    def __call__(self, detected_gest: int) -> int:
        """
        Analyze a given result in the context of previous results. After consecutively seeing the same gesture
        the number of times equal to threshold, the Analyzer will start outputting this gesture.

        :param detected_gest: class id of the detected gesture.

        :return: class id of the current gesture.
        """
        if detected_gest != self.current_gest and detected_gest == self.predicted_gest:
            if self.counter < self.threshold:
                self.counter += 1
            else:
                self.counter = 0
                self.current_gest = detected_gest
        else:
            self.predicted_gest = detected_gest
            self.counter = 0

        return self.current_gest


if __name__ == '__main__':
    rec = Recognizer(model_name='mobilenet224')
    images = [Image.open("test_images/series_1/" + x) for x in os.listdir('test_images/series_1')]
    repeats = 15
    print(f'running test on loaded images {repeats} times')
    start_time = time.time()
    for i in range(repeats):
        for image in images:
            print(rec(image))

    end_time = time.time()
    frames = len(os.listdir('test_images/series_1'))
    print("Test concluded. Frame rate: %.3f" % (frames * repeats / (end_time - start_time)))
