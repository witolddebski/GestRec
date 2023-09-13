import gestrec.gesture as gesture
import cv2 as cv
import time
from PIL import Image


class Camera:
    def __init__(self):
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.rec = gesture.Recognizer(model_name='mobilenet224')
        self.text_font = cv.FONT_HERSHEY_SIMPLEX
        self.labels = ["1", "2", "3", "3_alt", "4", "5", "thumbs_down", "thumbs_up", "ok", "salute", "4_down",
                       "1_down", "3_down", "stop", "fist", "delete", "L", "Y", "rock", "none"]

    def __del__(self):
        self.cap.release()

    def draw(self, image, fps, label):
        cv.putText(image, str(fps), (50, image.shape[0] - 50), self.text_font, 1, (0, 0, 255), 2)
        cv.putText(image, label, (image.shape[1] - 50 - 10 * len(label), image.shape[0] - 50), self.text_font, 1, (255, 0, 0), 2)

    def launch(self):
        prev_time = time.time()
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                continue

            # image needs to be flipped
            image = cv.flip(image, 1)
            image.flags.writeable = False
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            # run the inference and apply custom label
            result = self.rec(Image.fromarray(image))
            label = self.labels[result]

            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            image.flags.writeable = True

            # calculate fps
            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time))
            prev_time = curr_time

            # draw results and display
            self.draw(image, fps, label)
            cv.imshow('camera', image)

            if cv.waitKey(5) & 0xFF == 27:
                break


if __name__ == '__main__':
    cam = Camera()
    cam.launch()
