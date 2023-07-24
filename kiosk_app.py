import gesture
import cv2 as cv
import time
from PIL import Image
from enum import Enum


class State(Enum):
    DEBUG = 0
    IDLE = 1
    ORDER = 2
    PAYMENT = 3
    DONE = 4


class Kiosk:
    def __init__(self):
        self.rec = gesture.Recognizer(model_name='mobilenet224')
        self.reader = Reader()
        self.state = State.IDLE
        self.order = None

    def __call__(self, img):
        return self.rec(img)

    def operate(self, result):
        if self.state == State.IDLE:
            if result == '00':
                self.order = 'coffee'
                self.state = State.ORDER
            elif result == '01':
                self.order = 'tea'
                self.state = State.ORDER
        if self.state == State.ORDER:
            if result == '07':
                self.state = State.PAYMENT
            if result == '06':
                self.state = State.IDLE

    def launch(self):
        prev_time = time.time()
        while not cv.waitKey(5) & 0xFF == 27:
            image = self.reader.fetch_frame()
            if image is None:
                continue
            image = cv.flip(image, 1)
            image.flags.writeable = False
            image = image[:, :, [2, 1, 0]]

            result = self.rec(Image.fromarray(image))

            # business logic
            self.operate(result)

            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            image.flags.writeable = True

            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time))

            # update view
            self.reader.draw(image, self.state, result, fps)

            prev_time = curr_time
        self.reader.release()
        return


class Reader:
    def __init__(self):
        self.text_font = cv.FONT_HERSHEY_SIMPLEX
        self.text_size = 1
        self.text_thick = 2
        self.text_color = (255, 0, 0)
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)

    def fetch_frame(self):
        if self.cap.isOpened():
            success, image = self.cap.read()
            if success:
                return image
        return None

    def draw(self, image, state, result, fps):
        display_side = False
        if state == State.IDLE:
            text = "please choose a drink"
            display_side = True
        elif state == State.ORDER:
            text = "great choice! Please confirm order"
        elif state == State.PAYMENT:
            text = "please pay at the terminal"
        else:
            text = ""

        if display_side:
            cv.putText(image, "1. coffee", (image.shape[1] - 200, 150), self.text_font, self.text_size, self.text_color,
                       self.text_thick)
            cv.putText(image, "2. tea", (image.shape[1] - 200, 150 + 30*self.text_size), self.text_font, self.text_size,
                       self.text_color, self.text_thick)
        cv.putText(image, text, (50, 50), self.text_font, self.text_size, self.text_color, self.text_thick)
        cv.putText(image, result, (image.shape[1] - 100, image.shape[0] - 50), self.text_font, self.text_size,
                   self.text_color, self.text_thick)
        cv.putText(image, str(fps), (50, image.shape[0] - 50), self.text_font, self.text_size, (0, 0, 255),
                   self.text_thick)
        cv.imshow('camera', image)

    def release(self):
        self.cap.release()


if __name__ == '__main__':
    kiosk = Kiosk()
    kiosk.launch()

