import gesture
import cv2 as cv
import time
from PIL import Image
from enum import Enum


class Beverage:
    def __init__(self, name: str, price: float, stock: int):
        self.name = name
        self.price = price
        self.stock = stock


class Distributor:
    def __init__(self):
        self.beverages = []
        self.beverages.append(Beverage('coffee', 3.0, 10))
        self.beverages.append(Beverage('tea', 2.5, 8))
        self.beverages.append(Beverage('juice', 2.2, 0))

    def order(self, bev_id: int):
        # if out of stock, reject
        # otherwise proceed to payment
        pass

    def payment(self):
        pass


class State(Enum):
    DEBUG = 0
    IDLE = 1
    ORDER = 2
    PAYMENT = 3
    DONE = 4


class VendingMachine:
    def __init__(self):
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.rec = gesture.Recognizer(model_name='mobilenet224')
        self.display = Display()
        self.distributor = Distributor()
        self.state = State.IDLE

    def fetch_frame(self):
        if self.cap.isOpened():
            success, image = self.cap.read()
            if success:
                return image
        return None

    def __del__(self):
        self.cap.release()

    def launch(self):
        prev_time = time.time()
        while not cv.waitKey(5) & 0xFF == 27:
            image = self.fetch_frame()
            if image is None:
                continue
            image = cv.flip(image, 1)
            image.flags.writeable = False
            image = image[:, :, [2, 1, 0]]

            result = self.rec(Image.fromarray(image))

            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            image.flags.writeable = True

            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time))

            # business logic
            offer = self.distributor.beverages
            if int(result) < len(offer):
                # order one of the beverages
                pass
            # draw current offer
            else:
                self.display.display_offer(image, offer, fps, result)

            prev_time = curr_time
        self.cap.release()
        return


class Display:
    def __init__(self):
        self.text_font = cv.FONT_HERSHEY_SIMPLEX
        self.text_size = 1
        self.text_thick = 2
        self.text_color = (255, 0, 0)

    def display_offer(self, image, offer, fps, result):
        cv.putText(image, "please choose a drink", (50, 50), self.text_font, self.text_size, self.text_color,
                   self.text_thick)

        self._draw_side(image, offer)
        cv.putText(image, str(fps), (50, image.shape[0] - 50), self.text_font, self.text_size, (0, 0, 255),
                   self.text_thick)
        cv.putText(image, result, (image.shape[1] - 100, image.shape[0] - 50), self.text_font, self.text_size,
                   self.text_color, self.text_thick)
        cv.imshow('display', image)

    def _draw_side(self, image, offer: list):
        for i in range(0, len(offer)):
            text = str(i) + ". " + offer[i].name
            cv.putText(image, text, (image.shape[1] - 200, 150 + i * 30 * self.text_size), self.text_font,
                       self.text_size, self.text_color, self.text_thick)


if __name__ == '__main__':
    machine1 = VendingMachine()
    machine1.launch()
