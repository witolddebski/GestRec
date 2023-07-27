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

    def get_offering(self):
        return self.beverages


class State(Enum):
    DEBUG = 0
    IDLE = 1
    NO_STOCK = 2
    ORDER = 3
    PAYMENT = 4
    DONE = 5


class VendingMachine:
    def __init__(self):
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.rec = gesture.Recognizer(model_name='mobilenet224')
        self.display = Display()
        self.distributor = Distributor()
        self.state = State.IDLE
        self.order = None

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
        timer = 0
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
            prev_time = curr_time

            # business logic
            offer = self.distributor.get_offering()
            if self.state == State.IDLE and int(result) < len(offer):

                if offer[int(result)].stock < 1:
                    self.state = State.NO_STOCK
                    timer = 2 * fps
                else:
                    self.order = offer[int(result)]
                    self.state = State.ORDER

            elif self.state == State.NO_STOCK:
                self.display.display_no_stock(image, offer, fps, result)
                timer -= 1
                if timer < 1:
                    self.state = State.IDLE

            elif self.state == State.ORDER:
                if result == '06':
                    self.state = State.IDLE
                elif result == '07':
                    self.state = State.PAYMENT
                    timer = 3 * fps
                self.display.display_order(image, self.order, fps, result)

            elif self.state == State.PAYMENT:
                self.display.display_payment(image, fps, result)
                timer -= 1
                if timer < 1:
                    self.state = State.IDLE
            else:
                self.display.display_offer(image, offer, fps, result)

        self.cap.release()
        return


class Display:
    def __init__(self):
        self.text_font = cv.FONT_HERSHEY_SIMPLEX
        self.text_size = 1
        self.text_thick = 2
        self.text_color = (255, 0, 0)

    def display_offer(self, image, offer, fps, result):
        self._draw_main_text(image, "please choose a drink")
        self._draw_side(image, offer)
        self._draw_fps(image, fps)
        self._draw_result(image, result)
        cv.imshow('display', image)

    def display_no_stock(self, image, offer, fps, result):
        self._draw_main_text(image, "drink out of stock")
        self._draw_side(image, offer)
        self._draw_fps(image, fps)
        self._draw_result(image, result)
        cv.imshow('display', image)

    def display_order(self, image, order, fps, result):
        self._draw_main_text(image, "please confirm order: " + order.name + " $" + str(order.price))
        self._draw_subtext(image, "use thumbs up / down")
        self._draw_fps(image, fps)
        self._draw_result(image, result)
        cv.imshow('display', image)

    def display_payment(self, image, fps, result):
        self._draw_main_text(image, "pending payment")
        self._draw_subtext(image, "please follow card reader instructions")
        self._draw_fps(image, fps)
        self._draw_result(image, result)
        cv.imshow('display', image)

    def _draw_side(self, image, offer: list):
        for i in range(0, len(offer)):
            text = str(i+1) + ". " + offer[i].name
            cv.putText(image, text, (image.shape[1] - 200, 150 + i * 30 * self.text_size), self.text_font,
                       self.text_size, self.text_color, self.text_thick)

    def _draw_fps(self, image, fps):
        cv.putText(image, str(fps), (50, image.shape[0] - 50), self.text_font, self.text_size, (0, 0, 255),
                   self.text_thick)

    def _draw_result(self, image, result):
        cv.putText(image, result, (image.shape[1] - 100, image.shape[0] - 50), self.text_font, self.text_size,
                   self.text_color, self.text_thick)

    def _draw_main_text(self, image, text):
        cv.putText(image, text, (50, 50), self.text_font, self.text_size, self.text_color, self.text_thick)

    def _draw_subtext(self, image, text):
        cv.putText(image, text, (50, 50 + 30 * self.text_size), self.text_font, self.text_size * .7, self.text_color,
                   self.text_thick)


if __name__ == '__main__':
    machine1 = VendingMachine()
    machine1.launch()
