import gesture
import cv2 as cv
from PIL import Image
import time


class Kiosk:
    def __init__(self):
        self.rec = gesture.Recognizer()

    def __call__(self, img):
        return self.rec(img)


if __name__ == '__main__':
    kiosk = Kiosk()
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    print(cap.isOpened())
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        result = kiosk(Image.fromarray(image))
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.putText(image, result, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv.imshow('camera', image) # you can do cv.flip(image, 1)
        if cv.waitKey(5) & 0xFF == 27:
            break
    cap.release()

