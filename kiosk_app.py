import gesture
import cv2 as cv
import time
from PIL import Image


class Kiosk:
    def __init__(self):
        self.rec = gesture.Recognizer(model_name='mobilenet224')

    def __call__(self, img):
        return self.rec(img)


if __name__ == '__main__':
    kiosk = Kiosk()
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    print(cap.isOpened())
    prev_time = time.time()
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        image = cv.flip(image, 1)
        image.flags.writeable = False
        image = image[:, :, [2, 1, 0]]
        result = kiosk(Image.fromarray(image))
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image.flags.writeable = True
        cv.putText(image, result, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), thickness=2)

        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        cv.putText(image, str(fps), (50, image.shape[0] - 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        cv.imshow('camera', image)
        prev_time = curr_time
        if cv.waitKey(5) & 0xFF == 27:
            break
    cap.release()
