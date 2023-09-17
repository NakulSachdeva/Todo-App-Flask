import cv2
import time
from datetime import datetime
import argparse
import os

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)
counter = 0

while True:
    check, frame = video.read()

    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
        
        for x, y, w, h in faces:
            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            exact_time = datetime.now().strftime('%Y-%b-%d-%H-%M-%S-%f')
            file_name = "C:/Users/dell/Desktop/Computer Vision/project/face_detected_{}.jpg".format(counter)
            cv2.imwrite(file_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print("Saved:", file_name)
            counter += 1

        cv2.imshow("home surv", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

video.release()
cv2.destroyAllWindows()

dir_path = "C:/Users/dell/Desktop/Computer Vision/project/"
ext = "jpg"
output = "C:/Users/dell/Desktop/Computer Vision/project/output.mp4"

images = [f for f in os.listdir(dir_path) if f.endswith(ext)]
images.sort()

if len(images) > 0:
    first_image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, channels = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, 5.0, (width, height))

    for image in images:
        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)
        out.write(frame)

    out.release()
