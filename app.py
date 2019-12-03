import cv2
import pandas
from datetime import datetime

first_frame = None
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
state = 0
times = []
df = pandas.DataFrame(columns=["Start", "End"])

# Process input from webcam
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    check, frame = video.read()
    frame_state = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    delta = cv2.absdiff(first_frame, gray)
    threshold = cv2.threshold(delta, 40, 255, cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold, None, iterations=2)

    (contours, _) = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, h, w) = cv2.boundingRect(contour)
        frame_state = 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for x_pos, y_pos, wi, he in faces:
        cv2.rectangle(frame, (x_pos, y_pos), (x_pos + wi, y_pos + he), (255, 0, 0), 3)

    cv2.imshow("Capture", frame)
    #cv2.imshow("Delta", delta)
    #cv2.imshow("Threshold", threshold)
    key = cv2.waitKey(1)

    if state != frame_state:
        times.append(datetime.now())
        state = frame_state

    if key == ord("q"):
        if frame_state == 1:
            times.append(datetime.now())
        break


for i in range(0 , len(times), 2):
    df = df.append({"Start": times[i], "End": times[i + 1]}, ignore_index=True)

df.to_csv("times.csv")

video.release()
cv2.destroyAllWindows()
