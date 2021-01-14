from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int,
                default=500, help="minimum area size")
args = vars(ap.parse_args())
if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    vs = cv2.VideoCapture(args["video"])
firstFrame = None
iteration = 0
haar_cascade_face = cv2.CascadeClassifier('./sample.xml')
while True:
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]
    text = "False"
    if frame is None:
        break
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rects = haar_cascade_face.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5);
    hStatus=len(faces_rects);
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if firstFrame is None:
        firstFrame = gray
        continue
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "True"
    cv2.putText(frame, "Motion detection status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Human Faces: {}".format(hStatus), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imshow("Security Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    iteration+=1;

    if(hStatus >= 1):
        print("Humans detected");
        print("Faces found {}".format(hStatus));
    if(((text == "True") and (iteration > 500)) or iteration > 5000):
        firstFrame = None
        iteration = 0
    if key == ord("q") or key == ord("x"):
        break
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
exit();
