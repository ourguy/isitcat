import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
ap.add_argument("-c", "--cascade", default="cascade.xml", help="path to cat detector")
args = vars(ap.parse_args())

img = cv2.imread(args['image'])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detector = cv2.CascadeClassifier(args['cascade'])
rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=10, minSize=(10, 10))

for (i, (x, y, w, h)) in enumerate(rects):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(img, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

cv2.imshow("Cat Faces", img)
cv2.waitKey(0)
