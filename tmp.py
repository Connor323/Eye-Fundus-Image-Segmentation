import cv2
img = cv2.imread("gt.png")
img = cv2.resize(img, (300, 300))
cv2.imwrite("gt.png", img)