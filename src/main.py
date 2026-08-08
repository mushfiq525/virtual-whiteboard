# import cv2
# cap = cv2.VideoCapture(0)  # try 0, 1, 2 until you see your phone feed
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     cv2.imshow("Webcam Test", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

import cv2

cap = cv2.VideoCapture(0)

# 1. Create a named window with flexible scaling
cv2.namedWindow("Webcam Test", cv2.WINDOW_NORMAL)

# 2. Set the window display size (width, height)
cv2.resizeWindow("Webcam Test", 960, 540)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Webcam Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()