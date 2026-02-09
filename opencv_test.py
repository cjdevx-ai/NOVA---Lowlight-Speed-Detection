import cv2, numpy as np
img = np.zeros((300,500,3), dtype=np.uint8)
cv2.imshow("test", img)
cv2.waitKey(1000)
cv2.destroyAllWindows()
print("ok")
