import cv2
import matplotlib.pyplot as plt


filename = "../../data/raw/raw-images/b00-kings-pawn-game-lichess.png"
img = cv2.imread(filename)
h, w, c = img.shape
print(h, w, c)
print(img[0][0])
new_size = (256, 256)
rimg = cv2.resize(img, new_size)
plt.subplot(121),plt.imshow(img), plt.title("Original Image")
plt.subplot(122), plt.imshow(rimg), plt.title("Resized Image")
plt.show()
