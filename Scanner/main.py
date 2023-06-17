import cv2 as cv
import numpy as np

from Scanner.utils import order_points

Dim = (960, 720)
############# Morphological Operation (Get A Blank Page)
##
original_img = cv.imread("src.jpg")
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Repeated Closing operation to remove text from the document.
kernel = np.ones((5, 5), np.uint8)
img = cv.morphologyEx(original_img, cv.MORPH_CLOSE, kernel, iterations=4)

cv.imshow('Blank Page', cv.resize(img, (Dim)))
cv.waitKey(0)

## Get rid of the background
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 15, cv.GC_INIT_WITH_RECT)

# im_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# (thresh, img) = cv.threshold(im_gray, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# img = cv.threshold(im_gray, 210, 255, cv.THRESH_BINARY)[1]


mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]

cv.imshow('Removing Background', cv.resize(img, Dim))
cv.waitKey(0)

######################## Edge Detection
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(img, (11, 11), 0)
canny = cv.Canny(gray, 0, 200)
## used to increase the size or thickness of the foreground object in an image.
# canny = cv.dilate(canny, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
cv.imshow('Edge Detection', cv.resize(canny, Dim))
cv.waitKey(0)
######################## Contour Detection
# Blank canvas.
con = np.zeros_like(img)
# Finding contours for the detected edges.
contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
# Keeping only the largest detected contour.
page = sorted(contours, key=cv.contourArea, reverse=True)[:5]
con = cv.drawContours(con, page, -1, (0, 255, 255), 3)

cv.imshow('Detecting Contour', cv.resize(con, Dim))
cv.waitKey(0)
######################## Corner Detection
# Blank canvas.
con = np.zeros_like(img)
# Loop over the contours.
for c in page:
    # Approximate the contour.
    epsilon = 0.02 * cv.arcLength(c, True)
    corners = cv.approxPolyDP(c, epsilon, True)
    # If our approximated contour has four points
    if len(corners) == 4:
        break
cv.drawContours(con, c, -1, (0, 255, 255), 3)
cv.drawContours(con, corners, -1, (0, 0, 255), 10)  # Red
# Sorting the corners and converting them to desired shape.
corners = sorted(np.concatenate(corners).tolist())
print(corners)
# Displaying the corners.
for index, c in enumerate(corners):
    character = chr(65 + index)
    cv.putText(con, character, tuple(c), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv.LINE_AA)

cv.imshow('Detecting Corners', cv.resize(con, Dim))
cv.waitKey(0)

corners = order_points(corners)


def get_max_width_and_max_height(corners):
    (tl, tr, br, bl) = corners
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    return max(int(widthA), int(widthB)), max(int(heightA), int(heightB))


maxWidth, maxHeight = get_max_width_and_max_height(corners)
# Final destination co-ordinates.
destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]


################## Getting the homography.
M = cv.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
# Perspective transform using homography.
scanned_img = cv.warpPerspective(original_img, M, (destination_corners[2][0], destination_corners[2][1]),
                                 flags=cv.INTER_LINEAR)
original_img = cv.resize(original_img, (scanned_img.shape[1], scanned_img.shape[0]))
print(original_img.shape)
print(scanned_img.shape)
diff_images = np.concatenate((original_img, scanned_img), axis=1)
cv.imshow('Original And Scanned Images', diff_images)
cv.waitKey(0)
