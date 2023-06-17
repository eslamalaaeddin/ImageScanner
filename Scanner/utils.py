import cv2 as cv
import numpy as np
def corner_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # img - Input image. It should be grayscale and float32 type.
    # blockSize - It is the size of neighbourhood considered for corner detection
    # ksize - Aperture parameter of the Sobel derivative used.
    # k - Harris detector free parameter in the equation.
    dst = cv.cornerHarris(gray, 10, 5, 0.04)  ## 200, 198
    # # result is dilated for marking the corners, not important
    # dst = cv.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    threshold = .1 * dst.max()
    image[dst > threshold] = [0, 0, 255]  # make it red corners
    # All pixels above a certain threshold are converted to white

    # Convert corners from white to red.
    # img[dst > 0.01 * dst.max()] = [0, 0, 255]

    # Create an array that lists all the pixels that are corners
    coordinates = np.argwhere(image)
    print(list(coordinates))
    return image


def order_points(pts):
    '''Rearrange coordinates to order:
      top-left, top-right, bottom-right, bottom-left'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # Return the ordered coordinates.
    return rect.astype('int').tolist()