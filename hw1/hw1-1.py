import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.typing import NDArray


def showImage(img: NDArray):
    # Convert the image from BGR to RGB (as OpenCV loads in BGR and matplotlib displays in RGB)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.axis('off')  # to hide axis values
    plt.show()


def convert_to_gray(
    img: NDArray, rgb_weights: tuple = (0.299, 0.587, 0.114)
) -> NDArray:
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    r_weight, g_weight, b_weight = rgb_weights
    gray = (r_weight * r + g_weight * g + b_weight * b).astype(np.uint8)
    return gray


def histogram_equalization(img: NDArray) -> NDArray:
    ...


def main():
    img: NDArray = cv.imread('hw1-1.jpg')

    gray_img = convert_to_gray(img)

    showImage(img)


if __name__ == '__main__':
    main()
