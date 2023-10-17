import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from numpy.typing import NDArray


def showImage(img: Image.Image | NDArray):
    plt.imshow(img)
    plt.axis('off')  # to hide axis values
    plt.show()


def convert_to_gray(
    img: Image.Image | NDArray, rgb_weights: tuple = (0.299, 0.587, 0.114)
) -> Image.Image:
    img_array = np.array(img)
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    r_weight, g_weight, b_weight = rgb_weights
    gray_img_array = (r_weight * r + g_weight * g + b_weight * b).astype(
        np.uint8
    )
    return Image.fromarray(gray_img_array, 'L')


def histogram_equalization(img: NDArray) -> NDArray:
    ...


def main():
    img = Image.open('hw1-1.jpg')
    gray_img = convert_to_gray(img)
    showImage(gray_img)


if __name__ == '__main__':
    main()
