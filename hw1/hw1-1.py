from operator import is_
from unicodedata import is_normalized
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from numpy.typing import NDArray


def showImage(img: Image.Image | NDArray):
    if img.mode == 'L':
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(img)
    plt.axis('off')
    plt.show()


# This function is not used
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


def get_histogram(img: Image.Image) -> NDArray:
    histogram = np.zeros(256, dtype=np.int32)
    for pixel in img.getdata():
        histogram[pixel] += 1

    for i in range(255):
        histogram[i + 1] += histogram[i]

    return histogram


def normalize_histogram(histogram: NDArray) -> NDArray:
    return histogram * 255 / histogram[255]


def plot_histogram(histogram: NDArray, is_normalized: bool = False):
    plt.figure()
    plt.xlabel('Gray Level')
    plt.xlim(0, 255)
    plt.bar(np.arange(256), histogram)

    if is_normalized:
        plt.title('Normalized Histogram')
        plt.ylabel('Output')
        plt.ylim(0, 255)
    else:
        plt.title('Histogram')
        plt.ylabel('Number of Pixels')

    plt.grid()
    plt.show()


def historgram_equalization(
    img: Image.Image, verbose: bool = False
) -> Image.Image:
    histogram = get_histogram(img)
    normalized_histogram = normalize_histogram(histogram)

    if verbose:
        plot_histogram(histogram)
        plot_histogram(normalized_histogram, is_normalized=True)

    return img.point(lambda x: normalized_histogram[x], 'L')


def image_comparison(img: Image.Image, equlized_img: Image.Image):
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.tight_layout()
    plt.title('Original Image')
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.tight_layout()
    plt.title('Equalized Image')
    plt.imshow(equlized_img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.show()


def main():
    img = Image.open('hw1-1.jpg')

    # convert the image to grayscale if it is not
    if img.mode != 'L':
        img = convert_to_gray(img)

    equlized_img = historgram_equalization(img, verbose=True)
    image_comparison(img, equlized_img)


if __name__ == '__main__':
    main()
