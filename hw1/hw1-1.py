import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from numpy.typing import NDArray

script_dir = Path(__file__).parent.absolute()
asset_dir = script_dir / 'assets'
output_dir = script_dir / 'output'


def showImage(img: Image.Image, fname: str | None = None):
    # Adjust figure size to match image aspect ratio
    dpi = 80
    width, height = img.size
    figsize = width / dpi, height / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)

    if img.mode == 'L':
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        ax.imshow(img)

    ax.axis('off')

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)

    plt.show()


def showHistogram(
    hist: NDArray,
    fname: str | None = None,
):
    plt.figure()
    plt.tight_layout()
    plt.xlabel('Gray Level')
    plt.xlim(0, 255)
    plt.ylabel('Number of Pixels')
    plt.bar(np.arange(256), hist)
    plt.grid()

    if fname is not None:
        plt.savefig(fname)
    plt.show()


def get_histogram(img: Image.Image) -> NDArray:
    histogram = np.zeros(256, dtype=np.int32)
    for pixel in img.getdata():
        histogram[pixel] += 1

    return histogram


def histogram_equalization(
    img: Image.Image,
) -> tuple[NDArray, Image.Image, NDArray]:
    histogram = get_histogram(img)
    cdf = np.cumsum(histogram)
    normalized_cdf = cdf * 255 / cdf[255]

    equalized_img = img.point(lambda x: normalized_cdf[x], 'L')
    equalized_histogram = get_histogram(equalized_img)

    return img, histogram, equalized_img, equalized_histogram


def image_comparison(img: Image.Image, equalized_img: Image.Image):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.tight_layout()
    plt.title('Original Image')
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.tight_layout()
    plt.title('Equalized Image')
    plt.imshow(equalized_img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    # save equalized image alone
    equalized_img.save(output_dir / 'backrooms.jpg')

    # plt.savefig('hw1-1_images.png')
    plt.show()


def histogram_comparison(histogram: NDArray, equalized_histogram: NDArray):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.tight_layout()
    plt.title('Original Histogram')
    plt.xlabel('Gray Level')
    plt.xlim(0, 255)
    plt.bar(np.arange(256), histogram)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
    plt.title('Equalized Histogram')
    plt.xlabel('Gray Level')
    plt.xlim(0, 255)
    plt.bar(np.arange(256), equalized_histogram)
    plt.grid()

    # plt.savefig('hw1-1_histograms.png')
    plt.show()


def main():
    img = Image.open(asset_dir / 'backrooms.jpg')
    img = img.convert('L')
    img, hist, equalized_img, equalized_hist = histogram_equalization(img)
    image_comparison(img, equalized_img)
    histogram_comparison(hist, equalized_hist)


if __name__ == '__main__':
    main()
