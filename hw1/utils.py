import matplotlib.pyplot as plt
from PIL import Image
from numpy.typing import NDArray


def showImage(img: Image.Image | NDArray, title: str | None = None):
    if img.mode == 'L':
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(img)

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.axis('off')
    plt.show()
