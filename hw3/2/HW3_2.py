from pathlib import Path
from typing import Sequence

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy.spatial import distance as scipy_distance

SiftDetectionResult = tuple[Sequence[cv.KeyPoint], MatLike]

script_dir = Path(__file__).parent.absolute()
output_dir = script_dir / 'output'


def showImageCV(
    img: MatLike, title: str | None = None, fname: str | None = None
):
    # Convert BGR image (OpenCV default) to RGB for visualization
    if len(img.shape) == 3:  # if the image has channels
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    else:
        img_rgb = img

    # Adjust figure size to match image aspect ratio
    dpi = 80
    height, width = img.shape[:2]
    figsize = width / dpi, height / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)

    # Display grayscale image if it's a single channel
    if len(img.shape) == 2 or img.shape[2] == 1:
        ax.imshow(img_rgb, cmap='gray', vmin=0, vmax=255)
    else:
        ax.imshow(img_rgb)

    ax.axis('off')

    # if title is not None:
    #     ax.set_title(title)

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)

    plt.show()


def kmeans(
    img: MatLike,
    k: int,
    n_initials: int = 50,
    threshold: float = 0.1,
    n_iterations: int = 10000,
):
    height, width, channels = img.shape
    img = img.reshape((-1, channels))

    # Randomly initialize k centroids and initialize cluster assignments
    centroids = np.random.randint(0, 256, size=(n_initials, channels))
    cluster_assignments = np.zeros((height * width, 1))

    # Initialize distance matrix and previous centroids
    distances = np.zeros((height * width, k))
    prev_centroids = np.zeros(centroids.shape)

    # Iterate until convergence or max iterations
    for _ in trange(n_iterations):
        # Compute pairwise distances
        distances = scipy_distance.cdist(img, centroids, 'euclidean')

        # Assign clusters
        cluster_assignments = np.argmin(distances, axis=1)

        # Update centroids
        for i in range(k):
            centroids[i] = np.mean(img[cluster_assignments == i], axis=0)

        # Check convergence
        if np.linalg.norm(centroids - prev_centroids) < threshold:
            break

        # Update previous centroids
        prev_centroids = centroids

    # Assign pixels to their cluster centroids
    img_clustered = np.zeros(img.shape)
    for i in range(k):
        img_clustered[cluster_assignments == i] = centroids[i]

    # Reshape image
    img_clustered = img_clustered.reshape((height, width, channels))

    return img_clustered


def kmeans_pp():
    ...


def meanshift():
    ...


def main():
    image = cv.imread('2-image.jpg')

    # perform k-means clustering
    img_clustered = kmeans(image, 3)
    showImageCV(img_clustered)


if __name__ == '__main__':
    main()
