import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from typing import Sequence
from cv2.typing import MatLike

SiftDetectionResult = tuple[Sequence[cv.KeyPoint], MatLike]


def showImage(
    img: MatLike,
    title: str | None = None,
    fname: str | None = None,
):
    plt.imshow(img)
    plt.tight_layout()
    plt.axis('off')

    if title is not None:
        plt.title(title)

    if fname is not None:
        plt.savefig(fname)

    plt.show()


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


def bf_match(
    sift_res1: SiftDetectionResult,
    sift_res2: SiftDetectionResult,
    matches_per_object: int = 20,
    n_objects: int = 3,
) -> Sequence[Sequence[cv.DMatch]]:
    """Optimized brute-force matching with ratio test using matrix operations."""

    _, des1 = sift_res1
    kp2, des2 = sift_res2

    # Compute pairwise distances
    distances = np.linalg.norm(des1[:, np.newaxis] - des2, axis=2)

    # Find two smallest distances
    sorted_indices = np.argsort(distances, axis=1)
    smallest_distances = distances[np.arange(len(des1)), sorted_indices[:, 0]]
    second_smallest_distances = distances[
        np.arange(len(des1)), sorted_indices[:, 1]
    ]

    # Apply ratio test
    mask = smallest_distances < 0.75 * second_smallest_distances

    # Create DMatch objects
    matches = [
        cv.DMatch(i, sorted_indices[i, 0], distances[i, sorted_indices[i, 0]])
        for i in np.where(mask)[0]
    ]

    # Extract the keypoints' (x, y) coordinates for the matched features
    matched_kp_coords = np.array([kp2[m.trainIdx].pt for m in matches])

    # Use kmeans2 from scipy to cluster keypoints into n_objects groups
    centroids, labels = kmeans2(matched_kp_coords, n_objects, minit='points')

    # Get the top 'matches_per_object' matches for each cluster
    final_matches = []
    for i in range(n_objects):
        cluster_indices = np.where(labels == i)[0]
        cluster_matches = [matches[idx] for idx in cluster_indices]

        # Sort matches in this cluster by distance
        cluster_matches = sorted(cluster_matches, key=lambda x: x.distance)

        # Add top matches from this cluster
        final_matches.extend(cluster_matches[:matches_per_object])

    # For drawMatchesKnn, the matches need to be in a specific format
    final_matches = [[m] for m in final_matches]

    return final_matches


def sift(img1: MatLike, img2: MatLike) -> MatLike:
    sift: cv.SIFT = cv.SIFT_create()

    res1: SiftDetectionResult = sift.detectAndCompute(img1, None)
    res2: SiftDetectionResult = sift.detectAndCompute(img2, None)
    kp1, _ = res1
    kp2, _ = res2

    matches = bf_match(res1, res2)

    return cv.drawMatchesKnn(
        img1,
        kp1,
        img2,
        kp2,
        matches,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def main():
    img1 = cv.imread('hw1-3-1.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('hw1-3-2.jpg', cv.IMREAD_GRAYSCALE)

    # Perform SIFT object recognition on original images
    img3 = sift(img1, img2)

    # Scale the scene image by 2.0x
    img1_2x = cv.resize(
        img1, None, fx=2.0, fy=2.0, interpolation=cv.INTER_LINEAR
    )

    # Perform SIFT object recognition on scaled image
    img4 = sift(img1_2x, img2)

    # Show the results
    showImage(img3, 'Matching result', fname='hw1-3_1x.jpg')
    showImage(img4, 'Matching result (scaled 2x)', fname='hw1-3_2x.jpg')


if __name__ == '__main__':
    main()
