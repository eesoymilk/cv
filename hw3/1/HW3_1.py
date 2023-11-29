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


def bf_match(
    sift_des1: MatLike,
    sift_des2: MatLike,
) -> Sequence[cv.DMatch]:
    """Optimized brute-force matching with ratio test using matrix operations."""
    # Compute pairwise distances
    distances = scipy_distance.cdist(sift_des1, sift_des2, 'euclidean')

    # Find two smallest distances
    sorted_indices = np.argsort(distances, axis=1)
    smallest_distances = distances[
        np.arange(len(sift_des1)), sorted_indices[:, 0]
    ]
    second_smallest_distances = distances[
        np.arange(len(sift_des1)), sorted_indices[:, 1]
    ]

    # Apply ratio test
    mask = smallest_distances < 0.75 * second_smallest_distances

    # Create DMatch objects
    matches = [
        cv.DMatch(i, sorted_indices[i, 0], distances[i, sorted_indices[i, 0]])
        for i in np.where(mask)[0]
    ]

    return matches


def find_homography(src: NDArray, dst: NDArray) -> NDArray:
    assert len(src) == len(dst) and len(src) >= 4, 'Need at least 4 point pairs'

    # Create the A matrix
    A = np.zeros((len(src) * 2, 9))

    for i, (src_pt, dst_pt) in enumerate(zip(src, dst)):
        x1, y1, _ = src_pt
        x2, y2, _ = dst_pt

        A[i * 2] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]
        A[i * 2 + 1] = [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]

    # Compute the SVD of A
    _, _, Vt = np.linalg.svd(A)

    # The homography matrix is the last column of V
    h = Vt[-1].reshape((3, 3))

    return h / h[2, 2]


def ransac(
    matches: Sequence[cv.DMatch],
    src_kps: Sequence[cv.KeyPoint],
    des_kps: Sequence[cv.KeyPoint],
    threshold: float = 10.0,
    n_iterations: int = 1000,
):
    """RANSAC implementation for homography estimation."""
    n_matches = len(matches)
    assert n_matches >= 4, 'Need at least 4 matches'

    # Convert keypoints to numpy arrays
    src_pts = np.float64([src_kps[m.queryIdx].pt for m in matches])
    des_pts = np.float64([des_kps[m.trainIdx].pt for m in matches])

    # Add ones to convert to homogeneous coordinates
    src_pts = np.hstack((src_pts, np.ones((n_matches, 1))))
    des_pts = np.hstack((des_pts, np.ones((n_matches, 1))))

    # Initialize best transformation
    best_H = np.zeros((3, 3))
    best_inliers = np.zeros_like(matches, dtype=bool)

    # RANSAC loop
    for _ in trange(n_iterations, desc='RANSAC'):
        # Randomly select 4 matches
        idx = np.random.choice(len(matches), 4, replace=False)
        src = src_pts[idx]
        dst = des_pts[idx]

        # Estimate homography
        H = find_homography(src, dst)

        # Transform source points
        src_pts_transformed = H @ src_pts.T
        src_pts_transformed = src_pts_transformed.T
        src_pts_transformed = (
            src_pts_transformed[:, :2] / src_pts_transformed[:, 2:]
        )

        # Compute Euclidean distance
        dist = np.linalg.norm(src_pts_transformed - des_pts[:, :2], axis=1)

        # Count inliers
        n_inliers = np.sum(dist < threshold)

        # Update best transformation
        if n_inliers > np.sum(best_inliers):
            best_H = H
            best_inliers = dist < threshold

    best_matches = [m for m, inlier in zip(matches, best_inliers) if inlier]

    return best_H, best_matches


def main():
    image = cv.imread('1-image.jpg')
    books = [
        cv.imread('1-book1.jpg'),
        cv.imread('1-book2.jpg'),
        cv.imread('1-book3.jpg'),
    ]

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_books = [cv.cvtColor(book, cv.COLOR_BGR2GRAY) for book in books]

    sift: cv.SIFT = cv.SIFT_create()

    image_sift_result: SiftDetectionResult = sift.detectAndCompute(
        gray_image, None
    )
    image_kps, image_des = image_sift_result

    books_sift_results: list[SiftDetectionResult] = [
        sift.detectAndCompute(gray_book, None)
        for gray_book in tqdm(gray_books, desc='Computing SIFT')
    ]

    matches_list = [
        bf_match(des, image_des)
        for _, des in tqdm(books_sift_results, desc='Matching')
    ]

    results_wo_ransac = [
        cv.drawMatchesKnn(
            book,
            kps,
            image,
            image_kps,
            [matches[:500]],
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        for book, (kps, _), matches in tqdm(
            zip(books, books_sift_results, matches_list), desc='Drawing matches'
        )
    ]

    for i, result in enumerate(results_wo_ransac):
        showImageCV(result, fname=output_dir / f'1-a{i}.jpg')

    ransacs_matches = [
        ransac(matches, kps, image_kps)
        for (kps, _), matches in zip(books_sift_results, matches_list)
    ]

    results_ransac = [
        cv.drawMatchesKnn(
            book,
            kps,
            image,
            image_kps,
            [matches[:500]],
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        for book, (kps, _), (_, matches) in zip(
            books, books_sift_results, ransacs_matches
        )
    ]

    for i, result in enumerate(results_ransac):
        showImageCV(result, fname=output_dir / f'1-b{i}.jpg')


if __name__ == '__main__':
    main()
