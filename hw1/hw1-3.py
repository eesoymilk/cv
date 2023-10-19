import cv2 as cv
import matplotlib.pyplot as plt

from typing import Sequence
from cv2.typing import MatLike

SiftDetectionResult = tuple[Sequence[cv.KeyPoint], MatLike]


def match_features(
    des1: MatLike, des2: MatLike
) -> Sequence[Sequence[cv.DMatch]]:
    """Match SIFT descriptors using BFMatcher and ratio test."""
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    return good


def sift(img1: MatLike, img2: MatLike):
    sift: cv.SIFT = cv.SIFT_create()

    res1: SiftDetectionResult = sift.detectAndCompute(img1, None)
    res2: SiftDetectionResult = sift.detectAndCompute(img2, None)
    kp1, des1 = res1
    kp2, des2 = res2

    good = match_features(des1, des2)

    return cv.drawMatchesKnn(
        img1,
        kp1,
        img2,
        kp2,
        good,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def main():
    img1 = cv.imread('hw1-3-1.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('hw1-3-2.jpg', cv.IMREAD_GRAYSCALE)

    img3 = sift(img1, img2)

    plt.imshow(img3)
    plt.tight_layout()
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
