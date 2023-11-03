from pathlib import Path

import cv2
import numpy as np

from cv2.typing import MatLike
from numpy.typing import NDArray


script_dir = Path(__file__).parent.absolute()
asset_dir = script_dir / 'assets'
output_dir = script_dir / 'output'


def find_homography(src: NDArray, dst: NDArray) -> NDArray:
    assert len(src) == len(dst) and len(src) >= 4, 'Need at least 4 point pairs'

    # Create the A matrix
    A = np.zeros((len(src) * 2, 9))

    print(src)
    print(dst)

    for i, (src_pt, dst_pt) in enumerate(zip(src, dst)):
        src_x, src_y = src_pt
        dst_x, dst_y = dst_pt

        A[i * 2] = [
            src_x,
            src_y,
            1,
            0,
            0,
            0,
            -dst_x * src_x,
            -dst_x * src_y,
            -dst_x,
        ]
        A[i * 2 + 1] = [
            0,
            0,
            0,
            src_x,
            src_y,
            1,
            -dst_y * src_x,
            -dst_y * src_y,
            -dst_y,
        ]

    # Compute the SVD of A
    _, _, Vt = np.linalg.svd(A)

    # The homography matrix is the last column of V
    h = Vt[-1].reshape((3, 3))

    return h / h[2, 2]


def draw_lines(img: MatLike, *lines: tuple[int, int]) -> MatLike:
    # Get image dimensions
    height, width, _ = img.shape

    for l in lines:
        pt1, pt2 = l

        # Ax + By = C
        A = pt2[1] - pt1[1]
        B = pt1[0] - pt2[0]
        C = A * pt1[0] + B * pt1[1]

        if B == 0:  # Vertical line
            cv2.line(img, (pt1[0], 0), (pt1[0], height), (0, 255, 0), 1)
        elif A == 0:  # Horizontal line
            cv2.line(img, (0, pt1[1]), (width, pt1[1]), (0, 255, 0), 1)
        else:
            intersections = np.array(
                [
                    [0, int(C / B)],
                    [int(C / A), 0],
                    [(width - 1), int((C - A * (width - 1)) / B)],
                    [int((C - B * (height - 1)) / A), (height - 1)],
                ]
            )

            intersections = intersections[
                (intersections[:, 0] >= 0)
                & (intersections[:, 0] < width)
                & (intersections[:, 1] >= 0)
                & (intersections[:, 1] < height)
            ]

            assert (
                len(intersections) == 2
            ), 'Line must intersect with image bounds'

            cv2.line(
                img,
                tuple(intersections[0]),
                tuple(intersections[1]),
                (0, 255, 0),
                1,
            )

    return img


def find_intersection(
    l1: tuple[tuple[int, int], tuple[int, int]],
    l2: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[int, int] | None:
    # Get the points defining each line
    p1, p2 = l1
    p3, p4 = l2

    # Calculate the determinant
    det = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])

    # If the determinant is zero, the lines are parallel
    if det == 0:
        return None

    # Calculate the intersection point
    x = (
        (p1[0] * p2[1] - p1[1] * p2[0]) * (p3[0] - p4[0])
        - (p1[0] - p2[0]) * (p3[0] * p4[1] - p3[1] * p4[0])
    ) / det
    y = (
        (p1[0] * p2[1] - p1[1] * p2[0]) * (p3[1] - p4[1])
        - (p1[1] - p2[1]) * (p3[0] * p4[1] - p3[1] * p4[0])
    ) / det

    return int(x), int(y)


class HomographyFinder:
    def __init__(self, img_src: MatLike, img_tar: MatLike):
        self.img_src: MatLike = img_src
        self.img_tar: MatLike = img_tar

        self.corners: list[tuple[int, int]] = []
        self.warped_corners: list[tuple[int, int]] = []
        self.img_warped: MatLike = img_tar.copy()
        self.warped = False

    def _mouse_callback(self, event: int, x: int, y: int, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corners) < 4:
            self.corners.append((x, y))

    def backward_warp(self, h: NDArray) -> MatLike:
        assert h.shape == (3, 3), 'Homography matrix must be 3x3'

        # Get the dimensions of the target and source images
        tar_height, tar_width, _ = self.img_tar.shape
        src_height, src_width, _ = self.img_src.shape
        h_inv = np.linalg.inv(h)

        # Coordinates in the target image
        x_tar, y_tar = np.meshgrid(np.arange(tar_width), np.arange(tar_height))
        ones = np.ones_like(x_tar.flatten())

        # Homogeneous coordinates
        homogenous_coords_tar = np.stack(
            (x_tar.flatten(), y_tar.flatten(), ones)
        )

        # Apply inverse homography
        homogenous_coords_src = h_inv @ homogenous_coords_tar
        zs = homogenous_coords_src[2, :]
        # Avoid division by zero or by very small numbers
        valid_idx = np.abs(zs) > 1e-10
        homogenous_coords_src[:, valid_idx] /= zs[valid_idx]

        # Non-homogeneous coordinates
        x_src, y_src = homogenous_coords_src[0], homogenous_coords_src[1]

        # Check for coordinates within bounds
        valid_coords = (
            (x_src >= 0)
            & (x_src < src_width)
            & (y_src >= 0)
            & (y_src < src_height)
            & valid_idx
        )

        x_src, y_src = x_src[valid_coords], y_src[valid_coords]
        x_tar, y_tar = (
            x_tar.flatten()[valid_coords],
            y_tar.flatten()[valid_coords],
        )

        # Calculate the integer parts and fractional parts of the source coordinates
        x0 = np.floor(x_src).astype(int)
        y0 = np.floor(y_src).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        # Ensure we don't go out of bounds
        x0 = np.clip(x0, 0, src_width - 1)
        y0 = np.clip(y0, 0, src_height - 1)
        x1 = np.clip(x1, 0, src_width - 1)
        y1 = np.clip(y1, 0, src_height - 1)

        # Calculate the weights for interpolation
        wx = x_src - x0
        wy = y_src - y0

        # Get the pixel values for each of the four neighbors
        Ia = self.img_src[y0, x0]
        Ib = self.img_src[y1, x0]
        Ic = self.img_src[y0, x1]
        Id = self.img_src[y1, x1]

        # Calculate the interpolated pixel values
        interpolated_values = (
            ((1 - wx) * (1 - wy)).reshape(-1, 1) * Ia
            + (wx * (1 - wy)).reshape(-1, 1) * Ic
            + ((1 - wx) * wy).reshape(-1, 1) * Ib
            + (wx * wy).reshape(-1, 1) * Id
        )

        # Map the interpolated values back to the target image coordinates
        # This should create an array with shape (number of valid coordinates, channels)
        self.img_warped[y_tar, x_tar] = interpolated_values

    def draw_vanishing_point(self):
        # Solve the system of equations to find the intersection point
        vanishing_point = find_intersection(
            self.corners[0:2], self.corners[2:4]
        )

        # Draw the vanishing point
        if vanishing_point is not None:
            cv2.circle(self.img_warped, vanishing_point, 5, (0, 0, 255), -1)

        return vanishing_point

    def run(self):
        cv2.namedWindow("Interactive window")
        cv2.setMouseCallback("Interactive window", self._mouse_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                self.corners = []
                self.img_warped = self.img_tar.copy()
                self.warped = False
            elif key == ord("p"):
                print(self.corners)

            if len(self.corners) == 4 and not self.warped:
                # define points in the target image (clockwise from the top left corner)
                src_height, src_width, _ = self.img_src.shape
                points_src = np.float32(
                    [
                        [0, 0],
                        [src_width - 1, 0],
                        [src_width - 1, src_height - 1],
                        [0, src_height - 1],
                    ]
                )
                points_dst = np.float32(self.corners)

                # Compute the homography matrix
                h = find_homography(points_src, points_dst)

                self.backward_warp(h)
                self.warped = True

                self.img_warped = draw_lines(
                    self.img_warped,
                    self.corners[0:2],
                    self.corners[2:4],
                )
                self.draw_vanishing_point()

            cv2.imshow("Interactive window", self.img_warped)

        cv2.destroyAllWindows()


def main():
    img_src = cv2.imread(f'{asset_dir / "post.png"}')
    img_tar = cv2.imread(f'{asset_dir / "display.jpg"}')

    finder = HomographyFinder(img_src, img_tar)
    finder.run()


if __name__ == "__main__":
    main()
