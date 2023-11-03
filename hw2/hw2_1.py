from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray

script_dir = Path(__file__).parent.absolute()
asset_dir = script_dir / 'assets'


def read_pts(fname: Path | str) -> NDArray[np.float64]:
    '''Read points from a text file and return them as homogeneous coordinates.'''

    with open(fname) as f:
        lines = f.readlines()

    # First line contains the number of points
    n_pts = int(lines[0].strip())

    # The rest of the lines contain the point coordinates
    pts = np.array(
        [list(map(float, line.strip().split())) for line in lines[1:]]
    )

    assert (
        pts.shape[0] == n_pts
    ), "Number of points does not match the file's first line."

    # Convert to homogeneous coordinates
    pts = np.hstack((pts, np.ones((n_pts, 1))))

    return pts


def normalize_pts(
    pts: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''Normalize the points so that they have centroid at the origin and average distance from the origin is sqrt(2).'''

    # Translate pts to have centroid at the origin
    centroid: np.float64 = np.mean(pts, axis=0)
    shifted_pts = pts - centroid

    # Scale points so that the average distance from the origin is sqrt(2)
    scale: np.float64 = np.sqrt(2) / np.mean(
        np.sqrt(np.sum(shifted_pts**2, axis=1))
    )
    translate_mat = np.array(
        [
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1],
        ]
    ).astype(np.float64)
    normalized_pts = (translate_mat @ shifted_pts.T).T

    return normalized_pts, translate_mat


def construct_matrix_A(
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
) -> NDArray[np.float64]:
    '''Construct the matrix A used in the eight-point algorithm.'''
    assert pts1.shape == pts2.shape, "Points must have the same shape."

    A = np.zeros((pts1.shape[0], 9))
    A[:, 0] = pts1[:, 0] * pts2[:, 0]
    A[:, 1] = pts1[:, 0] * pts2[:, 1]
    A[:, 2] = pts1[:, 0]
    A[:, 3] = pts1[:, 1] * pts2[:, 0]
    A[:, 4] = pts1[:, 1] * pts2[:, 1]
    A[:, 5] = pts1[:, 1]
    A[:, 6] = pts2[:, 0]
    A[:, 7] = pts2[:, 1]
    A[:, 8] = 1

    return A


def get_rank2_matrix(m: NDArray[np.float64]) -> NDArray[np.float64]:
    '''Enforce the rank-2 constraint on the matrix m.'''
    U, S, Vt = np.linalg.svd(m)
    S[-1] = 0  # Set the smallest singular value to zero
    m_rank2 = U @ np.diag(S) @ Vt
    return m_rank2


def eight_point_algorithm(
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    normalize: bool = True,
) -> NDArray[np.float64]:
    '''Compute the fundamental matrix using the eight-point algorithm.'''
    if normalize:
        # Normalization of the points
        pts1, translate_mat1 = normalize_pts(pts1)
        pts2, translate_mat2 = normalize_pts(pts2)
    else:
        translate_mat1 = translate_mat2 = np.eye(3)

    # Construct matrix A
    A = construct_matrix_A(pts1, pts2)

    # Solve for the fundamental matrix using the linear least squares method
    Vt: NDArray[np.float64] = np.linalg.svd(A)[2]
    f = Vt[-1].reshape(3, 3)

    # Enforce the rank-2 constraint
    f = get_rank2_matrix(f)

    # Return the denormalized fundamental matrix if normalization is used
    return f if not normalize else translate_mat1.T @ f @ translate_mat2


def plot_epipolar_lines(
    image: NDArray[np.float64],
    pts: NDArray[np.float64],
    F: NDArray[np.float64],
    ax: plt.Axes,
):
    '''Plot the epipolar lines on the image.'''
    height, width = image.shape[:2]

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    for pt in pts:
        # Line equation: ax + by + c = 0
        line = F @ np.array([pt[0], pt[1], 1])

        # Points for plotting the line
        x = np.linspace(0, width, 100)
        y = -(line[2] + line[0] * x) / line[1]

        # Filter out points outside the image
        mask = (y >= 0) & (y < height)
        x = x[mask]
        y = y[mask]

        ax.plot(x, y, linewidth=1)


def average_epipolar_distance(
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    F: NDArray[np.float64],
):
    '''Calculate the average distance from points to their epipolar lines.'''
    lines = F @ pts2.T
    distances = np.abs(np.sum(lines * pts1.T, axis=0)) / np.sqrt(
        lines[0, :] ** 2 + lines[1, :] ** 2
    )
    return np.mean(distances)


def main():
    # Read points from the text files as homogeneous coordinates
    pts1 = read_pts(asset_dir / 'pt_2D_1.txt')
    pts2 = read_pts(asset_dir / 'pt_2D_2.txt')

    # Load images
    image1 = plt.imread(asset_dir / 'image1.jpg')
    image2 = plt.imread(asset_dir / 'image2.jpg')

    # Compute the fundamental matrices
    F_wo_normalized = eight_point_algorithm(pts1, pts2, normalize=False)
    F_normalized = eight_point_algorithm(pts1, pts2, normalize=True)

    # Plot the epipolar lines for the unnormalized fundamental matrix
    ax: list[plt.Axes] = plt.subplots(1, 2, figsize=(15, 8))[1]
    ax[0].imshow(image1)
    plot_epipolar_lines(image1, pts1, F_wo_normalized, ax[0])
    ax[1].imshow(image2)
    plot_epipolar_lines(image2, pts2, F_wo_normalized.T, ax[1])
    plt.suptitle('Epipolar lines without normalization')
    plt.tight_layout()
    plt.show()

    # Plot the epipolar lines for the normalized fundamental matrix
    ax: list[plt.Axes] = plt.subplots(1, 2, figsize=(15, 8))[1]
    ax[0].imshow(image1)
    plot_epipolar_lines(image1, pts1, F_normalized, ax[0])
    ax[1].imshow(image2)
    plot_epipolar_lines(image2, pts2, F_normalized.T, ax[1])
    plt.suptitle('Epipolar lines with normalization')
    plt.tight_layout()
    plt.show()

    # Calculate the average distances
    avg_dist_wo_normalized = average_epipolar_distance(
        pts1, pts2, F_wo_normalized
    )
    avg_dist_normalized = average_epipolar_distance(pts1, pts2, F_normalized)

    print(
        f"Average epipolar distance without normalization: {avg_dist_wo_normalized}"
    )
    print(
        f"Average epipolar distance with normalization: {avg_dist_normalized}"
    )


if __name__ == "__main__":
    main()
