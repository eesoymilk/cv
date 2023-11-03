from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray

script_dir = Path(__file__).parent.absolute()
asset_dir = script_dir / 'assets'
output_dir = script_dir / 'output'


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

    # Element-wise multiplication using broadcasting
    x1, y1, _ = pts1.T  # Transpose to unpack efficiently
    x2, y2, _ = pts2.T

    # Stack arrays in columns to form the A matrix
    A = np.column_stack(
        [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, np.ones_like(x1)]
    )

    return A


def get_rank2_matrix(m: NDArray[np.float64]) -> NDArray[np.float64]:
    '''Enforce the rank-2 constraint on the matrix m.'''
    assert m.shape == (3, 3), "Matrix must be 3x3."

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
    return translate_mat1.T @ f @ translate_mat2


def get_epipolar_lines(
    pts: NDArray[np.float64],
    F: NDArray[np.float64],
    image_height: int,
    image_width: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''Calculate the epipolar lines for the given points and fundamental matrix.'''
    assert F.shape == (3, 3), "Fundamental matrix must be 3x3."

    # Calculate the epipolar lines
    lines = pts @ F

    # Generate a set of x values to plot the lines against
    x_vals = np.linspace(0, image_width, 100)

    # Calculate the corresponding y values for each line
    y_vals = (
        np.outer(-lines[:, 2], np.ones_like(x_vals))
        - lines[:, 0].reshape(-1, 1) * x_vals
    )
    y_vals /= lines[:, 1].reshape(-1, 1)

    # Filter out y-values that are outside the image bounds
    y_vals[(y_vals < 0) | (y_vals >= image_height)] = np.nan

    return x_vals, y_vals.T


def plot_epipolar_lines_and_pts(
    image: NDArray[np.float64],
    x_vals: NDArray[np.float64],
    y_vals: NDArray[np.float64],
    pts: NDArray[np.float64],
    ax: plt.Axes,
):
    '''Plot the epipolar lines and points on the image.'''
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.imshow(image)

    for pt, y in zip(pts, y_vals.T):
        ax.plot(x_vals, y, linewidth=1)
        ax.scatter(pt[0], pt[1], s=10)


def test_fundamental_mat(
    image1: NDArray[np.float64],
    image2: NDArray[np.float64],
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    f: NDArray[np.float64],
    title: str | None = None,
):
    '''Plot the epipolar lines and points on the images.'''
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    axs: list[plt.Axes] = plt.subplots(1, 2, figsize=(15, 8))[1]

    x_vals1, y_vals1 = get_epipolar_lines(pts2, f.T, h1, w1)
    x_vals2, y_vals2 = get_epipolar_lines(pts1, f, h2, w2)

    plot_epipolar_lines_and_pts(image1, x_vals1, y_vals1, pts1, axs[0])
    plot_epipolar_lines_and_pts(image2, x_vals2, y_vals2, pts2, axs[1])

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()


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
    f_wo_normalized = eight_point_algorithm(pts1, pts2, normalize=False)
    f_normalized = eight_point_algorithm(pts1, pts2, normalize=True)

    test_fundamental_mat(
        image1,
        image2,
        pts1,
        pts2,
        f_wo_normalized,
        title='Epipolar lines without normalization',
    )
    test_fundamental_mat(
        image1,
        image2,
        pts1,
        pts2,
        f_normalized,
        title='Epipolar lines with normalization',
    )

    # Calculate the average distances
    avg_dist_wo_normalized = average_epipolar_distance(
        pts1, pts2, f_wo_normalized
    )
    avg_dist_normalized = average_epipolar_distance(pts1, pts2, f_normalized)

    result_str = '\n'.join(
        [
            'Average epipolar distance',
            f' - No normalization: {avg_dist_wo_normalized}',
            f' - Normalization: {avg_dist_normalized}',
        ]
    )

    print(result_str)


if __name__ == "__main__":
    main()
