from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray

script_dir = Path(__file__).parent.absolute()
asset_dir = script_dir / 'assets'
output_dir = script_dir / 'output'


def read_pts(fname: Path | str) -> NDArray[np.float64]:
    '''
    Read points from a text file where the first line contains the number of points and the rest of the lines contain the point coordinates.
    Returns the points in homogeneous coordinates.
    '''
    with open(fname) as f:
        lines = f.readlines()

    n_pts = int(lines[0].strip())
    pts = np.array([list(map(float, l.strip().split())) for l in lines[1:]])

    return np.hstack((pts, np.ones((n_pts, 1))))


def normalize_pts(
    pts: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''Normalize the points so that the average distance from the origin is sqrt(2).'''
    centroid: np.float64 = np.mean(pts, axis=0)
    shifted_pts = pts - centroid

    # Scale points so that the avg dist from the origin is sqrt(2)
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
    norm_pts = (translate_mat @ shifted_pts.T).T

    return norm_pts, translate_mat


def construct_matrix_A(
    pts1: NDArray[np.float64], pts2: NDArray[np.float64]
) -> NDArray[np.float64]:
    '''Construct the A matrix for the linear least squares method.'''
    x1, y1, _ = pts1.T
    x2, y2, _ = pts2.T

    # Stack arrays in columns to form the A matrix
    A = np.column_stack(
        [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, np.ones_like(x1)]
    )

    return A


def get_rank2_matrix(m: NDArray[np.float64]) -> NDArray[np.float64]:
    '''Enforce the rank-2 constraint on the 3x3 matrix m.'''
    assert m.shape == (3, 3), "Matrix must be 3x3."

    U, S, Vt = np.linalg.svd(m)
    S[-1] = 0  # Set the smallest singular value to zero
    m_rank2 = U @ np.diag(S) @ Vt
    return m_rank2


def eight_point_algorithm(
    pts1: NDArray[np.float64], pts2: NDArray[np.float64], normalize: bool = True
) -> NDArray[np.float64]:
    '''Compute the fundamental matrix using the eight-point algorithm.'''
    if normalize:
        pts1, translate_mat1 = normalize_pts(pts1)
        pts2, translate_mat2 = normalize_pts(pts2)
    else:
        translate_mat1 = translate_mat2 = np.eye(3)

    A = construct_matrix_A(pts1, pts2)

    # Solve for the fundamental matrix using the linear least squares method
    Vt: NDArray[np.float64] = np.linalg.svd(A)[2]
    f = Vt[-1].reshape(3, 3)

    # Enforce the rank-2 constraint
    f = get_rank2_matrix(f)

    return translate_mat1.T @ f @ translate_mat2


def get_epipolar_lines(
    pts: NDArray[np.float64],
    F: NDArray[np.float64],
    image_height: int,
    image_width: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''Calculate the epipolar lines for the given points and fundamental matrix.'''
    lines = pts @ F
    x_vals = np.linspace(0, image_width, 100)
    y_vals = (
        np.outer(-lines[:, 2], np.ones_like(x_vals))
        - lines[:, 0].reshape(-1, 1) * x_vals
    )
    y_vals /= lines[:, 1].reshape(-1, 1)
    y_vals[(y_vals < 0) | (y_vals >= image_height)] = np.nan

    return x_vals, y_vals.T


def plot_epipolar_and_pts(
    image: NDArray[np.float64],
    x_vals: NDArray[np.float64],
    y_vals: NDArray[np.float64],
    pts: NDArray[np.float64],
    fname: str | None = None,
):
    '''Plot the epipolar lines and points on the image.'''
    dpi = 80
    height, width = image.shape[:2]
    figsize = width / dpi, height / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.imshow(image)
    ax.axis('off')

    for pt, y in zip(pts, y_vals.T):
        ax.plot(x_vals, y, linewidth=1)
        ax.scatter(pt[0], pt[1], s=10)

    if fname is not None:
        plt.savefig(output_dir / fname, bbox_inches='tight', pad_inches=0)

    plt.show()


def cmp_fund_mat(
    image1: NDArray[np.float64],
    image2: NDArray[np.float64],
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    f: NDArray[np.float64],
    norm_f: NDArray[np.float64],
):
    '''Compare the fundamental matrices and plot the epipolar lines.'''
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    x1, y1 = get_epipolar_lines(pts2, f.T, h1, w1)
    x2, y2 = get_epipolar_lines(pts1, f, h2, w2)
    norm_x1, norm_y1 = get_epipolar_lines(pts2, norm_f.T, h1, w1)
    norm_x2, norm_y2 = get_epipolar_lines(pts1, norm_f, h2, w2)

    plot_epipolar_and_pts(image1, x1, y1, pts1, 'wo_normalized_img1.jpg')
    plot_epipolar_and_pts(image2, x2, y2, pts2, 'wo_normalized_img2.jpg')
    plot_epipolar_and_pts(image1, norm_x1, norm_y1, pts1, 'normalized_img1.jpg')
    plot_epipolar_and_pts(image2, norm_x2, norm_y2, pts2, 'normalized_img2.jpg')


def avg_epipolar_dist(
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    f: NDArray[np.float64],
):
    '''Calculate the average distance from points to their epipolar lines.'''
    lines = f @ pts2.T
    distances = np.abs(np.sum(lines * pts1.T, axis=0)) / np.sqrt(
        lines[0, :] ** 2 + lines[1, :] ** 2
    )
    return np.mean(distances)


def main():
    pts1 = read_pts(asset_dir / 'pt_2D_1.txt')
    pts2 = read_pts(asset_dir / 'pt_2D_2.txt')
    image1 = plt.imread(asset_dir / 'image1.jpg')
    image2 = plt.imread(asset_dir / 'image2.jpg')

    f = eight_point_algorithm(pts1, pts2, normalize=False)
    norm_f = eight_point_algorithm(pts1, pts2, normalize=True)
    cmp_fund_mat(image1, image2, pts1, pts2, f, norm_f)

    avg_dist = avg_epipolar_dist(pts1, pts2, f)
    avg_dist_norm = avg_epipolar_dist(pts1, pts2, norm_f)

    print('Average epipolar distance')
    print(f' - No normalization: {avg_dist:.4f}')
    print(f' - Normalization: {avg_dist_norm:.4f}')


if __name__ == "__main__":
    main()
