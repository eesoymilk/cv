from pathlib import Path

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as scipy_spatial

from tqdm import tqdm, trange
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy.spatial import distance as scipy_distance

script_dir = Path(__file__).parent.absolute()
output_dir = script_dir / 'output'
image_name = '2-masterpiece'
ext = 'png'

# create output directory if not exists
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)


def showImageCV(
    img: MatLike,
    fname: str | None = None,
    show_plot: bool = True,
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

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)

    if show_plot:
        plt.show()

    plt.close(fig)


def kmeans_clustering(
    img: MatLike,
    k: int,
    means: NDArray[np.int8],
    threshold: float = 0.1,
    n_iterations: int = 100,
):
    height, width, channels = img.shape
    img = img.reshape((-1, channels))
    prev_means = np.copy(means)

    # Iterate until convergence or max iterations
    for i in trange(n_iterations):
        # Compute pairwise distances and assign clusters
        distances = scipy_distance.cdist(img, means, 'euclidean')
        cluster_assignments = np.argmin(distances, axis=1)

        # Update means
        for j in range(k):
            if np.sum(cluster_assignments == j) == 0:
                # If there are no pixels in the cluster, reinitialize centroid
                means[j] = np.random.randint(0, 256, size=channels)
                continue
            means[j] = np.mean(img[cluster_assignments == j], axis=0)

        # Check convergence
        delta = np.linalg.norm(means - prev_means)
        if delta < threshold:
            break

        # Update previous means
        prev_means = np.copy(means)

    # Assign pixels to their cluster means
    img_clustered = np.zeros(img.shape)
    for i in range(k):
        img_clustered[cluster_assignments == i] = means[i]

    # Reshape image and return
    return img_clustered.reshape((height, width, channels)).astype(np.uint8)


def initialize_means(
    img: MatLike,
    k: int,
    n_initial_guesses: int = 50,
) -> NDArray[np.int8]:
    _, _, channels = img.shape
    img = img.reshape((-1, channels))

    # Randomly initialize n_initial_guesses means
    initial_means = np.random.randint(
        0, 256, size=(n_initial_guesses, channels)
    )

    # Compute pairwise distances between each pixel and guesses
    distances = scipy_distance.cdist(img, initial_means, 'euclidean')

    # Find the n_initial_guesses means with the k smallest average distances
    avg_distances = np.mean(distances, axis=0)
    return initial_means[np.argsort(avg_distances)[:k]]


def kmeans(
    img: MatLike,
    k: int,
    threshold: float = 0.1,
    n_iterations: int = 100,
):
    means = initialize_means(img, k, n_initial_guesses=50)
    return kmeans_clustering(img, k, means, threshold, n_iterations)


def initialize_means_pp(img: MatLike, k: int):
    _, _, channels = img.shape
    img = img.reshape((-1, channels))

    means = np.zeros((k, 3))
    means[0] = np.random.randint(0, 256, size=3)

    for i in range(1, k):
        distances = scipy_distance.cdist(img, means[:i], 'euclidean')
        min_distances = np.min(distances, axis=1)

        # Compute the probability of each pixel being chosen as a mean
        probs = min_distances / np.sum(min_distances)

        # Choose a pixel as a mean based on the probability distribution
        means[i] = img[np.random.choice(len(img), p=probs)]

    return means


def kmeans_pp(
    img: MatLike,
    k: int,
    threshold: float = 0.1,
    n_iterations: int = 100,
):
    means = initialize_means_pp(img, k)
    return kmeans_clustering(img, k, means, threshold, n_iterations)


def plot_rgb_space(
    img: NDArray, fname: Path | str | None = None, show_plot: bool = True
):
    _, _, channels = img.shape
    img_norm = img.astype(np.float32) / 255
    img_flat = img_norm.reshape((-1, channels))
    img_flat_rgb = img_flat[..., ::-1]

    # Create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(
        img_flat_rgb[:, 0],
        img_flat_rgb[:, 1],
        img_flat_rgb[:, 2],
        color=[tuple(color) for color in img_flat_rgb[:, :3]],
    )

    # Set labels
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)

    if show_plot:
        plt.show()

    plt.close(fig)


def group_pixels(
    img_flat: MatLike,
    width: int,
    height: int,
    size: int,
    spatial: bool = False,
):
    color = np.linspace(0, 255, size, dtype=np.float32) / 255

    if spatial:
        # Add spatial coordinates with size
        x_line = np.linspace(0, width, size, dtype=np.float32) / width
        y_line = np.linspace(0, height, size, dtype=np.float32) / height
        bb, gg, rr, xx, yy = np.meshgrid(color, color, color, x_line, y_line)
        pts = np.stack((bb, gg, rr, xx, yy), axis=-1).astype(np.float32)
        pts = pts.reshape(-1, 5)
    else:
        bb, gg, rr = np.meshgrid(color, color, color)
        pts = np.stack((bb, gg, rr), axis=-1).astype(np.float32)
        pts = pts.reshape(-1, 3)

    kdtree = scipy_spatial.KDTree(pts)
    _, cloest_groups = kdtree.query(img_flat)

    return cloest_groups, pts


def meanshift(
    img: MatLike,
    grouping_size: int = 5,
    band_width: int = 5,
    threshold: float = 0.1,
    n_iterations: int = 50,
    spatial: bool = False,
    verbose: bool = False,
):
    height, width, channels = img.shape
    img_norm = img.astype(np.float32) / 255
    img_flat = img_norm.reshape((-1, channels))
    threshold_squared = threshold**2

    if spatial:
        x = (np.linspace(0, width, width, dtype=np.float32) / width,)
        y = (np.linspace(0, height, height, dtype=np.float32) / height,)
        xx, yy = np.meshgrid(x, y)
        img_flat = np.concatenate((img_flat, xx.reshape(-1, 1)), axis=1)
        img_flat = np.concatenate((img_flat, yy.reshape(-1, 1)), axis=1)

    img_tree = scipy_spatial.KDTree(img_flat)
    grouped_indices, means = group_pixels(
        img_flat, width, height, grouping_size, spatial=spatial
    )

    # initialize means
    means_stable = np.zeros(means.shape[0], dtype=bool)

    # Iterate over the number of iterations
    stability_progress = tqdm(total=means.shape[0], desc='Stability')
    for iteration in trange(n_iterations, desc='Iteration'):
        unstable_indices = np.where(~means_stable)[0]
        means_tree = scipy_spatial.KDTree(means[unstable_indices])

        neigbors_list = means_tree.query_ball_tree(
            img_tree, band_width, p=2, eps=0
        )

        updated_means = np.copy(means[unstable_indices])
        n_neighborless = 0
        for i, neighbors in enumerate(neigbors_list):
            if len(neighbors) == 0:
                n_neighborless += 1
                continue
            updated_means[i] = np.mean(img_flat[neighbors], axis=0)

        if n_neighborless > 0 and verbose:
            print(f'#{iteration}: {n_neighborless} pixels have no neighbors')

        # Check convergence using distance between means
        delta_squared = np.sum(
            (updated_means - means[unstable_indices]) ** 2, axis=1
        )
        means_stable[unstable_indices] = delta_squared < threshold_squared

        # update means
        means[unstable_indices] = updated_means

        # Check convergence
        stability_progress.update(np.sum(means_stable) - stability_progress.n)
        if np.all(means_stable):
            break

    return (
        (means[grouped_indices, :3] * 255)
        .reshape((height, width, channels))
        .astype(np.uint8)
    )


def main():
    show_plot = True

    image = cv.imread(f'{image_name}.{ext}')
    showImageCV(image, show_plot=show_plot)

    # perform k-means and k-means++ clustering
    ks = [4, 8, 16]
    for k in ks:
        print(f'{k}-means Clustering')
        img_clustered = kmeans(image, k)
        showImageCV(
            img_clustered,
            fname=output_dir / f'{image_name}-k{k}.png',
            show_plot=show_plot,
        )
        print(f'{k}-means++ Clustering')
        img_clustered_pp = kmeans_pp(image, k, n_iterations=200)
        showImageCV(
            img_clustered_pp,
            fname=output_dir / f'{image_name}-kpp{k}.png',
            show_plot=show_plot,
        )

    # perform mean-shift clustering
    print('Mean-shift Clustering')
    band_widths = [0.1, 0.2, 0.4]
    plot_rgb_space(
        image, fname=output_dir / f'{image_name}-rgb.png', show_plot=show_plot
    )
    for band_width in band_widths:
        fname = f'{image_name}-ms{f"{band_width}".split(".")[1]}'

        print(f'band_width={band_width}')
        img_meanshift = meanshift(
            image, grouping_size=4, band_width=band_width, threshold=0.01
        )
        plot_rgb_space(
            img_meanshift,
            fname=output_dir / f'{fname}-rgb.png',
            show_plot=show_plot,
        )
        showImageCV(
            img_meanshift,
            fname=output_dir / f'{fname}.png',
            show_plot=show_plot,
        )

    # perform mean-shift clustering with spatial coordinates
    print('Mean-shift Clustering with Spatial Coordinates')
    img_meanshift_spatial = meanshift(
        image, grouping_size=5, band_width=0.3, threshold=0.01, spatial=True
    )
    showImageCV(
        img_meanshift_spatial,
        fname=output_dir / f'{image_name}-ms-spatial.png',
        show_plot=show_plot,
    )


if __name__ == '__main__':
    main()
