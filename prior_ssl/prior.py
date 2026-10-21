from pathlib import Path

import numpy as np
from PIL import Image


def load_binary_mask(mask_file, image_size):
    mask = Image.open(mask_file).convert("L")
    mask = mask.resize(tuple(image_size), Image.NEAREST)
    return (np.array(mask) > 0).astype(np.float32)


def thickness_sequence(mask):
    """Convert a binary mask to t(w)=sum_h M(h,w)."""

    if mask.ndim != 2:
        raise ValueError("thickness_sequence expects a 2D binary mask")
    return mask.astype(np.float32).sum(axis=0)


def _fit_pca(samples, n_components):
    mean = samples.mean(axis=0)
    centered = samples - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(int(n_components), vt.shape[0])
    components = vt[:n_components].T
    return mean, components


def _kmeans_pp(features, n_clusters, random_state=42, max_iter=100):
    rng = np.random.default_rng(random_state)
    n_samples = features.shape[0]
    n_clusters = min(max(1, int(n_clusters)), n_samples)
    centers = [features[rng.integers(n_samples)]]

    for _ in range(1, n_clusters):
        dist_sq = np.min(
            np.stack([np.sum((features - c) ** 2, axis=1) for c in centers], axis=1),
            axis=1,
        )
        total = dist_sq.sum()
        if total <= 0:
            centers.append(features[rng.integers(n_samples)])
        else:
            centers.append(features[rng.choice(n_samples, p=dist_sq / total)])

    centers = np.stack(centers, axis=0).astype(np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)
    for _ in range(max_iter):
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for idx in range(n_clusters):
            cluster = features[labels == idx]
            if len(cluster) > 0:
                centers[idx] = cluster.mean(axis=0)
    return centers, labels


class ThicknessMemoryBank:
    """PCA + KMeans++ memory bank for retinal membrane thickness priors."""

    def __init__(self, mean, components, centers, mu, sigma, tolerance_sigma=3.0):
        self.mean = mean.astype(np.float32)
        self.components = components.astype(np.float32)
        self.centers = centers.astype(np.float32)
        self.mu = mu.astype(np.float32)
        self.sigma = sigma.astype(np.float32)
        self.tolerance_sigma = float(tolerance_sigma)

    @classmethod
    def fit(
        cls,
        mask_files,
        image_size,
        n_components=2,
        n_clusters=3,
        random_state=42,
        kmeans_max_iter=100,
        tolerance_sigma=3.0,
    ):
        if not mask_files:
            raise ValueError("Cannot build thickness memory bank without labeled masks")

        sequences = [
            thickness_sequence(load_binary_mask(mask_file, image_size))
            for mask_file in mask_files
        ]
        return cls.fit_sequences(
            sequences,
            n_components=n_components,
            n_clusters=n_clusters,
            random_state=random_state,
            kmeans_max_iter=kmeans_max_iter,
            tolerance_sigma=tolerance_sigma,
        )

    @classmethod
    def fit_masks(
        cls,
        masks,
        n_components=2,
        n_clusters=3,
        random_state=42,
        kmeans_max_iter=100,
        tolerance_sigma=3.0,
    ):
        if not masks:
            raise ValueError("Cannot build thickness memory bank without labeled masks")
        sequences = [thickness_sequence(mask) for mask in masks]
        return cls.fit_sequences(
            sequences,
            n_components=n_components,
            n_clusters=n_clusters,
            random_state=random_state,
            kmeans_max_iter=kmeans_max_iter,
            tolerance_sigma=tolerance_sigma,
        )

    @classmethod
    def fit_sequences(
        cls,
        sequences,
        n_components=2,
        n_clusters=3,
        random_state=42,
        kmeans_max_iter=100,
        tolerance_sigma=3.0,
    ):
        samples = np.stack(sequences, axis=0).astype(np.float32)
        mean, components = _fit_pca(samples, n_components=n_components)
        features = (samples - mean) @ components
        centers, labels = _kmeans_pp(
            features,
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=kmeans_max_iter,
        )

        mu = []
        sigma = []
        for idx in range(centers.shape[0]):
            cluster = features[labels == idx]
            if len(cluster) == 0:
                distances = np.array([0.0], dtype=np.float32)
            else:
                distances = np.linalg.norm(cluster - centers[idx], axis=1)
            mu.append(float(distances.mean()))
            sigma.append(float(distances.std()))

        return cls(mean, components, centers, np.array(mu), np.array(sigma), tolerance_sigma)

    def transform_mask(self, mask):
        sequence = thickness_sequence(mask)
        return (sequence - self.mean) @ self.components

    def score_mask(self, mask, eps=1e-6):
        feature = self.transform_mask(mask)
        distances = np.linalg.norm(self.centers - feature[None, :], axis=1)
        cluster = int(np.argmin(distances))
        distance = float(distances[cluster])
        tolerance = float(self.mu[cluster] + self.tolerance_sigma * self.sigma[cluster])
        tolerance = max(tolerance, eps)
        if distance > tolerance:
            score = 0.0
        else:
            score = float(np.exp(-distance / tolerance))
        return {
            "score": score,
            "distance": distance,
            "cluster": cluster,
            "tolerance": tolerance,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            components=self.components,
            centers=self.centers,
            mu=self.mu,
            sigma=self.sigma,
            tolerance_sigma=np.array([self.tolerance_sigma], dtype=np.float32),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(
            data["mean"],
            data["components"],
            data["centers"],
            data["mu"],
            data["sigma"],
            tolerance_sigma=float(data["tolerance_sigma"][0])
            if "tolerance_sigma" in data
            else 3.0,
        )
