#!/usr/bin/python
import numpy as np


def generate_random_targets(n: int, z: int):
    """
    Generate a matrix of random target assignment.
    Each target assignment vector has unit length (hence can be view as random point on hypersphere)
    :param n: the number of samples to generate.
    :param z: the latent space dimensionality
    :return: the sampled representations
    """

    # Generate random targets using gaussian distrib.
    samples = np.random.normal(0, 1, (n, z)).astype(np.float32)
    # rescale such that fit on unit sphere.
    radiuses = np.expand_dims(np.sqrt(np.sum(np.square(samples), axis=1)), 1)
    # return rescaled targets
    return samples/radiuses
