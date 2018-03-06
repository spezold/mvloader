#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def rotation_matrix(angle_x, angle_y, angle_z, right_handed=True):

    c = np.cos(angle_x)
    s = np.sin(angle_x)
    rotation_x = np.array([[1, 0,  0],
                           [0, c, -s],
                           [0, s,  c]])

    c = np.cos(angle_y)
    s = -np.sin(angle_y) if right_handed else np.sin(angle_y)
    rotation_y = np.array([[c, 0, -s],
                           [0, 1,  0],
                           [s, 0,  c]])

    c = np.cos(angle_z)
    s = np.sin(angle_z)
    rotation_z = np.array([[c, -s, 0],
                           [s,  c, 0],
                           [0,  0, 1]])

    rotation = rotation_z @ rotation_y @ rotation_x

    return rotation


def transformation_matrix(angles=(0, 0, 0), scalings=(1, 1, 1), offsets=(0, 0, 0)):

    rotation = rotation_matrix(*angles) @ np.diag(scalings)

    transformation = np.eye(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = offsets

    return transformation


def random_transformation_matrix(verbose=False):

    angles = np.random.uniform(0, 2 * np.pi, size=3)
    scalings = np.random.exponential(size=3)
    offsets = np.random.normal(scale=10, size=3)

    transformation = transformation_matrix(angles=angles, scalings=scalings, offsets=offsets)
    if verbose:
        print("angles: {}\nscalings: {}\noffsets: {}".format(angles, scalings, offsets))

    return transformation


def random_voxel_data(size=(10, 10, 10)):

    voxel_data = np.random.normal(size=size)
    return voxel_data
