#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel
import nrrd
import numpy as np
from numpy import ma
from pathlib import Path
import tempfile


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


def generate_test_data():
    """
    Returns
    -------
    str
        Path to the temporary directory containing the test data. To be cleaned up after use!
    """
    testdata_dir = Path(tempfile.mkdtemp(prefix="mvloader-")).resolve()
    testdata = np.arange(8, dtype=np.uint16).reshape(2,2,2)
    src_testdata = {}
    src_transformations = {}
    tau = 2 * np.pi

    def transform(array, transformation):

        # If `transformation` is a 3x3 matrix, also calculate the respective 4x4 transformation matrix, with
        # the offset calculated to have the origin at the testdata voxel with value zero

        def offset(perm, shape):

            # 4x4 matrix to be premultiplied

            must_be_flipped = lambda p: (np.sum(p, axis=0) < 0).astype(int)

            ndim = len(shape)
            max_indices = np.asarray(shape) - 1
            # Swap if the row's set element's sign is negative -> add offset there
            offset_vector = must_be_flipped(perm[:ndim, :ndim]) * (-max_indices)
            offset_matrix = np.eye(ndim + 1, dtype=np.int)
            offset_matrix[:-1, -1] = offset_vector
            return offset_matrix

        calculate_4x4 = (transformation.shape == (3, 3))
        transformation_3x3 = transformation[:3, :3]
        transformed_array = array

        abs_transformation_3x3 = ma.masked_array(np.abs(transformation_3x3), mask=np.zeros_like(transformation_3x3, dtype=np.bool))
        perm = np.zeros(abs_transformation_3x3.shape, dtype=np.int)
        ji_argmaxes = []
        while np.sum(~abs_transformation_3x3.mask) > 0:
            ij_argmax = np.unravel_index(abs_transformation_3x3.argmax(), abs_transformation_3x3.shape)
            perm[ij_argmax] = np.sign(transformation_3x3[ij_argmax])
            abs_transformation_3x3.mask[ij_argmax[0], :] = True
            abs_transformation_3x3.mask[:, ij_argmax[1]] = True
            ji_argmaxes.append(ij_argmax[::-1])
        ji_argmaxes = np.asarray(sorted(ji_argmaxes))

        transformed_array = np.transpose(transformed_array, axes=ji_argmaxes[:, 1])
        for j in ji_argmaxes[:, 0]:
            if transformation[ji_argmaxes[j, 1], j] < 0:
                transformed_array = np.flip(transformed_array, axis=j)

        if calculate_4x4:
            transformation_3x3 = transformation
            transformation = np.eye(4)
            transformation[:3, :3] = transformation_3x3
            transformation = transformation @ offset(perm, array.shape)
        return transformed_array, transformation

    def to_nifti(transformation, filepath):

        transformation = np.asarray(transformation)
        transformed_testdata, transformation = transform(testdata, transformation)
        key = str(filepath.name).split("-")[0][-3:]
        src_testdata[key] = transformed_testdata
        src_transformations[key] = transformation
        nibabel.Nifti1Image(transformed_testdata, transformation).to_filename(str(filepath))

    def to_nrrd(transformation, filepath, src_system):

        transformation = np.asarray(transformation)
        transformed_testdata, transformation = transform(testdata, transformation)
        key = str(filepath.name).split("2")[0][-3:]
        src_testdata[key] = transformed_testdata
        src_transformations[key] = transformation
        space_directions = transformation[:3, :3].T.tolist()
        space_origin = transformation[:3, 3].tolist()
        options = {"space": src_system, "space directions": space_directions, "space origin": space_origin}
        nrrd.write(filename=str(filepath), data=transformed_testdata, options=options)

    # NIfTI: We just generate a few of the 48 possible coordinate systems (3 * 2 * 1 axis permutations times 2 ** 3 axis
    # flips) for testing

    # Source data aligned with RAS anatomical coordinates
    file = testdata_dir / "RAS-1.0x1.0x1.0.nii.gz"
    t = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    to_nifti(t, file)
    # Source data aligned with LAS anatomical coordinates
    file = testdata_dir / "LAS-0.3x0.4x0.5.nii.gz"
    t = [[-0.3, 0.0, 0.0, 0.3],
         [ 0.0, 0.4, 0.0, 0.0],
         [ 0.0, 0.0, 0.5, 0.0],
         [ 0.0, 0.0, 0.0, 1.0]]
    to_nifti(t, file)
    # Source data aligned with LPS anatomical coordinates
    file = testdata_dir / "LPS-3.0x3.1x3.2.nii.gz"
    t = [[-3.0,  0.0, 0.0, 3.0],
         [ 0.0, -3.1, 0.0, 3.1],
         [ 0.0,  0.0, 3.2, 0.0],
         [ 0.0,  0.0, 0.0, 1.0]]
    to_nifti(t, file)
    # Source data aligned with RSA anatomical coordinates
    file = testdata_dir / "RSA-1.0x0.9x1.1.nii.gz"
    t = [[1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.1, 0.0],
         [0.0, 0.9, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    to_nifti(t, file)
    # Source data aligned with SLP anatomical coordinates
    file = testdata_dir / "SLP-0.9x0.8x0.5.nii.gz"
    t = [[0.0, -0.8,  0.0, 0.8],
         [0.0,  0.0, -0.5, 0.5],
         [0.9,  0.0,  0.0, 0.0],
         [0.0,  0.0,  0.0, 1.0]]
    to_nifti(t, file)
    # Source data aligned with IPL anatomical coordinates
    file = testdata_dir / "IPL-3.0x9.0x0.1.nii.gz"
    t = [[ 0.0,  0.0, -0.1, 0.1],
         [ 0.0, -9.0,  0.0, 9.0],
         [-3.0,  0.0,  0.0, 3.0],
         [ 0.0,  0.0,  0.0, 1.0]]
    to_nifti(t, file)
    # Source data *almost* aligned with RAI anatomical coordinates
    file = testdata_dir / "aRAI-1.1x1.2x0.9.nii.gz"
    f = np.diag([1, 1, -1])
    r = rotation_matrix(0.01 * tau, 0.02 * tau, 0.03 * tau)
    fr = f @ r
    s = np.diag([1.1, 1.2, 0.9])
    frs = fr @ s
    to_nifti(frs, file)

    # NRRD: Same here -- just some combinations, but use all three of NRRD's supported anatomical coordinate systems

    # Source data aligned with IRA anatomical coordinates, NRRD system: RAS
    file = testdata_dir / "IRA2RAS-0.1x0.2x0.4.nrrd"
    t = [[ 0.0, 0.2, 0.0, 0.0],
         [ 0.0, 0.0, 0.4, 0.0],
         [-0.1, 0.0, 0.0, 0.1],
         [ 0.0, 0.0, 0.0, 1.0]]
    to_nrrd(t, file, "RAS")
    # Source data aligned with IPR anatomical coordinates, NRRD system: LAS
    file = testdata_dir / "IPR2LAS-2.5x5.2x1.0.nrrd"
    t = [[ 0.0,  0.0, -1.0, 1.0],
         [ 0.0, -5.2,  0.0, 5.2],
         [-2.5,  0.0,  0.0, 2.5],
         [ 0.0,  0.0,  0.0, 1.0]]
    to_nrrd(t, file, "LAS")
    # Source data aligned with AIR anatomical coordinates, NRRD system: LPS
    file = testdata_dir / "AIR2LPS-1.0x1.3x3.7.nrrd"
    t = [[ 0.0,  0.0, -3.7, 3.7],
         [-1.0,  0.0,  0.0, 1.0],
         [ 0.0, -1.3,  0.0, 1.3],
         [ 0.0,  0.0,  0.0, 1.0]]
    to_nrrd(t, file, "LPS")
    # Source data *almost* aligned with LAI anatomical coordinates, NRRD system: LAS
    file = testdata_dir / "aLAI2LAS-2.1x1.3x0.8.nrrd"
    f = np.diag([1, 1, -1])
    r = rotation_matrix(0.13 * tau, 0.10 * tau, 0.07 * tau)
    fr = f @ r
    s = np.diag([2.1, 1.3, 0.8])
    frs = fr @ s
    to_nrrd(frs, file, "LAS")

    return str(testdata_dir), testdata, src_testdata, src_transformations

