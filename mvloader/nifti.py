#!/usr/bin/env python
# coding: utf-8

"""
A module for reading and writing NIfTI-1 files [1]_, basically a wrapper for calls
on the Nibabel library [2]_.

References
----------
.. [1] http://niftilib.sourceforge.net/c_api_html/nifti1_8h-source.html (20180212)
.. [2] http://nipy.org/nibabel/ (20180212).
"""

import gzip
import nibabel
import numpy as np
from pathlib import Path

from mvloader.volume import Volume


def open_image(path, verbose=True, repair=False):
    """
    Open a 3D NIfTI-1 image at the given path.

    Parameters
    ----------
    path : str
        The path of the file to be loaded.
    verbose : bool, optional
        If `True` (default), print some meta data of the loaded file to standard output.
    repair : bool, optional
        If `True`, remove trailing 4th dimension of the image volume if it contains a single entry only (default is
        `False`). Note that in this case it has not been tested whether the coordinate transformations from the NIfTI-1
        header still apply.

    Returns
    -------
    Volume
        The resulting 3D image volume, with the ``src_object`` attribute set to the respective
        ``nibabel.nifti1.Nifti1Image`` instance and the desired anatomical world coordinate system ``system`` set to
        "RAS".

    Raises
    ------
    IOError
        If something goes wrong.
    """
    # According to the NIfTI-1 specification [1]_, the world coordinate system
    # of NIfTI-1 files is always RAS.
    src_system = "RAS"
    
    try:
        src_object = nibabel.nifti1.load(path)
    except Exception as e:
        raise IOError(e)

    voxel_data = np.array(src_object.get_data())
    hdr = src_object.header

    ndim = hdr["dim"][0]
    if ndim != 3:
        raise IOError("Currently only 3D images can be handled. The given image has {} dimension(s).".format(ndim))
    
    if verbose:
        print("Image loaded:", path)
        print("Meta data:")
        print(hdr)
        print("Image dimensions:", voxel_data.ndim)

    # Repair superfluous 4th dimension
    if repair:
        voxel_data = __repair_dim(voxel_data, verbose)
        
    # Create new ``Volume`` instance
    
    # Get quaternion + voxel size + offset information, convert to transformation matrix
    quaternion = hdr.get_qform_quaternion()
    voxel_size = np.asarray(hdr.get_zooms()[:3])  # Discard the last value for 4D data
    offset = np.array([hdr["qoffset_x"], hdr["qoffset_y"], hdr["qoffset_z"]])
    # Adjust the voxel size according to the "qfac" stored in pixdim[0] (cf. [1]_)
    qfac = hdr["pixdim"][0]
    qfac = qfac if qfac in [-1, 1] else 1
    voxel_size = voxel_size * (1, 1, qfac)
    mat = __matrix_from_quaternion(quaternion, voxel_size, offset)

    volume = Volume(src_voxel_data=voxel_data, src_transformation=mat, src_system=src_system, system="RAS",
                    src_object=src_object)
    return volume


def save_image(path, data, transformation):
    """
    Save the given image data as a NIfTI image file at the given path.

    Parameters
    ----------
    path : str
        The path for the file to be saved.
    data : array_like
        Three-dimensional array that contains the voxels to be saved.
    transformation : array_like
        :math:`4x4` transformation matrix that maps from ``data``'s voxel indices to a RAS anatomical world coordinate
        system.
    """
    nibabel.Nifti1Image(data, transformation).to_filename(path)


def save_volume(path, volume):
    """
    Save the given ``Volume`` instance as a NIfTI image file at the given path.

    Parameters
    ----------
    path : str
        The path for the file to be saved.
    volume : Volume
        The ``Volume`` instance containing the image data to be saved.
    """
    volume = volume.copy()
    volume.system = "RAS"
    save_image(path, data=volume.aligned_volume, transformation=volume.aligned_transformation)


def __repair_dim(data, verbose):
    """
    For 4d arrays with the last dimension containing only one element, return a new 3d array of the same content. For
    other arrays, simply return them.

    Parameters
    ----------
    data : array_like
        The array to be repaired
    verbose : bool
        If `True`, print a message in case the dimensions have changed.

    Return
    ------
    array_like
        The result from correction.
    """
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0].copy()
        if verbose:
            print("4D array has been corrected to 3D.")
    return data


def __matrix_from_quaternion(quaternion, voxel_size, offset):
    """
    Calculate the transformation matrix from voxel coordinates to the anatomical world coordinate system based on the
    given quaternion, voxel size, and offset information.

    Parameters
    ----------
    quaternion : array_like
        Quaternion that gives the rotation information (4-element array with the quaternion's scalar/real part as first
        element).
    voxel_size : array_like
        Voxel sizes in world coordinate system units per voxel (3-element array).
    offset : array_like
        Offset information (i.e. translational part of the transformation matrix; 3-element array).

    Returns
    -------
    ndarray
        The resulting :math:`4x4` transformation matrix.
    """
    trans_3x3 = nibabel.quaternions.quat2mat(quaternion)
    trans_3x3 = trans_3x3 @ np.diag(voxel_size)
    trans_4x4 = np.eye(4)
    trans_4x4[:3, :3] = trans_3x3
    trans_4x4[:3, 3] = offset
    return trans_4x4


def compress(path, delete_originals=False):
    """
    Compress the NIfTI file(s) at the given path to `.nii.gz` files. Save the result(s) with the same name(s) in the
    same folder, but with the file extension changed from `.nii` to `.nii.gz`.
    
    Parameters
    ----------
    path : str
        If a directory path is given, compress all contained .nii files (just in the folder, not in its subfolders). If
        a path to a `.nii` file is given, compress the file.
    delete_originals : bool, option
        If `True`, try to delete the original `.nii` file(s) after compressing (default is `False`).
    """
    path = Path(path).resolve()
    if path.is_dir():
        file_paths = sorted((f.resolve() for f in path.iterdir() if str(f).lower().endswith(".nii")),
                            key=lambda p: str(p).lower())
    else:
        file_paths = [path]

    for in_path in file_paths:
        in_path = str(in_path)
        try:
            print("Compressing {} ...".format(in_path))
            out_path = in_path + ".gz"
            with open(in_path, "rb") as in_file, gzip.open(out_path, "wb") as out_file:
                out_file.writelines(in_file)
            compress_success = True
        except Exception as e:
            print("Compressing {} failed! ({})".format(in_path, e))
            compress_success = False
        if compress_success and delete_originals:
            try:
                print("Deleting {} ...".format(in_path))
                Path(in_path).unlink()
            except Exception as e:
                print("Deleting {} failed! ({})".format(in_path, e))
