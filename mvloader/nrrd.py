#!/usr/bin/env python
# coding: utf-8

"""A module for reading NRRD files [1]_, basically a wrapper for calls on the pynrrd library [2]_.

References
----------
.. [1] http://teem.sourceforge.net/nrrd/format.html (20180212)
.. [2] https://github.com/mhe/pynrrd (20180212).
"""

import nrrd
import numpy as np

from mvloader.volume import Volume


def open_image(path, verbose=True):
    """
    Open a 3D NRRD image at the given path.

    Parameters
    ----------
    path : str
        The path of the file to be loaded.
    verbose : bool, optional
        If `True` (default), print some meta data of the loaded file to standard output.

    Returns
    -------
    Volume
        The resulting 3D image volume, with the ``src_object`` attribute set to the tuple `(data, header)` returned
        by pynrrd's ``nrrd.read`` (where `data` is a Numpy array and `header` is a dictionary) and the desired
        anatomical world coordinate system ``system`` set to "RAS".

    Raises
    ------
    IOError
        If something goes wrong.
    """
    try:
        src_object = (voxel_data, hdr) = nrrd.read(path)
    except Exception as e:
        raise IOError(e)

    if verbose:
        print("Image loaded:", path)
        print("Meta data:")
        for k in sorted(hdr.keys(), key=str.lower):
            print("{}: {!r}".format(k, hdr[k]))

    __check_data_kinds_in(hdr)
    src_system = __world_coordinate_system_from(hdr)  # No fixed world coordinates for NRRD images!
    mat = __matrix_from(hdr)  # Voxels to world coordinates
        
    # Create new ``Volume`` instance
    volume = Volume(src_voxel_data=voxel_data, src_transformation=mat, src_system=src_system, system="RAS",
                    src_object=src_object)
    return volume


def save_image(path, data, transformation):
    """
    Save the given image data as a NRRD image file at the given path.

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
    # Create the header entries from the transformation
    space = "RAS"
    space_directions = transformation[:3, :3].T.tolist()
    space_origin = transformation[:3, 3].tolist()
    options = {"space": space, "space directions": space_directions, "space origin": space_origin}
    nrrd.write(filename=path, data=data, options=options)


def save_volume(path, volume):
    """
    Save the given ``Volume`` instance as a NRRD image file at the given path.

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


def __check_data_kinds_in(header):
    """
    Sanity check on the header's "kinds" field: are all entries either "domain" or "space" (i.e. are we really dealing
    with scalar data on a spatial domain)?

    Parameters
    ----------
    header : dict
        A dictionary containing the NRRD header (as returned by ``nrrd.read``, for example).

    Returns
    -------
    None
        Simply return if everything is ok or the "kinds" field is not set.

    Raises
    ------
    IOError
        If the "kinds" filed contains entries other than "domain" or "space".
    """
    kinds = header.get("kinds")
    if kinds is None:
        return

    for k in kinds:
        if k.lower() not in ["domain", "space"]:
            raise IOError("At least one data dimension contains non-spatial data!")


def __world_coordinate_system_from(header):
    """
    From the given NRRD header, determine the respective assumed anatomical world coordinate system.

    Parameters
    ----------
    header : dict
        A dictionary containing the NRRD header (as returned by ``nrrd.read``, for example).

    Returns
    -------
    str
        The three-character uppercase string determining the respective anatomical world coordinate system (such as
        "RAS" or "LPS").

    Raises
    ------
    IOError
        If the header is missing the "space" field or the "space" field's value does not determine an anatomical world
        coordinate system.
    """
    try:
        system_str = header["space"]
    except KeyError as e:
        raise IOError("Need the header's \"space\" field to determine the image's anatomical coordinate system.")

    if len(system_str) == 3:
        # We are lucky: this is already the format that we need
        return system_str.upper()

    # We need to separate the string (such as "right-anterior-superior") at its dashes, then get the first character
    # of each component. We cannot handle 4D data nor data with scanner-based coordinates ("scanner-...") or
    # non-anatomical coordinates ("3D-...")
    system_components = system_str.split("-")
    if len(system_components) == 3 and not system_components[0].lower() in ["scanner", "3d"]:
        system_str = "".join(c[0].upper() for c in system_components)
        return system_str

    raise IOError("Cannot handle \"space\" value {}".format(system_str))


def __matrix_from(header):
    """
    Calculate the transformation matrix from voxel coordinates to the header's anatomical world coordinate system.

    Parameters
    ----------
    header : dict
        A dictionary containing the NRRD header (as returned by ``nrrd.read``, for example).

    Returns
    -------
    ndarray
        The resulting :math:`4x4` transformation matrix.
    """
    try:
        space_directions = header["space directions"]
        space_origin = header["space origin"]
    except KeyError as e:
        raise IOError("Need the header's \"{}\" field to determine the mapping from voxels to world coordinates.".format(e))

    # "... the space directions field gives, one column at a time, the mapping from image space to world space
    # coordinates ... [1]_" -> list of columns, needs to be transposed
    trans_3x3 = np.array(space_directions).T
    trans_4x4 = np.eye(4)
    trans_4x4[:3, :3] = trans_3x3
    trans_4x4[:3, 3] = space_origin
    return trans_4x4
