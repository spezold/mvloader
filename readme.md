MVloader
========

*MVloader* is meant to be a tiny helper to load and save *medical
volumetric* data (therefore *MV*) or *image volumes*, i.e.
three-dimensional medical images (DICOM, NIfTI, or NRRD). It is also
meant to simplify dealing with their different anatomical world
coordinate systems.


Installing MVloader
-------------------

*MVloader* can be installed using *pip*:
```shell
pip install git+https://github.com/spezold/mvloader.git
```

This should also work inside *conda*.

*MVloader*'s only dependencies are the underlying image libraries
([*pydicom*](https://github.com/pydicom/pydicom),
[*nibabel*](http://nipy.org/nibabel/),
[*pynrrd*](https://github.com/mhe/pynrrd)) and
[*NumPy*](http://www.numpy.org/), which should all be resolved
automatically during installation.


The `Volume` Class
------------------

All loaded image volumes are represented as instances of the `Volume`
class.

`Volume` serves as an abstraction layer to handle transformations from
voxel indices to arbitrary anatomical world coordinate systems. In
particular

1. the user may choose each `Volume`'s anatomical world coordinate
   system independent of the underlying file format's world coordinate
   system, and
2. ``Volume`` provides a voxel representation (called `aligned_volume`)
   with the voxel axes aligned as closely as possible to the user's
   choice of anatomical world coordinate system. This representation
   simplifies visualizing volumes in *almost* correct anatomical
   orientation and processing data differently and consistently along
   different body axes. The `aligned_volume` representation is therefore
   suitable for cases in which one is not interested in a *precise*
   alignment, which, in turn, would generally require an expensive
   interpolation of the voxel data.

### Voxel Data Representations

Each `Volume` instance provides two representations of the image
volume's voxel data as three-dimensional NumPy arrays:

*  `src_volume` provides the voxels in the same order as they have been
   returned by the underlying image library.
*  `aligned_volume` provides the voxels with their axes aligned as
   closely as possible to the anatomical world coordinate system that
   has been chosen by the user. For example, if the user chooses an
   "LPS" anatomical world coordinate system (meaning that the first
   coordinate axis will point to the patient's left side, the second
   axis will point to their back, and the third will point towards their
   head), then `aligned_volume[1, 0, 0]` will lie to the left of
   `aligned_volume[0, 0, 0]`, `aligned_volume[0, 1, 0]` will lie closer
   to the patient's back, and `aligned_volume[0, 0, 1]` will lie closer
   to their head.

### Transformation Matrix Representations

Each `Volume` instance provides three 4x4 transformation matrices to map
from voxel indices to anatomical world coordinates:

*  `src_transformation` maps from `src_volume`'s voxel indices to the
   world coordinate system that has been assumed by the underlying image
   format (which is provided via `Volume`'s `src_system` property).
*  `aligned_transformation` maps from `aligned_volume`'s voxel indices
   to the world coordinate system that has been chosen by the user
   (which is provided via `Volume`'s `system` property).
*  `src_to_aligned_transformation` maps from `src_volume`'s voxel
   indices to the world coordinate system that has been chosen by the
   user (namely `system`).

Apart from that, the mappings from `src_volume` and `aligned_volume` to
arbitrary anatomical world coordinate systems can be determined via
`Volume`'s methods `get_src_transformation()` and
`get_aligned_transformation()`.

### Choosing a World Coordinate System

By default, all `Volume` instances are created so that the user-chosen
anatomical world coordinate system is "RAS". This may be adjusted via
the `Volume`'s `system` property. All common choices, such as "RAS",
"LAS", and "LPS", but also more "exotic" ones like
```python
volume.system = "IAR"  # 1st axis: inferior, 2nd: anterior, 3rd: right
```
are possible here. Technically, all permutations of {A,P}, {I,S}, {L,R}
may be provided to `system` as a (case-insensitive) three-character
string. Any update of `system` will update the `aligned_volume` voxel
data, the respective voxel size information, and the transformation
matrices accordingly.

### An Example

We create a new `Volume` instance:
```python
import numpy as np
from mvloader.volume import Volume

# A simple 10x10x10 volume
given_voxels = np.arange(1000).reshape(10, 10, 10)
# No rotations or translations from the provided voxel indices to the
# provided anatomical world coordinate system ...
given_voxels2given_world = np.eye(4)
# ... which we assume is DICOM-style ("LPS")
given_world = "LPS"
# However, as we prefer to work with NIfTI-style world coordinates, our
# user choice is "RAS"
our_world = "RAS"

volume = Volume(src_voxel_data=given_voxels,
                src_transformation=given_voxels2given_world,
                src_system=given_world,
                system=our_world)
```
In a real application, both `src_transformation` and `src_system` are
usually not our choice, but the result of loading a particular image of
a particular format (see *Loading Images* below). In our example, by
setting `src_transformation` to an identity matrix, we know that the
`src_system`'s world coordinate origin must lie at voxel index
`[0, 0, 0]`, which we can easily check:
```python
voxels2world = volume.src_transformation
world2voxels = np.linalg.inv(voxels2world)
world_origin = [0, 0, 0, 1]
voxel_index_of_world_origin = world2voxels[:3] @ world_origin
print(voxel_index_of_world_origin)
# [0. 0. 0.]
```

Note that we had to use homogeneous coordinates for the transformation,
which explains the trailing 1 for the world origin's coordinate;
however, by using only the first three rows of the transformation matrix
`world2voxels`, our resulting voxel index contains only three values, as
one could expect.

As we chose the voxel data as `np.arange(1000).reshape(10, 10, 10)`, the
value at voxel index `[0, 0, 0]` is zero. As voxel index `[0, 0, 0]`
coincides with the world coordinate system's origin (see above), this
means that the value at the world coordinate sytem's origin is zero:
```python
voxel_index_of_world_origin = tuple(voxel_index_of_world_origin.astype(np.int))
print(volume.src_volume[voxel_index_of_world_origin])
# 0
```

Now, as we seem to prefer working with RAS rather than LPS coordinates
(remember that we *chose* `system=our_world` with `our_world="RAS"`
above), things are different with `aligned_volume`, the voxel data
representation whose axes are more or less aligned with our chosen world
coordinate system's axes: the world coordinate system's origin does
*not* lie at `aligned_volume`'s voxel index `[0, 0, 0]`. Indeed, as
`aligned_volume`'s voxel indices along axis 0 should increase when
moving to the right of the patient (rather than to their left like in
`src_system`, which is "LPS"), and its indices along axis 1 should
increase when moving to the patient's front (rather than their back),
the origin must now lie at the greatest voxel index along these two
axes, which is index 9 (recall that the voxel data shape is
`(10, 10, 10)`):
```python
voxels2world = volume.aligned_transformation
world2voxels = np.linalg.inv(voxels2world)
voxel_index_of_world_origin = world2voxels[:3] @ world_origin
print(voxel_index_of_world_origin)
# [9. 9. 0.]
```
We can easily check that the voxel data at the world coordinate origin
remains the same in `aligned_volume` as in `src_volume`:
```python
print(volume.src_volume[0, 0, 0] == volume.aligned_volume[9, 9, 0])
# True
```
This index shift is reflected in the translational part of
`aligned_transformation` -- in order to get from `aligned_volume`'s
voxel indices to "RAS" world coordinates, we must subtract 9 in
coordinate axis 0 and 1:
```python
print(volume.aligned_transformation)
# [[ 1.  0.  0. -9.]
#  [ 0.  1.  0. -9.]
#  [ 0.  0.  1.  0.]
#  [ 0.  0.  0.  1.]]
```
Thus, in case an axis direction is swapped (e.g. from "L" to "R") the
world coordinate system's origin will remain in the same voxel position.
However, as the voxel position will also change (with the respective
voxel axis being reversed), the offset part of the resulting
transformation matrix may look hugely different.

As `src_transformation` is an identity matrix and as both anatomical
world coordinate systems have the same order of axes (first axis:
left--right, second axis: anterior--posterior, third axis:
superior--inferior) the mapping from `src_volume` to our choice of world
coordinate system, which is provided via
`src_to_aligned_transformation`, *almost* remains an identity matrix as
well. However, as the first two axes are flipped, we find a -1 rather
than a 1 there:
```python
print(volume.src_to_aligned_transformation)
# [[-1.  0.  0.  0.]
#  [ 0. -1.  0.  0.]
#  [ 0.  0.  1.  0.]
#  [ 0.  0.  0.  1.]]
```

If we wish to do so, we may now choose an "exotic" anatomical world
coordinate system, and everything will be adjusted accordingly:
```python
volume.system = "IAR"

voxels2world = volume.aligned_transformation
world2voxels = np.linalg.inv(voxels2world)
voxel_index_of_world_origin = world2voxels[:3] @ world_origin
print(voxel_index_of_world_origin)
# [9. 9. 9.]

print(volume.aligned_transformation)
# [[ 1.  0.  0 -9.]
#  [ 0.  1.  0 -9.]
#  [ 0.  0.  1 -9.]
#  [ 0.  0.  0  1.]]
```

As we now swapped all coordinate axes *and* reversed the order of
axes compared to our identity transformation matrix
`src_transformation`, the rotational part of the mapping
`src_to_aligned_transformation` is now a flipped, negated identity
matrix:
```python
print(volume.src_to_aligned_transformation)
# [[ 0.  0. -1.  0.]
#  [ 0. -1.  0.  0.]
#  [-1.  0.  0.  0.]
#  [ 0.  0.  0.  1.]]
```
Still, the voxel data is correctly moving along:
```python
print(volume.src_volume[0, 0, 0] == volume.aligned_volume[9, 9, 9])
# True
```


Loading Images
--------------

### Loading and Stacking DICOM Images

Loading DICOM files requires the `pydicom` package. Currently, loading
multiple files with 2D slices that together form a 3D volume is
supported.

To load DICOM files in the folder `/foo` and stack them into a `Volume`
instance, call:
```python
volume = dicom.open_stack("/foo")
```
In this case, the alphanumerically first loadable DICOM file and all
files with the same *Series Instance UID* (0020,000E) in the folder
`/foo` will be stacked.

We can also specify a file directly, e.g. `/foo/bar.dcm`:
```python
volume = dicom.open_stack("/foo/bar.dcm")
```
In this case, all files in the folder `/foo` that share their *Series
Instance UID* with `bar.dcm` will be stacked.

In both cases, stacking will *not* take the loaded file names into
account but use the files' position and orientation information (*Image
Position (Patient)* (0020,0032) and *Image Orientation (Patient)*
(0020,0037)) to determine their stacking order -- which is, in fact, the
only meaningful way of stacking DICOM files.

For more options, see the documentation of the `mvloader.dicom` module.

### Loading NIfTI Images

Loading NIfTI files requires the `nibabel` package. Currently, loading
3D volumes (both `.nii` and `.nii.gz`) is supported.

To load the file `/foo/bar.nii` into a `Volume` instance, call:
```python
volume = nifti.open_image("/foo/bar.nii")
```
For more options, see the documentation of the `mvloader.nifti` module.

### Loading NRRD Images

Loading NRRD files requires the `pynrrd` package. Currently, loading 3D
volumes with scalar data and with a defined patient-based coordinate
system is supported. This means that

*  the NRRD header's *space*, *space directions*, and *space origin*
   fields must be present, and
*  the *space* field's value must be *right-anterior-superior*,
   *left-anterior-superior*, *left-posterior-superior*, *RAS*, *LAS*, or
   *LPS*. Furthermore,
*  if the *kinds* field is present, all of its entries must be either
   *domain* or *space*.

To load the file `/foo/bar.nrrd` into a `Volume` instance, call:
```python
volume = nrrd.open_image("/foo/bar.nrrd")
```
For more options, see the documentation of the `mvloader.nrrd` module.


Saving Images
-------------

Saving DICOM images is currently not supported (and most likely won't be
in the foreseeable future).

### Saving NIfTI and NRRD Images

Saving NIfTI and NRRD images works pretty much the same.

#### Saving NumPy Array Data
If the voxel data is present as a NumPy array, one might use
`save_image`:
```python
nifti.save_image(path, data, transformation)
nrrd.save_image(path, data, transformation, system)
```
Here, `path` is the file path, `data` is the three-dimensional array
containing the voxel data, and `transformation` is a matrix that maps
from `data`'s voxel indices to an anatomical world coordinate system --
"RAS" in the case of NIfTI and also by default in the case of NRRD.

However, as the NRRD format allows to specify other coordinate systems,
`nrrd.save_image()` has an additional parameter, `system`, which may be
used to specify a different anatomical world coordinate system for the
saved image.

#### Saving `Volume` Instance Data

If the image data is available as a `Volume` instance, one might prefer
using `save_volume`:
```python
nifti.save_volume(path, volume, src_order)
nrrd.save_volume(path, volume, src_order, src_system)
```
Here, `path` is again the file path, and `volume` is the `Volume`
instance to be saved. Additionally, `src_order` is a boolean that
determines the order of the voxels to be saved: if `True`, voxels will
be sorted as in `volume.src_volume`; if `False`, voxels will be sorted
as in `volume.aligned_volume`.

Again, the NRRD function has an additional parameter for choosing the
anatomical world coordinate system: if `src_system` is True, the
function will try to use `volume.src_system` as the saved image's
anatomical world coordinate system; if `False`, the function will try to
use `volume.system` instead. Why *try to*? Because not all coordinate
system's supported by `Volume` are supported by the NRRD format (and
vice versa). Thus, if an unsupported system is detected,
`nrrd.save_volume` will silently use "RAS" -- with a correctly adjusted
transformation matrix, of course.

For more options, see the documentation of the respective `save_*`
functions.


History
-------

### (2018-02-15)

*MVloader* resulted from my PhD work. The different parts grew between
2011 and now, as part of my actual, still unreleased PhD project
(*cordial* -- the cord image analyzer) and should work pretty stable by
now. I decided to make *MVloader* a self-contained package, as I thought
it might be helpful to others who also struggle with loading medical
image volumes and handling their coordinate systems. Before uploading, I
ported everything from Python 2 to Python 3.5+, adjusted namings, and
expanded the documentation. I hope I did not introduce any bugs on the
way (a test suite is unfortunately still missing) -- if you find any,
please let me know.