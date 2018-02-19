MVloader
========

*MVloader* is meant to be a tiny helper to load and save *medical
volumetric* data (therefore *MV*) aka. *image volumes*, i.e.
three-dimensional medical images (DICOM, NIfTI, or NRRD), in Python 3.5+.
It is also meant to simplify dealing with their different anatomical
world coordinate systems.


Installing MVloader
-------------------

*MVloader* can be installed using *pip* or *pip3*, depending on your
system:
```shell
pip install git+https://github.com/spezold/mvloader.git
pip3 install git+https://github.com/spezold/mvloader.git
```

This should also work inside *conda*.

*MVloader*'s only dependencies are the underlying image libraries
([*pydicom*](https://github.com/pydicom/pydicom),
[*nibabel*](http://nipy.org/nibabel/),
[*pynrrd*](https://github.com/mhe/pynrrd)) and
[*NumPy*](http://www.numpy.org/), which should all be resolved
automatically during installation.


Motivation: Voxel Indices, World Coordinates, and Patient Anatomy
-----------------------------------------------------------------

When dealing with medical image volumes, one must realize that they live
in two different worlds: their *voxel space* and a *world coordinate
system*, the latter of which has an attached *anatomical meaning*.

### What does a voxel index stand for …

The voxel space very unsurprisingly tells us what image intensity is
stored in what sampling position of the volume: Say, we have a
three-dimensional array ``voxel_data_array`` that contains our image
volume:
```python
import numpy as np
voxel_data_array = np.linspace(0, 1, num=1000).reshape(10, 10, 10)
i, j, k = 0, 0, 0
print(voxel_data_array[i, j, k])
# 0.0
i, j, k = 9, 9, 9
print(voxel_data_array[i, j, k])
# 1.0
```
The call of `print(voxel_data_array[i, j, k])` simply tells us we have a
value of zero in the `[0, 0, 0]` corner of the image cube, a value of
1 in the `[9, 9, 9]` corner, and so on.

### … in terms of the physical world?

What the
voxel index `[i, j, k]` does *not* tell us, is, where in the imaged
patient (or healthy subject) we found this value. This, however, may be
crucial for medical applications.

Medical image formats therefore provide a mapping from voxel indices
to a patient-based world coordinate system: Via rotation, scaling, and
translation, we may map from voxel indices to patient coordinates.
Using homogeneous coordinates, we can store this mapping in a *4x4*
matrix `M`:
```python
r_11, r_12, r_13, r_21, r_22, r_23, r_31, r_32, r_33 = ...  # rotation
s_i, s_j, s_k = ...  # scaling (world units per voxel)
t_x, t_y, t_z = ...  # translation (world units)

M = [[r_11 * s_i, r_12 * s_j, r_13 * s_k, t_x],
     [r_21 * s_i, r_22 * s_j, r_23 * s_k, t_y],
     [r_31 * s_i, r_32 * s_j, r_33 * s_k, t_z],
     [         0,          0,          0,   1]]
M = np.asarray(M)
```
We may now use `M` to find for each voxel index `[i, j, k]` its
respective position `[x, y, z]` in world coordinates:
```python
homogeneous = lambda c3d: np.r_[c3d, 1]  # append 1 to 3D coordinate
x, y, z = M[:3] @ homogeneous([i, j, k])
```

Note that, as mentioned, we had to use homogeneous coordinates for the
transformation, which explains why we append 1 for the world origin's
coordinate; however, by using only the first three rows of the
transformation matrix `M`, our resulting voxel index contains only three
values, as one could expect.

### … in terms of the patient?

What so far remains open, is, what does `[x, y, z]` stand for? We may
well have a value in millimeters (or whatever world units are
assumed/defined) by now, but we still do not know what the value means
in terms of the imaged patient's *anatomy*. For this reason, medical
image formats define the world coordinate system's axes relative to the
patient's body axes:

* one world axis points along the patient's left-right axis and
  therefore its value increases when moving from the patient's left side
  to their right side -- or vice versa,
* one world axis points along the patient's anterior-posterior axis and
  therefore its value increases when moving from the patient's front to
  their back -- or vice versa,
* one world axis points along the patient's superior-inferior axis and
  therefore its value increases when moving from the patient's head to
  their feet -- or vice versa.

This may be encapsulated in a definition like "left-posterior-superior
(LPS)" or "right-anterior-superior (RAS)". In the first case, this means
"`x` increases to the left (*L*), `y` increases to the back (*P* as in
*posterior*), `z` increases to the head (*S* as in *superior*)"; in the
second case, this means "`x` increases to the right (*R*), `y` increases
to the front (*A* as in *anterior*), `z` increases to the head (*S* as
in *superior*)". Such a definition of mapping from
world coordinate system axes to the patient's anatomy is either
implicitly assumed by a particular image format
(for example, DICOM uses LPS, NIfTI uses RAS) or explicity stored in the
image volume's meta information (NRRD defines the *space* field in its
header for that purpose). Notice we still don't know where we are
*absolutely* positioned within the patient (the origin of the coordinate
system is usually defined with respect to some point in the imaging
modality, as far as I know), but at least we now have a *relative*
understanding of what a voxel index means with respect to the patient's
anatomy.

### How can we simplify this in practice?

A remaining open issue is a more practical one: what if we want to
display or process an image volume in a certain anatomical orientation?
Say, we want to display axial slices of the patient or apply a certain image
filter along its left-right axis -- do we always have to consult the
mapping `M` from voxel indices to world coordinates (or its inverse) in
order to find the right voxel indices?

Theoretically, indeed we have to do this: notice that a definition like
"LPS" does *not* tell us that the first *voxel index* increases from the
patient's right to their left. For example, if our transformation matrix
`M` looks something like:
```python
M = np.asarray([[0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]])
```
then voxel index `i` will actually increase with the `z` coordinate of
the world coordinate system (which, remember, in the case of RAS and
LPS means moving towards the patient's head):
```python
i, j, k = 1, 0, 0
x, y, z = M[:3] @ homogeneous([i, j, k])
print("x={}, y={}, z={}".format(x, y, z))
# x=0, y=0, z=1
```
But couldn't we rearrange the voxel array, aligning it with the world
coordinate system so that we can be sure increasing voxel index `i`
indeed always means moving to the patient's right side (for an RAS
world) or left side (for an LPS world)? We can -- precisely, if the
rotational part of `M` contains zeros, ones, and minus ones only;
approximately, if it contains arbitrary rotations. And that is where the
`Volume` class comes into play.

The `Volume` Class
------------------

`MVloader` represents all loaded image volumes as instances of the
`Volume` class.

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
   orientation (see above) and processing data differently and
   consistently along different body axes.

### Voxel data representations

Each `Volume` instance provides two representations of the image
volume's voxel data as three-dimensional NumPy arrays:

* `src_volume` provides the voxels in the same order as they have been
  returned by the underlying image library.
* `aligned_volume` provides the voxels with their axes aligned as
  closely as possible to the anatomical world coordinate system that
  has been chosen by the user. For example, if the user chooses an
  "LPS" anatomical world coordinate system (meaning that the first
  coordinate axis will point to the patient's left side, the second
  axis will point to their back, and the third will point towards their
  head), then `aligned_volume[1, 0, 0]` will lie to the left of
  `aligned_volume[0, 0, 0]`, `aligned_volume[0, 1, 0]` will lie closer
  to the patient's back, and `aligned_volume[0, 0, 1]` will lie closer
  to their head.

### Transformation matrix representations

Each `Volume` instance provides three 4x4 transformation matrices to map
from voxel indices to anatomical world coordinates:

* `src_transformation` maps from `src_volume`'s voxel indices to the
  world coordinate system that has been assumed by the underlying image
  format (which is provided via `Volume`'s `src_system` property).
* `aligned_transformation` maps from `aligned_volume`'s voxel indices
  to the world coordinate system that has been chosen by the user
  (which is provided via `Volume`'s `system` property).
* `src_to_aligned_transformation` maps from `src_volume`'s voxel
  indices to the world coordinate system that has been chosen by the
  user (namely `system`).

Apart from that, the mappings from `src_volume` and `aligned_volume` to
arbitrary anatomical world coordinate systems can be determined via
`Volume`'s methods `get_src_transformation()` and
`get_aligned_transformation()`.

### Choosing a world coordinate system

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

### An example

We create a new `Volume` instance:
```python
import numpy as np
from mvloader.volume import Volume

# Create a simple 10x10x10 volume
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
homogeneous = lambda c3d: np.r_[c3d, 1]  # append 1 to 3D coordinate

voxels2world = volume.src_transformation
world2voxels = np.linalg.inv(voxels2world)

world_origin = homogeneous([0, 0, 0])
voxel_index_of_world_origin = world2voxels[:3] @ world_origin

print(voxel_index_of_world_origin)
# [0. 0. 0.]
```

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
while the origin should still mark the same voxel (the one with value
0), the origin must now lie at the greatest voxel index along these two
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

### Loading and stacking DICOM images

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

### Loading NIfTI images

Loading NIfTI files requires the `nibabel` package. Currently, loading
3D volumes (both `.nii` and `.nii.gz`) is supported.

To load the file `/foo/bar.nii` into a `Volume` instance, call:
```python
volume = nifti.open_image("/foo/bar.nii")
```
For more options, see the documentation of the `mvloader.nifti` module.

### Loading NRRD images

Loading NRRD files requires the `pynrrd` package. Currently, loading 3D
volumes with scalar data and with a defined patient-based coordinate
system is supported. This means that

* the NRRD header's *space*, *space directions*, and *space origin*
  fields must be present, and
* the *space* field's value must be "right-anterior-superior",
  "left-anterior-superior", "left-posterior-superior", "RAS", "LAS", or
  "LPS". Furthermore,
* if the *kinds* field is present, all of its entries must be either
  "domain" or "space".

To load the file `/foo/bar.nrrd` into a `Volume` instance, call:
```python
volume = nrrd.open_image("/foo/bar.nrrd")
```
For more options, see the documentation of the `mvloader.nrrd` module.


Saving Images
-------------

Saving DICOM images is currently not supported (and most likely won't be
in the foreseeable future).

### Saving NIfTI and NRRD images

Saving NIfTI and NRRD images works pretty much the same.

#### Saving NumPy array data
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

#### Saving `Volume` instance data

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