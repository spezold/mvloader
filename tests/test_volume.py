#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from _ast import alias

import numpy as np
from pathlib import Path
import shutil
import unittest

from mvloader import nifti, nrrd
from mvloader.anatomical_coords import permutation_matrix, find_closest_permutation_matrix
from mvloader.volume import Volume
from tests.helpers import transformation_matrix, random_transformation_matrix, random_voxel_data, generate_test_data


class TestCreateVolume(unittest.TestCase):

    def setUp(self):

        # Each value corresponds to its concatenated index: data_array[1, 2, 3] == 123 etc.
        self._data_array = np.arange(1000).reshape(10, 10, 10)

    @property
    def data_array(self):

        return self._data_array.copy()

    def test_identity(self):

        v = Volume(src_voxel_data=self.data_array,
                   src_transformation=np.eye(4),
                   src_system="RAS")

        # Test correct default parameters
        self.assertEqual(v.src_object, None)
        self.assertEqual(v.system, "RAS")

        # Test correct storage of immediate inputs
        np.testing.assert_array_equal(v.src_volume, self.data_array)
        np.testing.assert_array_equal(v.src_transformation, np.eye(4))
        self.assertEqual(v.src_system, "RAS")

        # Test derived values
        np.testing.assert_array_equal(v.src_spacing, np.ones(3))
        np.testing.assert_array_equal(v.src_to_aligned_transformation, np.eye(4))

        np.testing.assert_array_equal(v.aligned_volume, self.data_array)
        np.testing.assert_array_equal(v.aligned_spacing, np.ones(3))
        np.testing.assert_array_equal(v.aligned_transformation, np.eye(4))

    def test_default_parameters(self):

        system = "LPS"
        src_object = "I am the source object!"

        v = Volume(src_voxel_data=self.data_array,
                   src_transformation=np.eye(4),
                   src_system="RAS",
                   system=system,
                   src_object=src_object)

        self.assertEqual(v.src_object, src_object)
        self.assertEqual(v.system, system)

    def test_wrong_src_voxel_data(self):

        # No voxels
        with self.assertRaises(Exception):
            v = Volume(src_voxel_data=None,
                       src_transformation=np.eye(4),
                       src_system="RAS")

        # 2D voxels
        with self.assertRaises(Exception):
            v = Volume(src_voxel_data=self.data_array[0],
                       src_transformation=np.eye(4),
                       src_system="RAS")

    def test_wrong_src_transformation(self):

        # No source transformation
        with self.assertRaises(Exception):
            v = Volume(src_voxel_data=self.data_array,
                       src_transformation=None,
                       src_system="RAS")

        # Wrong source transformation (not a 3D transformation matrix)
        with self.assertRaises(Exception):
            v = Volume(src_voxel_data=self.data_array,
                       src_transformation=np.ones((4, 4)),
                       src_system="RAS")

    def test_wrong_src_system(self):

        # No source system
        with self.assertRaises(Exception):
            v = Volume(src_voxel_data=self.data_array,
                       src_transformation=np.eye(4),
                       src_system=None)

        # Wrong source system (wrong characters)
        with self.assertRaises(Exception):
            v = Volume(src_voxel_data=self.data_array,
                       src_transformation=np.eye(4),
                       src_system="foo")

        # Wrong source system (one dimension missing)
        with self.assertRaises(Exception):
            v = Volume(src_voxel_data=self.data_array,
                       src_transformation=np.eye(4),
                       src_system="RA")

        # Wrong source system (conflicting characters, one dimension missing)
        with self.assertRaises(Exception):
            v = Volume(src_voxel_data=self.data_array,
                       src_transformation=np.eye(4),
                       src_system="LRP")

        # Wrong source system (conflicting characters, all dimensions present)
        with self.assertRaises(Exception):
            v = Volume(src_voxel_data=self.data_array,
                       src_transformation=np.eye(4),
                       src_system="LRPS")

    def test_wrong_system(self):

        # No system
        with self.assertRaises(Exception):
            v = Volume(src_voxel_data=self.data_array,
                       src_transformation=np.eye(4),
                       src_system="RAS",
                       system=None)

        # Wrong system (wrong characters)
        with self.assertRaises(Exception):
            v = Volume(src_voxel_data=self.data_array,
                       src_transformation=np.eye(4),
                       src_system="RAS",
                       system="foo")

        # Wrong system (conflicting characters, one dimension missing)
        with self.assertRaises(Exception):
            v = Volume(src_voxel_data=self.data_array,
                       src_transformation=np.eye(4),
                       src_system="RAS",
                       system="LRP")

        # Wrong system (conflicting characters, all dimensions present)
        with self.assertRaises(Exception):
            v = Volume(src_voxel_data=self.data_array,
                       src_transformation=np.eye(4),
                       src_system="RAS",
                       system="LRPS")

    def test_almost_identity(self):

        tau = 2 * np.pi
        src_transformation = transformation_matrix(angles=(0.01 * tau, 0.02 * tau, 0.03 * tau))

        # Just make sure that transformation_matrix() is actually not returning the identity matrix
        self.assertRaises(AssertionError, np.testing.assert_array_equal, src_transformation, np.eye(4))

        v = Volume(src_voxel_data=self.data_array,
                   src_transformation=src_transformation,
                   src_system="RAS")

        # As we have almost no rotation, src_volume and aligned_volume should still match
        np.testing.assert_array_equal(v.src_volume, self.data_array)
        np.testing.assert_array_equal(v.aligned_volume, self.data_array)

        # As both the arrays and coordinate systems match, aligned_transformation and src_to_aligned_transformation
        # should match src_transformation
        np.testing.assert_array_equal(v.src_to_aligned_transformation, src_transformation)
        np.testing.assert_array_equal(v.aligned_transformation, src_transformation)

        # Same is true for the scalings
        np.testing.assert_array_equal(v.src_spacing, np.ones(3))
        np.testing.assert_array_equal(v.aligned_spacing, np.ones(3))

    def test_random_values(self):

        # Enter random values for both the transformation matrix and the data array
        src_transformation = random_transformation_matrix()
        src_voxel_data = random_voxel_data(size=(9, 10, 11))

        v = Volume(src_voxel_data=src_voxel_data, src_transformation=src_transformation, src_system="ASL")

        # Values in v should still be the same as the initially given ones
        np.testing.assert_array_equal(v.src_transformation, src_transformation)
        np.testing.assert_array_equal(v.src_volume, src_voxel_data)

        # A quick sanity check should give us the same statistics for both volume representations
        s = v.src_volume
        a = v.aligned_volume
        np.testing.assert_array_almost_equal([np.mean(s), np.std(s)], [np.mean(a), np.std(a)])


class TestLoadVolume(unittest.TestCase):

    def setUp(self):

        self._testdata_dir, self._testdata_array, self._src_testdata_arrays, self._src_transformations = generate_test_data()
        # `paths`:       key: alignment coordinate system triple, value: full file path
        # `voxel_sizes`: key: alignment coordinate system triple, value: voxel size
        # `src_sytems`:  key: alignment coordinate system triple, value: assumed src_system triple
        self.paths, self.voxel_sizes, self.src_systems, self.almost_aligneds = self._assemble_metadata_from_filenames()

    def testAll(self):

        for alignment_triple in sorted(self.paths.keys()):

            path = self.paths[alignment_triple]
            voxel_size = self.voxel_sizes[alignment_triple]
            src_system = self.src_systems[alignment_triple]
            v = nrrd.open_image(path, verbose=False) if path.endswith(".nrrd") else nifti.open_image(path, verbose=False)

            self.assertEqual(v.system, "RAS", msg=alignment_triple)  # All volumes should be aligned to RAS by default

            # Changing the assumed system to the same as `src_sytem` allows us to simply compare `aligned_volume` with
            # the given `testdata_array` (and implicitly lets us check if changing coordinate systems works properly)
            v.system = self.src_systems[alignment_triple]

            # Check the aligned_transformation matrix
            np.testing.assert_array_almost_equal(self._src_transformations[alignment_triple], v.src_transformation, err_msg=alignment_triple)

            # Check the voxel sizes for both array representations
            np.testing.assert_array_almost_equal(voxel_size, v.src_spacing, err_msg=alignment_triple)
            np.testing.assert_array_almost_equal(voxel_size, np.abs(permutation_matrix(src_system, alignment_triple) @ v.aligned_spacing), err_msg=alignment_triple)

            # Check the rotational part of src_transformation: if it is actually and not almost aligned, it should
            # be the same as the permutation matrix that we can recreate from the file name
            if not self.almost_aligneds[alignment_triple]:
                should_transformation = permutation_matrix(alignment_triple, src_system)
                is_transformation = v.src_transformation[:3, :3] * (1 / np.linalg.norm(v.src_transformation[:3, :3], axis=0)[np.newaxis, :])
                np.testing.assert_array_almost_equal(should_transformation, is_transformation, err_msg=alignment_triple)

            # Check the actual array contents of src_volume and aligned_volume (should exactly match the src_testdata
            # and testdata arrays)
            np.testing.assert_array_equal(self._src_testdata_arrays[alignment_triple], v.src_volume, err_msg=alignment_triple)
            np.testing.assert_array_equal(self._testdata_array, v.aligned_volume, err_msg=alignment_triple)

            # Check the value at the coordinate system origins: should be zero
            world_origin = np.asarray([0, 0, 0, 1])
            i_src = tuple(np.round(np.linalg.inv(v.src_transformation)[:3] @ world_origin).astype(np.int))
            i_aligned = tuple(np.round(np.linalg.inv(v.aligned_transformation)[:3] @ world_origin).astype(np.int))

            self.assertEqual(self._testdata_array[0, 0, 0], v.src_volume[i_src], msg=alignment_triple)
            self.assertEqual(self._testdata_array[0, 0, 0], v.aligned_volume[i_aligned], msg=alignment_triple)

            # For all coordinates: Check if we can get from aligned_volume's voxel indices to src_volume's voxel
            # indices (via aligned_transformation and src_to_aligned_transformation) and find the same values there.
            # I guess this completes our transformation checks.
            aligned_indices = np.indices(v.aligned_volume.shape).reshape(v.aligned_volume.ndim, -1)
            src_indices = (np.linalg.inv(v.src_to_aligned_transformation) @ v.aligned_transformation)[:3] @ np.r_[aligned_indices, np.ones((1, aligned_indices.shape[-1]))]
            src_indices = np.round(src_indices).astype(np.int)
            np.testing.assert_array_equal(v.src_volume[tuple(src_indices)], v.aligned_volume[tuple(aligned_indices)], err_msg=alignment_triple)

    def _parse_name(self, testdata_filename):

        name = str(Path(testdata_filename).name).replace(".nii.gz", "").replace(".nrrd", "")
        coord_triples, voxel_size = name.split("-")
        if "2" not in coord_triples:
            alignment_triple = coord_triples
            src_system_triple = "RAS"
        else:
            alignment_triple, src_system_triple = coord_triples.split("2")
        alignment_triple = alignment_triple[-3:]
        voxel_size = tuple(float(v) for v in voxel_size.split("x"))
        almost_aligned = name.startswith("a")

        return alignment_triple, src_system_triple, voxel_size, almost_aligned

    def _assemble_metadata_from_filenames(self):

        paths = {}
        voxel_sizes = {}
        src_systems = {}
        almost_aligneds = {}

        for f in Path(self._testdata_dir).iterdir():

            alignment_triple, src_system_triple, voxel_size, almost_aligned = self._parse_name(f)

            paths[alignment_triple] = str(f.resolve())
            voxel_sizes[alignment_triple] = voxel_size
            src_systems[alignment_triple] = src_system_triple
            almost_aligneds[alignment_triple] = almost_aligned


        return paths, voxel_sizes, src_systems, almost_aligneds

    def tearDown(self):

        shutil.rmtree(self._testdata_dir)


class TestCopyVolume(TestLoadVolume):

    def testAll(self):

        is_deep_copy = lambda a, b: not np.may_share_memory(a, b)

        i = 0

        for alignment_triple_src in sorted(self.paths.keys()):

            path_src = self.paths[alignment_triple_src]
            v_src = nrrd.open_image(path_src, verbose=False) if path_src.endswith(".nrrd") else nifti.open_image(path_src, verbose=False)

            for alignment_triple_dst in sorted(self.paths.keys()):

                i += 1

                deep = bool(i % 2)  # Alternate between "deep" and "shallow" copies

                path_dst = self.paths[alignment_triple_dst]
                v_dst = nrrd.open_image(path_dst, verbose=False) if path_dst.endswith(".nrrd") else nifti.open_image(path_dst, verbose=False)

                v_src_copy = v_src.copy_like(v_dst, deep=deep)
                err_msg = "{} -> {}".format(alignment_triple_src, alignment_triple_dst)

                # 1. Compare copy to template

                # 1.1. Systems should be the same
                self.assertEqual(v_src_copy.src_system, v_dst.src_system, msg=err_msg)
                self.assertEqual(v_src_copy.system, v_dst.system, msg=err_msg)

                # 1.2. The closest permutation matrices should match
                np.testing.assert_array_equal(find_closest_permutation_matrix(v_src_copy.src_transformation[:3, :3]),
                                              find_closest_permutation_matrix(v_dst.src_transformation[:3, :3]), err_msg=err_msg)
                np.testing.assert_array_equal(find_closest_permutation_matrix(v_src_copy.aligned_transformation[:3, :3]),
                                              find_closest_permutation_matrix(v_dst.aligned_transformation[:3, :3]), err_msg=err_msg)
                np.testing.assert_array_equal(find_closest_permutation_matrix(v_src_copy.src_to_aligned_transformation[:3, :3]),
                                              find_closest_permutation_matrix(v_dst.src_to_aligned_transformation[:3, :3]), err_msg=err_msg)

                # 2. Compare copy to original

                # 2.1. Check if deep copies are deep and shallow copies are shallow
                self.assertEqual(deep, is_deep_copy(v_src_copy.aligned_volume, v_src.aligned_volume), msg=err_msg)

                # 2.2. When aligned to the same user system, array contents should match
                v_src_copy.system = v_src.system
                np.testing.assert_array_equal(v_src_copy.aligned_volume, v_src.aligned_volume, err_msg=err_msg)


if __name__ == "__main__":

    unittest.main()
