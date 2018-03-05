#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import unittest

from mvloader.volume import Volume
from tests.helpers import transformation_matrix


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

        # Wrong source transformation
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




if __name__ == "__main__":

    unittest.main()
