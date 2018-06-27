#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
See the project's "readme" file [MVLOADER1]_ for an overview of MVloader's capabilities.

References
----------
.. [MVLOADER1] https://github.com/spezold/mvloader/blob/master/readme.md (20180627)
"""

import pkg_resources

# Following one of the recipes in [*]_ (probably not the best one, though)
#
# .. [*] https://packaging.python.org/guides/single-sourcing-package-version/#single-sourcing-the-version (20180613)
__version__ = pkg_resources.get_distribution("mvloader").version
