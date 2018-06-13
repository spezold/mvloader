#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pkg_resources

# Following one of the recipes in [*]_ (probably not the best one, though)
#
# .. [*] https://packaging.python.org/guides/single-sourcing-package-version/#single-sourcing-the-version (20180613)
__version__ = pkg_resources.get_distribution("mvloader").version
