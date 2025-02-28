# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import functools
import os
import sys
from itertools import chain
import site
from pathlib import Path

import openvino
from openvino.runtime.utils.node_factory import NodeFactory

from .convert_tokenizer import convert_tokenizer
from .str_pack import pack_strings, unpack_strings
from .utils import add_greedy_decoding, connect_models

_ext_name = "user_ov_extensions"
if sys.platform == "win32":
    _ext_name = f"{_ext_name}.dll"
elif sys.platform == "darwin":
    _ext_name = f"lib{_ext_name}.dylib"
elif sys.platform == "linux":
    _ext_name = f"lib{_ext_name}.so"
else:
    sys.exit(f"Error: extension does not support the platform {sys.platform}")

# when the path to the extension set manually
_extension_path = os.environ.get("OV_TOKENIZER_PREBUILD_EXTENSION_PATH")
if _extension_path and Path(_extension_path).is_file():
    # when the path to the extension set manually
    _ext_path = Path(_extension_path)
else:
    site_packages = chain((Path(__file__).parent.parent, ), site.getusersitepackages(), site.getsitepackages())
    _ext_path = next(
        (
            ext
            for site_package in map(Path, site_packages)
            if (ext := site_package / __name__ / "lib" / _ext_name).is_file()
        ),
        _ext_name  # Case when the library can be found in the PATH/LD_LIBRAY_PATH
    )

del _ext_name

# patching openvino
old_core_init = openvino.runtime.Core.__init__


@functools.wraps(old_core_init)
def new_core_init(self, *args, **kwargs):
    old_core_init(self, *args, **kwargs)
    self.add_extension(str(_ext_path))  # Core.add_extension doesn't support Path object


openvino.runtime.Core.__init__ = new_core_init

_factory = NodeFactory()
_factory.add_extension(_ext_path)
