"""The package imports without PyYAML.

The QuPath PPM extension installs this package with ``pip --no-deps`` into an Appose
environment that has no PyYAML, and ppm_library imports it when it loads. A top-level
``import yaml`` anywhere on that path makes PPM analysis fail on a machine with no
microscope. Only the BaSiC profile files and the focus manifest need YAML, and they import
it where they read or write.
"""

import subprocess
import sys

_BLOCK_YAML = """
import importlib.abc, sys
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "yaml":
            raise ImportError("yaml blocked")
        return None
sys.meta_path.insert(0, _Block())
import microscope_imageprocessing
from microscope_imageprocessing.correction.background import BackgroundCorrectionUtils
from microscope_imageprocessing.debayering.cpu import CPUDebayer
from microscope_imageprocessing.io.writer import ome_tiff_writer
print("ok")
"""


def test_package_imports_without_yaml():
    result = subprocess.run([sys.executable, "-c", _BLOCK_YAML], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
