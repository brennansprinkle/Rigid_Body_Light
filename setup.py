from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys
import re


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        install_dir = os.path.abspath(
            sys.prefix
        )  # Use sys.prefix to install in the environment's prefix

        double_precision = os.environ.get("DOUBLEPRECISION", "0")
        if double_precision not in {"0", "1"}:
            raise ValueError(
                "Environment variable DOUBLEPRECISION must be '0' or '1'. It is 0 by default- set to 1 to compile in double precision."
            )
        cmake_args = [
            "-DCMAKE_INSTALL_PREFIX=" + install_dir,  # Point to the install directory
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DCMAKE_BUILD_TYPE=" + ("Debug" if self.debug else "Release"),
            f"-DDOUBLEPRECISION={double_precision}",
        ]

        build_args = ["--config", "Release"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"] + build_args,
            cwd=self.build_temp,
        )


setup(
    name="CManyBodies",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[CMakeExtension("CManyBodies")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
