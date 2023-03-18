import multiprocessing
import os
import sys
import pathlib
import shutil

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


CMAKE_EXTRA = []


class CMakeExtension(Extension):
    def __init__(self, name, sources=None):
        if sources is None:
            sources = []
        super().__init__(name=name, sources=sources)


class BuildExtEx(build_ext):
    def run(self):
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.build_cmake(ext)
            else:
                self.build_extension(ext)
        super().run()

    def build_cmake(self, ext):
        ext_path = pathlib.Path(self.get_ext_fullpath(ext.name))
        ext_path.parent.mkdir(parents=True, exist_ok=True)
        source_dir = pathlib.Path().absolute()
        build_lib = source_dir / 'build' / 'cmake-install'
        build_temp = pathlib.Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)
        config = 'Debug' if self.debug else 'RelWithDebInfo'

        build_params = [
            f'-DCMAKE_BUILD_TYPE={config}',
            f'-DCMAKE_INSTALL_PREFIX={build_lib}',
        ]

        self.spawn(['cmake', '-S', source_dir, '-B', build_temp] + build_params + CMAKE_EXTRA)
        if not self.dry_run:
            self.spawn([
                'cmake',
                '--build', build_temp,
                '--config', config,
                '--parallel', f'{multiprocessing.cpu_count()}',
                # '--target', ext.name[1:]
            ])
            self.spawn(['cmake', '--install', build_temp, '--config', config])

            build_lib = build_lib / 'bin'
            lib_name = f'{ext.name}{os.path.splitext(ext_path)[1]}'
            if os.path.isfile(ext_path / lib_name):
                os.unlink(ext_path / lib_name)
            shutil.copy(build_lib / lib_name, ext_path)


CMAKE_EXTRA = [arg for arg in sys.argv if arg.startswith('-D')]
sys.argv = [arg for arg in sys.argv if not arg.startswith('-D')]


setup(
    name='imgui_py',
    version='0.1.0',
    ext_modules=[
        CMakeExtension('_imgui_py'),
        CMakeExtension('_implot_py'),
    ],
    py_modules=['imgui', 'implot'],
    package_dir={'': 'build/cmake-install/bin'},
    cmdclass={'build_ext': BuildExtEx},
    requires=['numpy']
)
