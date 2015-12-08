from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Gamemodule',
  ext_modules = cythonize("game.pyx"),
)
setup(
  name = 'TDAgent',
  ext_modules = cythonize("rlagent.pyx"),
)
setup(
  name = 'TDAgent',
  ext_modules = cythonize("main.pyx"),
)