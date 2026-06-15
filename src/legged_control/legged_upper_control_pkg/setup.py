from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[
        'legged_upper_control',
        'legged_upper_control.core',
        'legged_upper_control.controllers',
        'legged_upper_control.fleet',
        'legged_upper_control.apps',
    ],
    package_dir={'': '.'},
)

setup(**d)
