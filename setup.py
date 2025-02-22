from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "numpy==1.20.3",
    "matplotlib==3.7.4",
    "imageio-ffmpeg",
    "openai",
    "seaborn",
]

setup(
    name='orso',
    version='1.0',
    author='Chen Bo Calvin Zhang',
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
)