from setuptools import setup, find_packages
setup(
    name="annotation",
    version="0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.14.5',
        'opencv-python>=4.1',
    ]
)