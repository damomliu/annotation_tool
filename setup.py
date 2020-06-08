from setuptools import setup, find_packages
setup(
    name="annotation",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.14.5',
        'opencv-python>=3.4',
    ]
)