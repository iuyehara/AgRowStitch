from setuptools import setup, find_packages

setup(
    name='AgRowStitch',
    version='0.1.0',
    description='A tool for stitching together images from agricultural scenes using LightGlue and OpenCV',
    author='GEMINI Breeding',
    author_email='ikuyehara@ucdavis.edu',
    url='https://github.com/iuyehara/AgRowStitch',
    packages=find_packages(),
    install_requires=[
        'numpy==2.2.6',
        'opencv-python==4.10.0.84',
        'pandas==2.2.3',
        'PyYAML==6.0.2',
        'scipy==1.14.1',
        'torch==2.5.1+cu118',
        'torchvision==0.20.1'
    ],
    dependency_links=[
        'https://github.com/cvg/LightGlue.git#egg=LightGlue'
    ],
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
