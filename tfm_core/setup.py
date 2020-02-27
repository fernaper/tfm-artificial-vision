from setuptools import setup

setup(
    name='tfm_core',
    version='1.0',
    packages=[
        'tfm_core',
    ],
    include_package_data=True,
    install_requires=[
        'tensorflow-gpu',
        'numpy',
        'opencv-python',
        'cv2-tools'
    ],
)
