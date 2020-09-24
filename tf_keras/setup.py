#nsml: nsml/ml:cuda9.0-cudnn7-tf-1.11torch0.4keras2.2
from distutils.core import setup
import setuptools

print('setup_test.py is running...')

setup(name='PNS_SAMPLE',
      version='1.0',
      install_requires=[
            'opencv-python',
            "efficientnet"
            # "scikit-learn",
            # "imbalanced-learn==0.5.0"
      ]
      ) ## install libraries, 'keras==xx.xx'