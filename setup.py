#!/usr/env/bin/python3
from distutils.core import setup

setup(
    name='tensorsv',
    version='0.0.1',
    author='Timothy James Becker',
    author_email='timothyjamesbecker@gmail.com',
    url='https://github.com/timothyjamesbecker/TensorSV',
    license='GPL 3 License',
    description='Moment Based Tensor SV Calling and Genotyping Analysis',
    classifiers=['Intended Audience :: Developers',
                 'License :: GPL 3 License',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Cython',
                 'Programming Language :: C',
                 'Operating System :: POSIX',
                 'Topic :: Software Development :: Libraries :: Python Modules'],
    cmdclass={},
    ext_modules=[],
    packages=['tensorsv'],
    package_data={'tensorsv': ['data/*.json', 'data/*.gz','models/*.hdf5']},
    scripts=['bin/data_prep.py','bin/predict_sv.py','bin/train_sv.py','bin/viz/server/tensor_server.py'])
