from setuptools import setup

setup(
    name='hemoji',
    version='1.0',
    packages=['lib'],
    description='HeMoji library',
    include_package_data=True,
    install_requires=[
        'emoji==0.4.5',
        'h5py==2.7.0',
        'Keras==2.1.3',
        'matplotlib==2.2.5',
        'numpy==1.13.1',
        'scikit-learn==0.19.0',
        'tensorflow==1.4.1',
        'text-unidecode==1.0',
        'tqdm==4.46.0',
        'flask==1.1.2',
        'flask-restful==0.3.8',
        'requests==2.23.0'
    ],
)
