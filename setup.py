import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='exhaufs',
    version='0.07',
    scripts=[
        'exhaufs',
    ],
    author='HSE.Bioinformatics',
    author_email='s.a.nersisyan@gmail.com',
    description='Exhaustive Feature Selection',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/s-a-nersisyan/ExhauFS',
    packages=setuptools.find_packages(),
    install_requires=[
        'scipy',
        'scikit-learn',
        'numpy',
        'pandas',
        'lifelines',
        'scikit-survival',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
    ],
)
