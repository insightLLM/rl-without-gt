from setuptools import setup, find_packages

setup(
    name='lib',
    version='0.0.0',
    description='',
    author='Rihui Xin (xrh)',
    packages=find_packages(include=['lib',]),

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)