from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='microagg1d',
    version='0.1.3',
    packages=['microagg1d', 'microagg1d.tests'],
    author='Felix Stamm',
    author_email='felix.stamm@cssh.rwth-aachen.de',
    description='A python package for optimal univariate microaggregation in 1d',
    install_requires=[
              'numpy',
              'numba',
          ],
    python_requires='>=3.8',
    long_description=long_description,
    long_description_content_type="text/markdown"
)
