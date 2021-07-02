from setuptools import setup, find_packages

setup(name='Rely',
      version='1.3.1',
      description='Functions for computing simple measures of reliability.',
      url='http://github.com/sahahn/Rely',
      author='Sage Hahn',
      author_email='sahahn@uvm.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'scikit-learn',
          'numpy',
          'pandas',
          'nibabel',
          'joblib',
          'statsmodels'
      ])
