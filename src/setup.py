from setuptools import setup

setup(
   name='src',
   version='1.0',
   description='Local source code for classes',
   author='Eric Bauer',
   author_email='nah@gmail.com',
   packages=['src'],  #same as name
   install_requires=['gym', 'numpy'], #external packages as dependencies
)