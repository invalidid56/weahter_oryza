from setuptools import setup

setup(
   name='weather_oryza',
   version='1.0',
   description="DNN Based Model to Estimate Leaf Temp. GPP, RECO of Oryza from a few Weather Data",
   author='Junseo Kang',
   author_email='invalidid56@snu.ac.kr',
   packages=['weather_oryza'],
   install_requires=['keras', 'pandas', 'sklearn', 'matplotlib', 'tqdm'],
)