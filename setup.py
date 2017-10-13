from setuptools import setup


DESCRIPTION = "Implementation of the Bayesian Sets algorithm and variants"

setup(
    name="BayesSets",
    version="0.2",
    packages=["bayessets"],
    install_requires=[
        'numpy>=1.13.3',
        'scipy>=0.19.1'
    ],

    author="Diorge Brognara",
    author_email="diorge.bs@gmail.com",
    description=DESCRIPTION,
    license="MIT",
    keywords="bayes set expansion machine learning information retrieval",
    url="https://github.com/MaLL-UFSCar/bayessets"
)
