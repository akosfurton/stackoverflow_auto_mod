from setuptools import setup, find_packages


with open("requirements.txt") as f:
    pkg_list = f.read().splitlines()


setup(
    name="okcupid_stackoverflow",
    version="1.0",
    description="Repository for OK Cupid Take Home Interview",
    author="Akos Furton",
    author_email="akosfurton@gmail.com",
    packages=find_packages(),
    install_requires=pkg_list,
)
