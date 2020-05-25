from setuptools import setup, find_packages
from datetime import datetime

with open("requirements.txt") as f:
    pkg_list = f.read().splitlines()

current_datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

setup(
    name="okcupid_stackoverflow",
    version="0.1",
    description="Repository for OK Cupid Take Home Interview",
    author="Akos Furton",
    author_email="akosfurton@gmail.com",
    packages=find_packages(),
    install_requires=pkg_list,
)
