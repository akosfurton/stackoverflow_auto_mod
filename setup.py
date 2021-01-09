from datetime import datetime

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    pkg_list = f.read().splitlines()

current_datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

setup(
    name="micdrop",
    version="0.1",
    description="Repository for Apple Take Home Interview",
    author="Akos Furton",
    author_email="akosfurton@gmail.com",
    packages=find_packages(),
    install_requires=pkg_list,
)
