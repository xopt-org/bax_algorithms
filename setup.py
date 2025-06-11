from distutils.core import setup
from os import path

cur_dir = path.abspath(path.dirname(__file__))

with open(path.join(cur_dir, "requirements.txt"), "r") as f:
    requirements = f.read().split()

setup(
    name="bax_algorithms",
    version="0.1",
    packages=["bax_algorithms"],
    url="",
    license="",
    author="Dylan Kennedy",
    author_email="kennedy1@slac.stanford.edu",
    description="Algorithms for BAX acquisition function in Xopt",
    install_requires=requirements,
)