from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent

NAME = "arpes"
DESCRIPTION = "Modular data analysis code for angle resolved photoemission spectroscopy (ARPES)"
URL = "https://gitlab.com/lanzara-group/python-arpes"
EMAIL = "chstan@berkeley.edu"
AUTHOR = "Conrad Stansbury"
LICENSE = "GPLv3"
LONG_DESCRIPTION = (ROOT / "README.rst").read_text()

about = {}
with open("./arpes/__init__.py") as fp:
    exec(fp.read(), about)

VERSION = about["VERSION"]
DOCUMENTATION_URL = "https://arpes.readthedocs.io/"

with open(ROOT / "requirements.txt") as f:
    requirements = f.read().splitlines()

packages = find_packages(
    exclude=(
        "tests",
        "source",
        "info_session",
        "scripts",
        "docs",
        "example_configuration",
        "conda",
        "figures",
        "exp",
        "datasets",
        "resources",
    )
)


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=packages,
    install_requires=requirements,
    include_package_data=True,
    license=LICENSE,
)
