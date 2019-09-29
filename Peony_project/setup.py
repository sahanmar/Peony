from distutils.core import setup

setup(
    name="PeonyPackage",
    version="1.0",
    description="Pakcage for easier Mongo database usage",
    packages=["PeonyPackage"],
    package_dir={"PeonyPackage": "PeonyPackage/"},
)
