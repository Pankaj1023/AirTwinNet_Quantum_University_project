from setuptools import find_packages, setup


setup(
    name="airtwinnet",
    version="0.0.1",
    author="Pankaj Sharma",
    description="AirTwinNet: Digital Twin Enabled Multi-Hybrid Framework",
    packages=find_packages("src"),
    package_dir={"": "src"},
)