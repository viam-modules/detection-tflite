from setuptools import find_packages, setup

setup(
    name="model",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "google-cloud-aiplatform",
        "google-cloud-storage",
        "tensorflow==2.21.0",
        "keras>=3.15,<4",
        "keras-cv==0.9.0",
    ],
    include_package_data=True,
)
