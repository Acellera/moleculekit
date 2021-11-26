import setuptools


requirements = ["numpy>=1.17", "scipy", "pandas", "networkx"]

if __name__ == "__main__":
    with open("README.md", "r") as fh:
        long_description = fh.read()

    setuptools.setup(
        name="molkitten",
        version="0.0.6",
        author="Acellera",
        author_email="info@acellera.com",
        description="A molecule reading/writing and manipulation package.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/acellera/moleculekit/",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: POSIX :: Linux",
        ],
        packages=setuptools.find_packages(
            include=["moleculekit*"],
            exclude=[
                "test-data",
                "*test*",
            ],
        ),
        package_data={
            "moleculekit": [
                "logging.ini",
                "share/*/*/*",
            ],
        },
        zip_safe=False,
        install_requires=requirements,
    )
