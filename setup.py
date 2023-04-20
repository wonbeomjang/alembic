import setuptools


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="alembic",
    version="0.0.1",
    author="Wonbeom Jang",
    author_email="jtiger958@gmail.com",
    description="Project of pytorch library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wonbeomjang/alembic",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
