from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="perceive",
    author="EduardoV06",
    author_email="evodopoives@protonmail.com",
    description="A CLI tool for image similarity detection.",
    version="0.0.1",
    license="MIT",
    packages=find_packages(include=["perceive", "perceive.*"]),
    entry_points={
        "console_scripts": [
            "perceive = perceive.__main__:main"
        ]
    },
    python_requires=">=3.11",
    install_requires=requirements,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)