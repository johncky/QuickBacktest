import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="QuickBacktest", # Replace with your own username
    version="0.0.1",
    author="JohnCKY",
    author_email="kleenex092@gmail.com",
    description="Streamline backtesting process of trading strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johncky/quickbacktest",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)