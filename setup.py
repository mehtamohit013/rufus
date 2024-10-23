from setuptools import setup, find_packages

setup(
    name="package_name",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    author="Mohit Mehta",
    author_email="mehtamohit013@gmail.com",
    description="Intelligent Web Scraper",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/yourusername/package_name",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)