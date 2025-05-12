from setuptools import setup, find_packages

setup(
    name="telco_churn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "plotly>=5.18.0",
        "datasets>=2.18.0",
        "huggingface-hub>=0.21.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Telco Customer Churn Analysis Project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TelcoChurn",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 