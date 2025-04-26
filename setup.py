from setuptools import setup, find_packages

setup(
    name="eeg-seizure-classification",
    version="0.1.0",
    description="EEG signal classification for seizure detection",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "tensorflow>=2.5.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
    ],
    python_requires=">=3.7",
)
