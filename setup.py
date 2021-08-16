from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vanilla-transformer-jax", # Replace with your own username
    version="0.0.3",
    author="bhavnicksm",
    author_email="bhavnicksm@gmail.com",
    description="JAX/Flax implimentation of 'Attention is All You Need' by Vaswani et al.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bhavnicksm/vanilla-transformer-jax",
    project_urls={
        "Bug Tracker": "https://github.com/Bhavnicksm/vanilla-transformer-jax/issues",
    },
    keywords = [
        'transformer',
        'JAX',
        'Flax',
        'Deep-Learning',
    ],
    install_requires=['jax>=0.0.0', 'flax>=0.0.0', 'numpy>=0.0.0'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages= find_packages(where="src"),
    python_requires=">=3.6",
)