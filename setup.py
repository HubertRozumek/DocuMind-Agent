"""
Setup script for DocuMind-Agent.

For modern builds, use pyproject.toml instead.
This file exists for backward compatibility.
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="DocuMind-Agent",
    version="1.0",
    author="Hubert Rozumek",
    author_email="hubert.rozumek9@gmail.com",
    description="Intelligent RAG agent with self-correction and multi-layer grading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HubertRozumek/DocuMind-Agent",
    project_urls={
        "Documentation": "https://github.com/HubertRozumek/DocuMind-Agent/blob/main/README.md",
        "Source Code": "https://github.com/HubertRozumek/DocuMind-Agent",
    },
    packages=find_packages(where=".", exclude=["tests*", "data*", "models*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9,<3.13",
    install_requires=[
        "streamlit>=1.28.0",
        "python-dotenv>=1.0.0",
        "langchain>=0.1.0,<0.2.0",
        "langchain-community>=0.0.10,<0.1.0",
        "langgraph>=0.0.20,<0.1.0",
        "chromadb>=0.4.18,<0.5.0",
        "sentence-transformers>=2.2.2,<3.0.0",
        "torch>=2.0.0,<3.0.0",
        "transformers>=4.35.0,<5.0.0",
        "pymupdf>=1.23.0,<2.0.0",
        "pypdf>=3.17.0,<4.0.0",
        "numpy>=1.24.0,<2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "DocuMind-Agent=streamlit_app.app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
