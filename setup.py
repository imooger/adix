# setup.py
#

from setuptools import setup, find_packages

from pathlib import Path

source_root = Path(".")
with (source_root / "README.md").open(encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="adix",
    version="0.2.0",
    project_urls={
        'Homepage': 'https://imooger.github.io',
#        'Changelog': 'https://github.com/your-username/your-package-name/blob/main/CHANGELOG.md',
#        'Discussions': 'https://github.com/your-username/your-package-name/discussions',
        'Documentation': 'https://imooger.github.io/docs',
        'Repository': 'https://github.com/imooger/adix',
        'Bug Tracker': 'https://github.com/imooger/adix/issues',
    },
    author='Marek Jindrich',
    author_email='adix.eda@gmail.com',
    description='Automated exploratory data analysis (EDA).',
    python_requires='>=3.10',
    package_data={'adix': ['datasets/*.csv']},
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Environment :: Console",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Framework :: IPython",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    install_requires=[
        'pandas>=2.1.0',
        'numpy>=1.26.2',
        'matplotlib>=3.8.2',
        'seaborn>=0.13.0',
        'scipy>=1.11.4',
        'Jinja2>=3.1.2',
        'notebook>=7.0.6',
        'wordcloud>=1.8.1',
       	'nltk >= 3.6.7'
    ],
    keywords="pandas eda data data-science data-analysis python jupyter ipython",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
)
