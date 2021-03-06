from setuptools import setup, find_packages
import avatarpy
setup(
    name                = 'avatarpy',
    version             = avatarpy.__version__,
    description         = 'avatar analysis module',
    author              = 'jeakwon',
    author_email        = 'onlytojay@gmail.com',
    url                 = 'https://github.com/jeakwon/avatarpy',
    install_requires    =  ['numpy', 'pandas', 'scipy', 'sklearn', 'jupyter',  'plotly', 'cufflinks', 'matplotlib', 'seaborn', 'statannot', 'umap-learn'],
    packages            = find_packages(exclude = []),
    keywords            = ['avatarpy'],
    python_requires     = '>=3',
    package_data={'avatarpy': ['data/*.csv']},
    include_package_data = True, 
    zip_safe            = False,
    classifiers         = [
        'Programming Language :: Python :: 3.6',
    ],
)