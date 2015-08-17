from setuptools import setup, find_packages

setup(
        name='eights',
        version='0.0.1',
        url='https://github.com/dssg/eights',
        author='Center for Data Science and Public Policy',
        description='A library and workflow template for machine learning',
        packages=find_packages(),
        install_requires=('numpy',
                          'scikit-learn', 
                          'matplotlib', 
                          'SQLAlchemy',
                          'joblib',
                          'pdfkit'),
        zip_safe=False)
