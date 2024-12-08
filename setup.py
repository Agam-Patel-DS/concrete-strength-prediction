from setuptools import setup, find_packages
from custom_logs import get_logger
logger = get_logger("project_setup.log")

setup(
    name='ConcreteStrengthPrediction',
    version='0.1',
    description='A machine learning project to predict the compressive strength of concrete',
    author='Agam-Patel-DS',
    author_email='colabwithagam@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)

logger.info("Setup and Requirements Done")