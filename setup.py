from setuptools import find_packages, setup
from typing import List

HPHEN_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:
    '''
    This function returns list of the required libraries
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HPHEN_E_DOT in requirements:
            requirements.remove(HPHEN_E_DOT)
    return requirements




setup(
    name="recommendation_system",
    author="Phoenix",
    author_email="writingcode2022@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)