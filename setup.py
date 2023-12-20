from setuptools import find_packages, setup
from typing import List

hypen_e = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    return a list of requirrments

    '''
    requirements=[]
    with open(file_path) as file_object:
        requirements = file_object.readline()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if hypen_e in requirements:
            requirements.remove(hypen_e)
            
    return requirements

setup(
name = 'mlproject',
version='0.0.1',
author='Farok',
email='farok.amo@gmail.com',
packages=find_packages(),
install_req = get_requirements('requirements.txt')

)