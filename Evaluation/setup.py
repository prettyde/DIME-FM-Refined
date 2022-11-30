import setuptools

VERSION = '0.0.1'

setuptools.setup(name='vision_benchmark',
                 author='chunyl',
                 author_email='chunyl@microsoft.com',
                 version=VERSION,
                 python_requires='>=3.6',
                 packages=setuptools.find_packages(exclude=['test', 'test.*']),
                 package_data={'': ['resources/*']},
                 install_requires=[
                     'yacs~=0.1.8',
                     'scikit-learn',
                     'timm>=0.3.4',
                   