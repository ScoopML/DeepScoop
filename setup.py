from setuptools import setup

setup(
    name='deepscoop',
    version='0.1',
    description='An efficient deep learning library for ScoopML.',
    author='Harish S.G',
    author_email="harishsg99@gmail.com",
    packages=['deepscoop', 'deepscoop.autograd', 'deepscoop.autograd.tensor_library', 'deepscoop.nn', 'deepscoop.optim'],
    install_requires=['numpy>=1.12', 'future>=0.15.2'],
    keywords=['Automatic differentiation', 'backpropagation', 'gradients',
              'machine learning', 'optimization', 'neural networks',
              'Python', 'Numpy', 'Scipy', 'deep learning'],
    license=None,
)
