from setuptools import setup, find_packages

sys.path[0:0] = ['StarGAN-pytorch']
#from version import __version__

setup(
  name = 'StarGAN-pytorch',
  packages = find_packages(),
  #version = __version__,
  license='MIT',
  description = 'Coding star gan in pytorch',
  author = 'Shauray Singh',
  author_email = 'shauray9@gmail.com',
  url = 'https://github.com/shauray8/StarGAN-pytorch',
  keywords = ['generative adversarial networks', 'machine learning'],
  install_requires=[
      'numpy',
      'tqdm',
      'torch',
      'torchvision',
      'pillow',
  ],
)
