from setuptools import setup, find_packages

setup(name='ngehtsim',
      description='ngEHT simulation tools',
      author='Dom Pesce',
      author_email='dpesce@cfa.harvard.edu',
      url='https://github.com/Smithsonian/ngeht-sims',
      packages=find_packages(),
      install_requires=['astropy',
                        'ehtim',
                        'matplotlib',
                        'numpy',
                        'scipy',
                        'git+https://github.com/Smithsonian/ngeht-util.git']
      )
