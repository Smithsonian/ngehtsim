from setuptools import setup, find_packages
import versioneer

setup(name='ngehtsim',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
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
                        'ngeht-util']
      )
