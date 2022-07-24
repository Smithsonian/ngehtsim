from setuptools import setup, find_packages
import os
import versioneer


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


extra_files = package_files('./ngehtsim/weather_data')

setup(name='ngehtsim',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='ngEHT simulation tools',
      author='Dom Pesce',
      author_email='dpesce@cfa.harvard.edu',
      url='https://github.com/Smithsonian/ngehtsim',
      license='GPLv3',
      packages=find_packages(),
      package_data={'': extra_files},
      include_package_data=True,
      install_requires=['astropy',
                        'ehtim',
                        'matplotlib',
                        'numpy',
                        'scipy',
                        'ngehtutil']
      )
