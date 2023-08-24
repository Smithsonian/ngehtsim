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
extra_files.append('.././ngehtsim/files/Telescope_Site_Matrix.csv')
extra_files.append('.././ngehtsim/files/Receivers.csv')
extra_files += package_files('./ngehtsim/files/eigenspectra')
extra_files += package_files('./ngehtsim/files/eigenspectra_Tb')

setup(name='ngehtsim',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='ngEHT simulation tools',
      author='Dom Pesce',
      author_email='dpesce@cfa.harvard.edu',
      url='https://github.com/Smithsonian/ngehtsim',
      license='MIT',
      packages=find_packages(),
      package_data={'': extra_files},
      include_package_data=True,
      install_requires=['astropy',
                        'ehtim',
                        'matplotlib',
                        'numpy',
                        'scipy',
                        'ngEHTforecast @ git+https://github.com/aeb/ngEHTforecast.git']
      )
