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
      install_requires=['astropy>=5.2.2',
                        'ehtim>=1.2.7',
                        'matplotlib>=3.7.4',
                        'numpy>=1.23.1, <2.0.0',
                        'scipy>=1.10.1',
                        # 'ngEHTforecast @ git+https://github.com/aeb/ngEHTforecast.git#115bf73e77f23336516ce385521aeb2bae2f9a98'
                        'ngEHTforecast @ git+https://github.com/dpesce/ngEHTforecast.git#84ae29663ad713b4394106c6d76afec68b71cb83'
                        ],
      extras_require={
                      'calib': ['eat @ git+https://github.com/sao-eht/eat.git#94df36a7b45a6ce6dd2dc005c2b71a81c5d80a00',
                                'pandas>=1.4.3']
                     }
      )
