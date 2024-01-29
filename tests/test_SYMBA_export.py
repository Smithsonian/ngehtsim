#######################################################
# imports

import ehtim as eh
import ngehtsim.obs.obs_generator as og
import os

#######################################################
# various settings

symba_workdir = './tests/data'
rpicard_path = '/usr/local/src/picard/input_template'
meqsilhouette_path = '/usr/local/src/MeqSilhouette/meqsilhouette/data'

weather_type = 'typical'
addnoise = False
addgains = False
use_two_letter = False

#######################################################
# produce material

# generate a symmetric Gaussian model object
mod = eh.model.Model()
mod = mod.add_circ_gauss(F0=1.0, FWHM=40.0*eh.RADPERUAS)

# save an image of the model
im = mod.make_image(500.0*eh.RADPERUAS, 512)
imfilename = './tests/000000-I-model.fits'
im.save_fits(imfilename)

# initialize the observation generator
settings = {'weather': weather_type}
obsgen = og.obs_generator(settings)

# generate and save the observation
obs = obsgen.make_obs(im, addnoise=addnoise, addgains=addgains)
obs.save_uvfits('./tests/example_dataset.uvfits')

# export SYMBA-compatible input files
master_input_args = {'rpicard_path': rpicard_path,
                     'meqsilhouette_path': meqsilhouette_path,
                     'add_thermal_noise': str(addnoise)}
obsgen.export_SYMBA(symba_workdir=symba_workdir, master_input_args=master_input_args, use_two_letter=use_two_letter)

# move image file to SYMBA working directory
os.system('mv ' + imfilename + ' ' + symba_workdir + '/symba_input/' + imfilename)
