#######################################################
# imports

import ehtim as eh
import ngehtsim.obs.obs_generator as og

#######################################################

# generate a symmetric Gaussian model object
mod = eh.model.Model()
mod = mod.add_circ_gauss(F0=1.0,FWHM=40.0*eh.RADPERUAS)

# save an image of the model
im = mod.make_image(500.0*eh.RADPERUAS,512)
imfilename = '000000-I-model.fits'
im.save_fits(imfilename)

# initialize the observation generator
settings = {'weather': 'typical'}
obsgen = og.obs_generator(settings)

# generate the observation
obs = obsgen.make_obs(im,addnoise=False,addgains=False)

# export SYMBA-compatible input files
master_input_args = {'input_fitsimage': imfilename}
obsgen.export_SYMBA(master_input_args=master_input_args)
