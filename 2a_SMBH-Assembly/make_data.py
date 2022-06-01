##################################################
# imports etc.

import numpy as np
import ehtim as eh
import os
import utils.obs_generator as og
import utils.obs_plotter as op

##################################################
# inputs

ring_flux_grid = [10.0,30.0,100.0,300.0,1000.0]     # mJy
ring_size_grid = [2.5,5.0,10.0,20.0,40.0]           # uas

RA = 12.5137287172
DEC = 12.3911232392
MJD = 51544

# input settings file
yamlfile = '2a_SMBH-Assembly/settings.yaml'

##################################################
# create synthetic data

# initialize the observation generator
obsgen = og.obs_generator(yamlfile)

for ring_flux in ring_flux_grid:
    for ring_size in ring_size_grid:

        # make output directory
        outdir = '2a_SMBH-Assembly/data/flux='+str(int(ring_flux))+'mJy/size='+str(int(ring_size))+'uas'
        os.makedirs(outdir,exist_ok=True)

        ##################################################
        # instantiate the model object

        F0 = ring_flux*0.001
        d = ring_size*eh.RADPERUAS
        alpha = d / 3.0
        stretch = 1.0
        stretch_PA = 0.0
        beta_list = np.random.uniform(-0.5,0.5,size=1) + (1.0j)*np.random.uniform(-0.5,0.5,size=1)

        bpol0 = np.random.uniform(-0.3,0.3) + (1.0j)*np.random.uniform(-0.3,0.3)
        bpoln1 = np.random.uniform(-0.1,0.1) + (1.0j)*np.random.uniform(-0.1,0.1)
        bpol1 = np.random.uniform(-0.1,0.1) + (1.0j)*np.random.uniform(-0.1,0.1)
        bpoln2 = np.random.uniform(-0.1,0.1) + (1.0j)*np.random.uniform(-0.1,0.1)
        bpol2 = np.random.uniform(-1.0,1.0) + (1.0j)*np.random.uniform(-1.0,1.0)
        beta_list_pol = [bpoln2,bpoln2,bpol0,bpol1,bpol2]

        mod = eh.model.Model()
        mod = mod.add_stretched_thick_mring(F0=F0,
                                            d=d,
                                            alpha=alpha,
                                            x0=0.0,
                                            y0=0.0,
                                            beta_list=beta_list,
                                            beta_list_pol=beta_list_pol,
                                            stretch=stretch,
                                            stretch_PA=stretch_PA)

        mod.ra = RA
        mod.dec = DEC
        mod.mjd = MJD

        # save the model
        mod.save_txt(outdir+'/input_model.txt')

        # save an image of the model
        mod.display(plotp=True,show=False,export_pdf=outdir+'/input_model.png')
        im = mod.make_image(200.0*eh.RADPERUAS,1024)
        path_to_im = outdir+'/input_model.fits'
        im.save_fits(path_to_im)

        ##################################################
        # generate observation

        # point to the right image file and generate the observation
        obsgen.settings['model_file'] = path_to_im
        obs = obsgen.make_obs()

        # save the uvfits file
        obs.save_uvfits(outdir+'/synthetic_data.uvfits')

        # save some plots
        op.plot_uv(obs,filename=outdir+'/coverage.png')
        op.plot_amp(obs,filename=outdir+'/radplot.png')
        
