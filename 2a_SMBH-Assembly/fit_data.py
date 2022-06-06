##################################################
# imports etc.

import numpy as np
import matplotlib.pyplot as plt
import ehtim as eh
import os
import pickle
from dynesty import plotting as dyplot

##################################################
# inputs

ring_flux_grid = [1.0,3.0,10.0,30.0,100.0,300.0,1000.0]     # mJy
ring_size_grid = [2.5,5.0,10.0,20.0,40.0]           # uas

arraynames = ['EHT2017',
              'EHT2022',
              'ngEHT1_minimal',
              'ngEHT1_partial',
              'ngEHT1_full',
              'ngEHT2_minimal',
              'ngEHT2_partial',
              'ngEHT2_full']

freq = '230GHz'

nlive = 1000

##################################################
# fit the synthetic data

# for ring_size in ring_size_grid:
#     for iarr, arrayname in enumerate(arraynames[::-1]):
#         for ring_flux in ring_flux_grid:

for iarr, arrayname in enumerate(arraynames):
    for ring_flux in ring_flux_grid:
        for ring_size in ring_size_grid:
            
            indir = '2a_SMBH-Assembly/data_'+freq+'/array='+arrayname+'/flux='+str(int(ring_flux))+'mJy/size='+str(int(ring_size))+'uas'
            input_file = indir + '/synthetic_data.uvfits'

            if os.path.exists(input_file):
                try:

                    # load synthetic data
                    obs = eh.obsdata.load_uvfits(input_file)

                    # initialize model
                    mod_init  = eh.model.Model()
                    mod_init  = mod_init.add_thick_mring(1.0, 30.*eh.RADPERUAS, 15.*eh.RADPERUAS, beta_list=[0], beta_list_pol=[0,0,0,0,0])

                    # set priors
                    mod_prior = mod_init.default_prior(fit_pol=True)
                    mod_prior[0]['F0'] = {'prior_type':'fixed'}
                    mod_prior[0]['d']  = {'prior_type':'flat', 'min':0.0*eh.RADPERUAS, 'max':100.0*eh.RADPERUAS}
                    mod_prior[0]['alpha']  = {'prior_type':'flat', 'min':0.0*eh.RADPERUAS, 'max':50.0*eh.RADPERUAS}
                    mod_prior[0]['x0'] = {'prior_type':'fixed'}
                    mod_prior[0]['y0'] = {'prior_type':'fixed'}
                    mod_prior[0]['beta1_abs'] = {'prior_type':'flat', 'min':0.0, 'max':1.0}
                    mod_prior[0]['betapol-2_abs'] = {'prior_type':'flat', 'min':0.0, 'max':0.2}
                    mod_prior[0]['betapol-1_abs'] = {'prior_type':'flat', 'min':0.0, 'max':0.1}
                    mod_prior[0]['betapol0_abs'] = {'prior_type':'flat', 'min':0.0, 'max':0.2}
                    mod_prior[0]['betapol1_abs'] = {'prior_type':'flat', 'min':0.0, 'max':0.1}
                    mod_prior[0]['betapol2_abs'] = {'prior_type':'flat', 'min':0.0, 'max':0.5}

                    # make output directory
                    outdir = '2a_SMBH-Assembly/fits/array='+arrayname+'/flux='+str(int(ring_flux))+'mJy/size='+str(int(ring_size))+'uas'

                    if not os.path.exists(outdir+'/mod_fit.p'):

                        # fit the model to closures and polarimetric ratios
                        mod_fit = eh.modeler_func(obs, mod_init, mod_prior, d1='logcamp', d2='cphase', d3='m', fit_pol=True, minimizer_func='dynesty_static',minimizer_kwargs={'nlive':nlive})

                        # save outputs
                        os.makedirs(outdir,exist_ok=True)
                        mod_fit_out = mod_fit.copy()
                        del mod_fit_out['sampler']
                        pickle.dump(mod_fit_out,open(outdir+'/mod_fit.p','wb'),protocol=pickle.HIGHEST_PROTOCOL)

                        tfig, taxes = dyplot.traceplot(mod_fit['res_natural'])
                        plt.savefig(outdir+'/trace_plot.png',dpi=300,bbox_inches='tight')
                        plt.close()

                        mod_true = eh.model.load_txt(indir + '/input_model.txt')
                        truths = [mod_true.params[0]['d']/eh.RADPERUAS,
                                  mod_true.params[0]['alpha']/eh.RADPERUAS,
                                  np.abs(mod_true.params[0]['beta_list'][0]),
                                  np.angle(mod_true.params[0]['beta_list'][0])*(180.0/np.pi),
                                  np.abs(mod_true.params[0]['beta_list_pol'][0]),
                                  np.angle(mod_true.params[0]['beta_list'][0])*(180.0/np.pi),
                                  np.abs(mod_true.params[0]['beta_list_pol'][1]),
                                  np.angle(mod_true.params[0]['beta_list_pol'][1])*(180.0/np.pi),
                                  np.abs(mod_true.params[0]['beta_list_pol'][2]),
                                  np.angle(mod_true.params[0]['beta_list_pol'][2])*(180.0/np.pi),
                                  np.abs(mod_true.params[0]['beta_list_pol'][3]),
                                  np.angle(mod_true.params[0]['beta_list_pol'][3])*(180.0/np.pi),
                                  np.abs(mod_true.params[0]['beta_list_pol'][4]),
                                  np.angle(mod_true.params[0]['beta_list_pol'][4])*(180.0/np.pi)]

                        cfig, caxes = dyplot.cornerplot(mod_fit['res_natural'], labels=mod_fit['labels_natural'],truths=truths)
                        plt.savefig(outdir+'/cornerplot.png',bbox_inches='tight')
                        plt.close()

                except:
                    pass
