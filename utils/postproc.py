###################################################
# imports

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import ehtim as eh

from . import const_def as const

###################################################
# post-processing functions

def read_metric_stats(filename):
    """
    Read in a distilled table of metric statistics
    
    filename : path to table file

    """
    
    sitelist = np.loadtxt(filename,usecols=(0),skiprows=2,unpack=True,dtype='str')
    met_mean, met_std, met_pc10, met_pc25, met_pc50, met_pc75, met_pc90 = np.loadtxt(filename,usecols=(1,2,3,4,5,6,7),skiprows=2,unpack=True)
    
    return sitelist, met_mean, met_std, met_pc10, met_pc25, met_pc50, met_pc75, met_pc90

def toplist(sitelist,metric,percentile,metric_name):
    """
    Output a list of the top arrays according to a particular metric
    
    sitelist : list of site combinations
    metric : list of metric values
    percentile : what percentile constitutes "top" ?
    metric_name : name of metric, for labeling
    
    """

    index = np.argsort(metric)[::-1]
    if ((metric_name.lower() == 'pss') | (metric_name.lower() == 'ar')):
        index = index[::-1]

    sitelist_sorted = sitelist[index]

    top_index = int(np.round((percentile/100.0)*len(index)))

    return sitelist_sorted[:top_index]

def trackplot(outdir,sitelist,metric,metric_lo,metric_hi,metric_name,thin=1):
    """
    Produce a trackplot for a particular metric
    
    outdir : path at which to save output plot
    sitelist : list of site combinations
    metric : list of metric values
    metric_lo : list of metric lower bounds
    metric_hi : list of metric upper bounds
    metric_name : name of metric, for labeling
    thin : factor by which to thin plotted polygons

    """

    indsort = np.argsort(metric)
    if ((metric_name.lower() == 'pss') | (metric_name.lower() == 'ar')):
        indsort = indsort[::-1]
    
    metric_sorted = metric[indsort]
    metric_lo_sorted = metric_lo[indsort]
    metric_hi_sorted = metric_hi[indsort]
    sitelist_sorted = sitelist[indsort]

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])

    x = np.arange(0.0,len(metric))
    xpoly = np.concatenate((x,x[::-1]))
    ypoly = np.concatenate((metric_hi_sorted,metric_lo_sorted[::-1]))
    ax.fill(xpoly[::thin],ypoly[::thin],color='black',alpha=0.05)

    ax.plot(x[::thin],metric_sorted[::thin],'k-')

    ax.set_xlabel('Array index, sorted from worst to best')
    ax.set_ylabel(metric_name+' metric value')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xtext = (0.02*(xlim[1] - xlim[0])) + xlim[0]
    ytext = (0.98*(ylim[1] - ylim[0])) + ylim[0]
    textstr = 'Top 5 new site combinations:'+'\n'
    textstr += '1. ' + sitelist_sorted[-1] + '\n'
    textstr += '2. ' + sitelist_sorted[-2] + '\n'
    textstr += '3. ' + sitelist_sorted[-3] + '\n'
    textstr += '4. ' + sitelist_sorted[-4] + '\n'
    textstr += '5. ' + sitelist_sorted[-5]
    ax.text(xtext,ytext,textstr,ha='left',va='top',fontsize=6)

    plt.savefig(outdir+'/trackplot_' + metric_name + '.png',dpi=300,bbox_inches='tight')
    plt.close()

def multitrackplot(topdir,outdir,multi,multidim,metrics,metric_names,thin=1):
    """
    Produce a multitrackplot for a particular metric and parameter dimension
    
    topdir : top of the path containing distilled metric tables
    outdir : path at which to save output plot
    multi : the dimension along which to plot multiple trackplots
    multidim : list of the dimension values
    sitelist : list of site combinations
    metrics : list of metrics to plot
    metric_names : list of metric names, for labeling
    thin : factor by which to thin plotted polygons

    """

    for i, met in enumerate(metrics):

        metric_name = metric_names[i]

        fig = plt.figure(figsize=(4,4))
        ax = fig.add_axes([0.1,0.1,0.8,0.8])

        for imult, mult in enumerate(multidim):
            
            # construct input directory name
            dirname = topdir

            if (multi == 'source'):
                dirname += '/' + mult
            else:
                dirname += '/' + outdir.split('/')[-1].split('_')[0]

            if (multi == 'freq'):
                dirname += '/freq=' + mult
            else:
                dirname += '/freq=' + outdir.split('/')[-1].split('_')[1]

            if (multi == 'inst'):
                dirname += '/' + mult
            else:
                dirname += '/' + outdir.split('/')[-1].split('_')[2]

            if (multi == 'D'):
                dirname += '/D=' + mult
            else:
                dirname += '/D=' + outdir.split('/')[-1].split('_')[3]

            if (multi == 'bw'):
                dirname += '/bw=' + mult
            else:
                dirname += '/bw=' + outdir.split('/')[-1].split('_')[4]

            if (multi == 'month'):
                dirname += '/month=' + mult
            else:
                dirname += '/month=' + outdir.split('/')[-1].split('_')[5]

            if (multi == 'base'):
                dirname += '/base=' + mult
            else:
                dirname += '/base=' + outdir.split('/')[-1].split('_')[6]

            filename = dirname + '/postproc_' + met + '.txt'
            sitelist, met_mean, met_std, met_pc10, met_pc25, met_pc50, met_pc75, met_pc90 = read_metric_stats(filename)
            metric = met_pc50
            metric_lo = met_pc10
            metric_hi = met_pc90

            indsort = np.argsort(metric)
            if ((metric_name.lower() == 'pss') | (metric_name.lower() == 'ar')):
                indsort = indsort[::-1]

            metric_sorted = metric[indsort]
            metric_lo_sorted = metric_lo[indsort]
            metric_hi_sorted = metric_hi[indsort]
            sitelist_sorted = sitelist[indsort]

            x = np.arange(0.0,len(metric))
            xpoly = np.concatenate((x,x[::-1]))
            ypoly = np.concatenate((metric_hi_sorted,metric_lo_sorted[::-1]))
            ax.fill(xpoly[::thin],ypoly[::thin],color='C'+str(imult),alpha=0.2)

            ax.plot(x[::thin],metric_sorted[::thin],linestyle='-',color='C'+str(imult),label=multi+' = '+mult)

        ax.set_xlabel('Array index, sorted from worst to best')
        ax.set_ylabel(metric_name+' metric value')

        ax.legend(loc='upper left')

        plt.savefig(outdir+'/multitrackplot_' + metric_name +'_multi-' + multi + '.png',dpi=300,bbox_inches='tight')
        plt.close()

def spikeplot(outdir,sitelist,metric,metric_name):
    """
    Produce a spikeplot for a particular metric
    
    outdir : path at which to save output plot
    sitelist : list of site combinations
    metric : list of metric values
    metric_name : name of metric, for labeling
    
    """

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_axes([0.05,0.05,0.9,0.9])

    # identify individual sites
    sitestring = '-'.join(sitelist)
    sites = np.unique(sitestring.split('-'))

    # sort by metric value
    indsort = np.argsort(metric)
    if ((metric_name.lower() == 'pss') | (metric_name.lower() == 'ar')):
        indsort = indsort[::-1]
    metric_sorted = metric[indsort]
    sitelist_sorted = sitelist[indsort]

    # determine the fraction of time that each site makes it into the top 1%
    percentile_values_top1pc = np.zeros(len(sites))
    ind = int(np.floor(0.99*len(sitelist_sorted)))
    sitelist_sorted_top1pc = np.copy(sitelist_sorted[ind:])
    for isite, site in enumerate(sites):
        for i in range(len(sitelist_sorted_top1pc)):
            if (site in sitelist_sorted_top1pc[i].split('-')):
                percentile_values_top1pc[isite] += 1.0
    percentile_values_top1pc /= float(len(sitelist_sorted_top1pc))

    # determine the fraction of time that each site makes it into the top 5%
    percentile_values_top5pc = np.zeros(len(sites))
    ind = int(np.floor(0.95*len(sitelist_sorted)))
    sitelist_sorted_top5pc = np.copy(sitelist_sorted[ind:])
    for isite, site in enumerate(sites):
        for i in range(len(sitelist_sorted_top5pc)):
            if (site in sitelist_sorted_top5pc[i].split('-')):
                percentile_values_top5pc[isite] += 1.0
    percentile_values_top5pc /= float(len(sitelist_sorted_top5pc))

    # determine the fraction of time that each site makes it into the top 10%
    percentile_values_top10pc = np.zeros(len(sites))
    ind = int(np.floor(0.90*len(sitelist_sorted)))
    sitelist_sorted_top10pc = np.copy(sitelist_sorted[ind:])
    for isite, site in enumerate(sites):
        for i in range(len(sitelist_sorted_top10pc)):
            if (site in sitelist_sorted_top10pc[i].split('-')):
                percentile_values_top10pc[isite] += 1.0
    percentile_values_top10pc /= float(len(sitelist_sorted_top10pc))

    # determine the fraction of time that each site makes it into the top 50%
    percentile_values_top50pc = np.zeros(len(sites))
    ind = int(np.floor(0.50*len(sitelist_sorted)))
    sitelist_sorted_top50pc = np.copy(sitelist_sorted[ind:])
    for isite, site in enumerate(sites):
        for i in range(len(sitelist_sorted_top50pc)):
            if (site in sitelist_sorted_top50pc[i].split('-')):
                percentile_values_top50pc[isite] += 1.0
    percentile_values_top50pc /= float(len(sitelist_sorted_top50pc))

    # plot percentages
    x = np.arange(0,len(sites),1)
    ax.plot(x,percentile_values_top1pc,color='C0',linestyle='-',alpha=1,label='top 1%',zorder=-2)
    ax.plot(x,percentile_values_top5pc,color='C1',linestyle='-',alpha=0.8,label='top 5%',zorder=-3)
    ax.plot(x,percentile_values_top10pc,color='C2',linestyle='-',alpha=0.6,label='top 10%',zorder=-4)
    ax.plot(x,percentile_values_top50pc,color='C3',linestyle='-',alpha=0.4,label='top 50%',zorder=-5)

    ax.set_ylabel(metric_name+' metric fraction')
    ax.set_ylim(0,1)

    ax.set_xticks(x)
    ax.set_xticklabels(sites,rotation=90)

    ax.legend(loc='upper right')

    # identify the best 10 sites, according to the top 1%
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xtext = (0.01*(xlim[1] - xlim[0])) + xlim[0]
    ytext = (0.98*(ylim[1] - ylim[0])) + ylim[0]
    indsort2 = np.argsort(percentile_values_top1pc)[::-1]
    sites_sort2 = sites[indsort2]
    sites_best10 = list(np.sort(sites_sort2[0:10]))
    textstr = 'Top 10 sites:' + '\n' + ', '.join(sites_best10)
    ax.text(xtext,ytext,textstr,ha='left',va='top')

    plt.savefig(outdir + '/spikeplot_' + metric_name + '.png',bbox_inches='tight',dpi=300)
    plt.close()

    return sites, sites_best10

def cornerplot(outdir,dirname,lowercase,diag_type='histogram',scatter=False,hist2d=True,nbins2d=20):
    """
    Produce a cornerplot for comparing several metrics
    
    outdir : path at which to save output plot
    dirname : path containing tables of metric stats
    lowercase : list of metric names, in lowercase
    diag_type : style of plot for diagonal panels; can be 'histogram' or 'trackplot'
    scatter : flag to toggle whether to show scatterplots
    hist2d : flag to toggle whether to show 2D histograms
    nbins2d : number of bins to use for 2D histograms
    
    """

    fig = plt.figure(figsize=(8,8))

    xstart = 0.05
    xend = 0.99
    xsep = 0.01
    xwidth = ((xend - xstart) - ((len(lowercase)-1.0)*xsep))/len(lowercase)

    ystart = xstart
    ysep = xsep
    ywidth = xwidth

    axlist = list()
    for ix, x in enumerate(lowercase):
        for iy, y in enumerate(lowercase):
            el1 = xstart + ix*(xsep+xwidth)
            el2 = ystart + iy*(ysep+ywidth)
            el3 = xwidth
            el4 = ywidth
            if (iy < (len(lowercase)-ix)):
                axlist.append(fig.add_axes([el1,el2,el3,el4]))

    count = 0
    for ix in range(len(lowercase)):
        for iy in range(len(lowercase)):

            if (iy < (len(lowercase)-ix)):

                xind = ix
                yind = len(lowercase)-iy-1

                # read in horizontal metric
                filename = dirname + '/postproc_' + lowercase[xind] + '.txt'
                sitelist, met_mean, met_std, met_pc10, met_pc25, met_pc50, met_pc75, met_pc90 = read_metric_stats(filename)
                x = met_pc50
                xmin = np.min(x)
                xmax = np.max(x)
                xlim = (xmin - (0.1*(xmax - xmin)), xmax + (0.1*(xmax - xmin)))

                # read in vertical metric
                filename = dirname + '/postproc_' + lowercase[yind] + '.txt'
                sitelist, met_mean, met_std, met_pc10, met_pc25, met_pc50, met_pc75, met_pc90 = read_metric_stats(filename)
                y = met_pc50
                ymin = np.min(y)
                ymax = np.max(y)
                ylim = (ymin - (0.1*(ymax - ymin)), ymax + (0.1*(ymax - ymin)))

                # plot scatterplots on the off-diagonals
                if (iy != (len(lowercase)-ix-1)):

                    # plot scatterplot
                    if scatter:
                        axlist[count].plot(x,y,marker='o',color='black',markersize=2,markeredgewidth=0,alpha=0.3,linewidth=0)

                    # plot 2D histogram
                    if hist2d:
                        H, xedges, yedges = np.histogram2d(x,y,bins=nbins2d,range=[[xlim[0],xlim[1]],[ylim[0],ylim[1]]])
                        axlist[count].pcolormesh(xedges.T,yedges.T,np.log10(H).T,cmap='Greys',vmin=0.0)

                    # set axis limits
                    axlist[count].set_xlim(xlim)
                    axlist[count].set_ylim(ylim)

                    # make a grid
                    axlist[count].grid(color='black',linestyle='--',linewidth=0.5,alpha=0.2)

                    # set axis labels and tickmarks
                    if (yind == (len(lowercase)-1)):
                        axlist[count].set_xlabel(lowercase[xind])
                        ticks_loc = axlist[count].get_xticks().tolist()
                        axlist[count].xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                        axlist[count].set_xticklabels(np.round(axlist[count].get_xticks(),8), rotation=60)
                    else:
                        axlist[count].set_xticklabels([])
                        axlist[count].tick_params(axis='x', colors='white')

                    if (xind == 0):
                        axlist[count].set_ylabel(lowercase[yind])
                        ticks_loc = axlist[count].get_yticks().tolist()
                        axlist[count].yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                        axlist[count].set_yticklabels(np.round(axlist[count].get_yticks(),8), rotation=0)
                    else:
                        axlist[count].set_yticklabels([])
                        axlist[count].tick_params(axis='y', colors='white')

                # plot something else on the diagonals
                if (iy == (len(lowercase)-ix-1)):

                    if (diag_type == 'histogram'):
                        axlist[count].hist(x,bins=20,range=xlim,color='black',linewidth=1,density=True,histtype='step')

                    elif (diag_type == 'trackplot'):
                        x_sorted = np.sort(x)
                        y_sorted = np.arange(0.0,len(x_sorted))
                        axlist[count].plot(x_sorted,y_sorted,color='black',linewidth=1)

                    # set axis limits
                    axlist[count].set_xlim(xlim)

                    # set axis labels
                    if (iy != 0):

                        axlist[count].set_xticklabels([])
                        axlist[count].tick_params(axis='x', colors='white')

                        axlist[count].set_yticklabels([])
                        axlist[count].tick_params(axis='y', colors='white')

                    else:
                        axlist[count].set_xlabel(lowercase[xind])
                        ticks_loc = axlist[count].get_xticks().tolist()
                        axlist[count].xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                        axlist[count].set_xticklabels(np.round(axlist[count].get_xticks(),8), rotation=60)

                        axlist[count].set_yticklabels([])
                        axlist[count].tick_params(axis='y', colors='white')

                # increment count
                count += 1

    plt.savefig(outdir+'/cornerplot_'+diag_type+'.png',dpi=300,bbox_inches='tight')
    plt.close()

def siteplot(outdir,sites,metric_name,psets,bestXarr):
    """
    Produce a siteplot showing the top X sites across a set of parameter combinations
    
    outdir : path at which to save output plot
    sites : list of all sites to consider
    metric_name : name of metric, for labeling
    psets : list of parameter sets
    bestXarr : array containing the best 10 sites for each pset

    """

    fig = plt.figure(figsize=(4,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])

    # plot the best 10 sites for each parameter set
    counts = np.zeros(len(sites))
    for row in psets.itertuples():
        bestX = bestXarr[row.Index].split('-')
        for isite, site in enumerate(sites):
            if (site in bestX):
                ax.plot([row.Index],[isite],'ks',markersize=3)
                counts[isite] += 1.0

    # highlight the best 10 sites overall
    ind = np.argsort(counts)[::-1]
    xlim = ax.get_xlim()
    for j in range(10):
        yhere = ind[j]
        y1 = (yhere - 0.4)*np.array([1.0,1.0])
        y2 = (yhere + 0.4)*np.array([1.0,1.0])
        ax.fill_between(np.array(xlim),y1,y2,color='green',alpha=0.2)
    ax.set_xlim(xlim)

    ax.set_xlabel('Parameter set index')

    ax.set_yticks(np.arange(0,len(sites),1))
    ax.set_yticklabels(sites)

    ax.set_ylim(-1,len(sites))
    ylim = ax.get_ylim()

    plt.savefig(outdir+'/siteplot_' + metric_name + '.png',bbox_inches='tight',dpi=300)
    plt.close()

def overlapplot(outdir,metric_name,top_arr,psets,extraname=''):
    """
    Produce a plot showing the percentage of top-array overlaps across a set of parameter combinations
    
    outdir : path at which to save output plot
    metric_name : name of metric, for labeling
    top_arr : array containing the top site sets for each parameter set
    psets : list of parameter sets
    extraname : additional name piece in output filename

    """

    fig = plt.figure(figsize=(16,16))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])

    xvals = list()
    yvals = list()
    colvals = list()
    textvals = list()
    for ix in range(len(psets)):
        top_x = top_arr[ix]
        list_as_set = set(top_x)
        for iy in range(len(psets)):
            top1pc_y = top_arr[iy]
            overlap_count = len(list_as_set.intersection(top1pc_y))
            colval = overlap_count/len(top_x)
            xvals.append(ix+0.5)
            yvals.append((len(psets)-iy)-0.5)
            colvals.append(colval)
            textvals.append(str(int(np.round(colval*100.0))))
    ax.scatter(xvals,yvals,c=colvals,s=360*((48.0/len(psets))**2.0),marker='s',cmap='Greens',vmin=0.0,vmax=1.0)

    for j in range(len(textvals)):
        if (colvals[j] < 0.75):
            ax.text(xvals[j]+0.45,yvals[j]-0.45,textvals[j],color='black',ha='right',va='bottom',fontsize=6)
        else:
            ax.text(xvals[j]+0.45,yvals[j]-0.45,textvals[j],color='white',ha='right',va='bottom',fontsize=6)

    for ix in range(len(psets)):
        ax.plot([ix,ix],[0,len(psets)],'k-',linewidth=0.5)
        ax.plot([0,len(psets)],[ix,ix],'k-',linewidth=0.5)
    ax.plot([ix+1,ix+1],[0,len(psets)],'k-',linewidth=0.5)
    ax.plot([0,len(psets)],[ix+1,ix+1],'k-',linewidth=0.5)

    xticks = np.linspace(0.5,len(psets)-0.5,len(psets))
    yticks = np.linspace(0.5,len(psets)-0.5,len(psets))
    labels = list()
    for row in psets.itertuples():
        strhere = ''
        strhere += row.source + ', '
        strhere += row.freq + ', '
        strhere += row.instantiation + ', '
        strhere += row.D + ', '
        strhere += row.bw + ', '
        strhere += row.month + ', '
        strhere += row.base_array_name
        labels.append(strhere)

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels,rotation=90)
    ax.xaxis.set_ticks_position('top')

    ax.set_yticks(yticks)
    ax.set_yticklabels(labels[::-1])

    ax.set_xlim(0,len(psets))
    ax.set_ylim(0,len(psets))

    plt.savefig(outdir + '/overlapplot_' + extraname + '_' + metric_name + '.png',bbox_inches='tight',dpi=300)
    plt.close()

