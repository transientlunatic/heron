import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
import numpy as np
def corner(data_object, figsize=(10,10)):

    data = data_object.denormalise(data_object.targets, "target")
    colnames = data_object.target_names
    f, ax = plt.subplots(data.shape[1],data.shape[1],figsize=figsize)
    yvalues = data_object.denormalise(data_object.labels, "label")

    for i, xcol in enumerate(colnames):
        for j, ycol in enumerate(colnames):   

            if not i==len(colnames)-1:
                ax[i,j].set_xticklabels([])
            else:
                ax[i,j].set_xlabel(colnames[j])
            if not j==0: 
                ax[i,j].set_yticklabels([])
            else:
                ax[i,j].set_ylabel(colnames[i])
            if j>i: 
                ax[i,j].spines['top'].set_visible(False)
                ax[i,j].spines['bottom'].set_visible(False)
                ax[i,j].spines['left'].set_visible(False)
                ax[i,j].spines['right'].set_visible(False)
                ax[i,j].grid(False)
                ax[i,j].yaxis.set_ticks_position('none')
                ax[i,j].xaxis.set_ticks_position('none')
                continue


            if i == j:
                ax[i,j]
                ax[i,j].spines['top'].set_visible(False)
                ax[i,j].spines['right'].set_visible(False)
                ax[i,j].grid(False)
                ax[i,j].yaxis.set_ticks_position('left')
                ax[i,j].xaxis.set_ticks_position('bottom')
                n, bins, rectangles = ax[i,j].hist(data[:,i], normed=True,alpha=0.6)
                #try:
                #kernel = gaussian_kde(data[:,i].T)
                #positions = np.linspace(data[:,i].min(), data[:,i].max(), 100)
                #ax[i,j].plot(positions, n.max()*kernel(positions)/kernel(positions).max())


                continue
            #ax[i,j].set_xlim([data[:,i].min(), data[:,i].max()])
            #ax[i,j].set_ylim([data[:,j].min(), data[:,j].max()])   
            #ax[i,j].plot(t[ycol], t[xcol], '.')
            hexes = ax[i,j].hexbin(data[:,j], data[:,i], gridsize=15, cmap="Reds", bins='log', vmin=0, vmax=2)

    f.subplots_adjust(wspace=0.05, hspace=0.05)
    #cb = f.colorbar(ax[2,3], cax = ax[4,5]) 
    cbar = f.colorbar(hexes, ax = ax[len(colnames)-2,len(colnames)-1], orientation="vertical")
    cbar.set_label("Number density")   
    return f
