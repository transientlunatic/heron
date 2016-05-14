import matplotlib.pyplot as plt
def corner(data, labels):
    f, ax = plt.subplots(data.shape[1],data.shape[1],)
    
    for i in xrange(data.shape[1]):
        for j in xrange(data.shape[1]):   
            
            if not i==data.shape[1]-1:
                ax[i,j].set_xticklabels([])
            else:
                ax[i,j].set_xlabel(labels[j])
            if not j==0: 
                ax[i,j].set_yticklabels([])
            else:
                ax[i,j].set_ylabel(labels[-i])
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
                ax[i,j].spines['top'].set_visible(False)
                ax[i,j].spines['right'].set_visible(False)
                ax[i,j].grid(False)
                ax[i,j].yaxis.set_ticks_position('left')
                ax[i,j].xaxis.set_ticks_position('bottom')
                n, bins, rectangles = ax[i,j].hist(data[:,i], normed=True,alpha=0.6)
                try:
                    kernel = gaussian_kde(data[:,i].T)
                    positions = np.linspace(data[:,i][0], data[:,i][-1], 100)
                    ax[i,j].plot(positions, n.max()*kernel(positions)/kernel(positions).max())
                    ax[i,j].set_xlim([positions.min(), positions.max()])
                except:
                    pass
                
                
                continue
            #ax[i,j].set_xlim([data[:,i].min(), data[:,i].max()])
            #ax[i,j].set_ylim([data[:,j].min(), data[:,j].max()])   
            ax[i,j].plot(data[:,j], data[:,i], '.')
            #ax[i,j].hist2d(targets[:,i], targets[:,j], bins=200, cmap='Blues')
    
    f.subplots_adjust(wspace=0.05, hspace=0.05)
