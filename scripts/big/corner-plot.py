import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np
import pickle


import gwgpr
import gwgpr.nr as nr

cat = nr.NRCatalogue('/scratch/aries/gt_bbh/')
cat_f = nr.NRCatalogue('/scratch/aries/gt_bbh/')
#cat.waveforms = cat.waveforms[cat.waveforms['series']=='S-series-v2']
cols = ['q', 'a1','a2', 'th1L', 'ph1', 'th12', 'thSL', 'thJL']


plt.style.use("/home/daniel/thesis/thesis-style.mpl")
figsize = (10,10)

SKIP = 5

training_x = []
training_y = []
for waveform in cat.waveforms[cols].iterrows():
    try:
        wave = cat.load(waveform[0])
        times = wave.times
        hp = wave.data[0]
        hc = wave.data[1]
        # Select a limited time span from -200 to 100
        #locs = times== 0 
        #locs = np.roll(locs, 100)
        locs = [(times>-50) & (times<10)]
        N = len(times[locs][::SKIP])
        data = hp[locs][::SKIP]
        params = [list(waveform[1])]*N
        times = times[locs]
        if np.isnan(params).any():
            print "{} contains nan".format(waveform[0])
            continue
        for i in xrange(N):
            out = []
            out.append(times[i])
            out.extend(params[i])
            training_x.append(out)
        training_y.extend(list(data))
        #training.append( times[locs][::4], params, data) 
        #print zip(cols, list(waveform[1]))
    except:
        print "{} unvailable".format(waveform[0])





gp, training_y = pickle.load( open( "full.gp", "rb" ) )

resolution = 100
cols_axis = {
    "times": np.linspace(-100, 100, resolution),
    "q": np.linspace(0, 10, resolution),
    "a1" : np.linspace(0, 1, resolution),
    "a2": np.linspace(0, 1, resolution),
    "th1L": np.linspace(0, 180, resolution),
    "ph1": np.linspace(-180, 180, resolution),
    "th12": np.linspace(0, 180, resolution),
    "thSL": np.linspace(0, 180, resolution),
    "thJL": np.linspace(0, 45, resolution)
}
cols = ['times', 'q', 'a1', 'a2', 'th1L', 'ph1', 'th12', 'thSL', 'thJL'] 

def gen2plane(col1, col2, intersept = [ 0,  1.5,    0.8,    0.8,   60. ,  180. ,   30. ,   75. ,   22. ], resolution = 100):
    
    pdata = np.zeros((100,100))
    udata = np.zeros((100,100))
    
    col1_ax = cols_axis[col1]
    col2_ax = cols_axis[col2]
    #
    col1_loc = cols.index(col1)
    col2_loc = cols.index(col2)
    #
    xv, yv = np.meshgrid(col1_ax, col2_ax, sparse=False, indexing='xy')
    for i in xrange(100):
        for j in xrange(100):
            new_vec = np.copy(intersept)
            new_vec[col1_loc] = xv[i,j]
            new_vec[col2_loc] = yv[i,j]
            # Calculate the spin/mass surface for time = 0.00
            
            pdata[i][j], udata[i][j] = gp.predict(training_y, [new_vec])
    return pdata, udata, [col1_ax.min(), col1_ax.max(), col2_ax.min(), col2_ax.max()]

spacings = np.exp(gp.kernel.get_vector())**0.5
spacings = spacings[1:]
samp_cols_axis = {
    #"times"
    "q": np.arange(0+10%spacings[0]/2, 10, spacings[0]),
    "a1" : np.arange(0+1%spacings[1]/2, 1, spacings[1]),
    "a2": np.arange(0+1%spacings[2]/2, 1, spacings[2]),
    "th1L": np.arange(0+180%spacings[3]/2, 180, spacings[3]),
    "ph1": np.arange(-180+360%spacings[4]/2, 180, spacings[4]),
    "th12": np.arange(0+180%spacings[5]/2, 180, spacings[5]),
    "thSL": np.arange(0+180%spacings[6]/2, 180, spacings[6]),
    "thJL": np.arange(0+45%spacings[7]/2, 45, spacings[7])
}
def sample_grid(col1, col2):
    
    resolution = 100

    col1_ax = samp_cols_axis[col1]
    col2_ax = samp_cols_axis[col2]
    #
    col1_loc = cols.index(col1)
    col2_loc = cols.index(col2)
    #
    xv, yv = np.meshgrid(col1_ax, col2_ax, sparse=False, indexing='xy')
    return xv, yv
    

f, ax = plt.subplots(len(cols), len(cols), figsize = figsize)
for i in range(len(cols)):
    for j in range(len(cols)):
        print i,j
        if j<i: 
            ax[j,i].axis('off')
            continue
        elif i == j:
            ax[j,i].axis("off")
            wv = np.array(training_x)
            #pars = [   1.5,    0.8,    0.8,   60. ,  180. ,   30. ,   75. ,   22. ]
            pars = [  0,  1.5,    0.8,    0.8,   60. ,  180. ,   30. ,   75. ,   22. ]
            diffs = np.array(wv / wv.max()) - pars/np.array(wv.max())
            ax[j,i].hist2d(wv[:,i], np.sqrt((diffs**2).sum(axis=1)), bins=20, cmap='Greys');
            
        else:
            
            plt.setp(ax[j,i].get_xticklabels(), visible=False, rotation='vertical');
            plt.setp(ax[j,i].get_yticklabels(), visible=False, rotation='vertical');
            pdata, udata, extent = gen2plane(cols[i], cols[j])
            ax[j,i].imshow(udata, extent = extent, aspect = (extent[1] - extent[0]) / (extent[3] - extent[2]), origin='lower')
            ax[j,i].plot(pars[i], pars[j], 'o', c='red')
            if (cols[i] != "times") and (cols[j] != "times"):
                xv, yv = sample_grid(cols[i], cols[j])
                for a in xrange(xv.shape[0]):
                    for b in xrange(yv.shape[1]):
                        ax[j,i].plot(xv[a,b], yv[a,b], '+', c='white')
        plt.savefig("spacings.pdf")
for i,val in enumerate(cols):
    ax[-1,i].set_xlabel(val);
    plt.setp(ax[-1,i].get_xticklabels(), visible=True, rotation='vertical');
    ax[i, 0].set_ylabel(val);
    plt.setp(ax[i, 0].get_yticklabels(), visible=True)
plt.savefig("spacings.pdf")