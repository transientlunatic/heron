Making new models
+++++++++++++++++

Heron has been designed to make it as easy as possible to create new models using new training data.

.. note:: Work in progress

	  Some of the flexibility we'd like to give heron in the long run is still under development, so things aren't always as smooth as we're planning for in the final release of the software.
	  In the meantime this guide should be fairly up-to-date.


Creating Training Data
======================

The first step towards creating a new model is selecting appropriate training data.
In this example we'll use waveforms generated from the ``IMRPhenomPv2`` waveform approximant.
While this might feel a bit odd, given that we're training an approximant with another approximant, it's a source where we have full control over the placement of waveforms, and we can choose the spacing of the training data however we wish.

Heron uses an ``hdf5``-based format to store training data for models, but also supplies code which allows these to be manipulated with relative ease.
Let's start by creating a file to store the data in.

.. code:: python

	  from heron.data import DataWrapper

	  data = DataWrapper.create("test_file.h5")


Now that we have our data file, we can start adding training data.
There are currently two ways to do this in ``heron``: we can either add many waveforms all at the same time, or add them one at a time.
It's probably easiest initially to add these one at a time.
We can simply do this in a for loop which generates a waveform and adds it to our training file.
The training data files can hold a number of data sets for multiple models, so we'll need to decide on a name for this training set.
For simplicity let's go with ``IMR training``.

I'll use ``pycbc`` to generate the waveform, but you can use whichever waveform interface you want.
We'll also need to select a total mass at which the waveforms will be generated; ``heron`` records this as it can use this mass to rescale waveforms later to other total masses.

Let's create a set of waveforms with zero spin across a range of mass ratios between 0.1 and 1.0.
In order to keep the size of the data set small we'll also 

.. code:: python

	  from pycbc.waveform import get_td_waveform
	  qs = np.linspace(0.1, 1.0, 40)
	  apx = "IMRPhenomPv2"
	  M = 20

	  for q in qs:

	      m1 = M / (1+q)
	      m2 = M / (1+1/q)
	      assert ((m1 + m2) - M ) < 1e-4

	      hp, hc = get_td_waveform(approximant=apx,
					   mass1=m1,
					   mass2=m2,
					   spin1z=0,
					   delta_t=1.0/4096,
					   f_lower=20)

	      idx = (hp.sample_times > -0.05) & (hp.sample_times < 0.02)

	      data.add_waveform(group="IMR training",
			    polarisation="+",
			    reference_mass=M,
			    source=apx,
			    locations={"mass ratio": q},
			    times=hp.sample_times[idx],
			    data=hp[idx]
			   )

We can plot the data which we've just created using some tools built-in to the ``DataWrapper`` object.

.. code:: python

	  fig = data.plot_surface(label="IMR training", x="time", y="mass ratio", polarisation=b"+", decimation=1);

.. image:: images/tutorial-training-scatter.png
   :width: 800

We now have a simple training set which can be used to construct a waveform model.
The built-in models in ``heron`` will attempt to work out as much information from the training data as they can in order to create the approximant, so this is now a compartiviely straightforward process.

Creating a Waveform Approximator
================================

Now that we have the underlying waveform data we can use it to create a model which is capable of producing a new waveform.
In this example we'll use a Gaussian process to generate the waveform.
The code uses ``pytorch`` and ``CUDA`` to enable GPU-based calculations to imrpove the speed of training and waveform generation if you have access to one.

.. code:: python

	  from heron.models.torchbased import HeronCUDA, train
	  import numpy as np

	  model = HeronCUDA(datafile="test_file.h5",
			    datalabel="IMR training",
			    device="cuda",
			    )

Having constructed the model, we then need to train it.
Fortunately this process is (mostly) automatic, and we just need to use the ``train`` function which we've already imported from ``heron``.

.. code:: python

	  train(model, iterations=10000)

We can then draw waveforms from the trained model.

.. code:: python

	  preds = model.mean(times=np.linspace(-0.2, 1, 100), p={"mass ratio":0.4})
			    
Mixins
======

Heron comes supplied with a number of mixin classes which yoou can use to easily add certain features to a model.
For example, if you wish to add CUDA functionality to a model this can be done using the ``CUDAModel`` mixin.


``models.torchbased.CUDAModel``
-------------------------------

Provides required additional settings for a model to run on a GPU using CUDA.
