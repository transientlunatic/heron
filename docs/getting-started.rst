Getting Started with Heron
==========================

This guide is intended to give you a quick overview of the features of the ``heron`` package and how you can use them to generate waveforms and perform basic inference on data using them.

We'll present the steps in setting up a basic inference analysis here, and then point towards the documentation which explains each step in more detail.

#. **Import heron**:
   In order to get started we'll need to import the ``heron`` package and a waveform model.
   For this guide we'll use the CUDA-backed Gaussian process model, ``HeronCUDA``.

   .. code-block:: python

		   import heron
		   import torch
		   from heron.models.torchbased import HeronCUDA

   This will give us access to the various features of this model.

#. **Initialise the model**:
   We need to initialise the waveform model with its training data and pre-trained settings.
   For this guide we'll use a fast and simple training set, which is called ``IMR training linear``, which was generated using non-spinning waveforms drawn from the IMRPhenom family of approximant waveforms.
   We'll set things to run on a CUDA device (you'll need to change this to ``cpu`` if the machine you're running on doesn't have a GPU).

   .. code-block:: python

		   generator = HeronCUDA(
		       datalabel="IMR training linear",
		       name="Heron IMR Non-spinning",
		       device=torch.device("cuda"),
		   )

#. **Draw a waveform**

   Now that we've set up the waveform generator we can start to use it to generate waveforms for us.
