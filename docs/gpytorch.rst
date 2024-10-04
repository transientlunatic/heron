=====================
GPyTorch-based models
=====================

A number of models implemented in ``heron`` make use of ``pytorch`` and the GPR library built atop it, ``gpytorch``.
These models can be used on both CPU and GPU hardware.

All of these models are contained within the `heron.models.torchbased` module.


HeronCUDA : A spinning, NR-trained, GPU-capable surrogate model
---------------------------------------------------------------


+------------------+-----------------------+------------+----------+--------------+
| Training data    | GPR Technique         | Model type | Spinning | Higher modes |
+==================+=======================+============+==========+==============+
| NR: Georgia Tech | Exact, LOVE, CUDA     | BBH        | Fully    | No           |
+------------------+-----------------------+------------+----------+--------------+

The model is trained on `numerical relativity waveforms <http://www.einstein.gatech.edu/catalog/>`_ produced by the Centre for Relativistic Astrophysics at Georgia Tech, and uses exact scalable GPR techniques implemented by `GPyTorch <https://gpytorch.readthedocs.io/>`_.

.. inheritance-diagram:: heron.models.torchbased.HeronCUDA

.. autoclass:: heron.models.torchbased.HeronCUDA
