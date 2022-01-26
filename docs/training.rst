Making new models
+++++++++++++++++


Mixins
======

Heron comes supplied with a number of mixin classes which yoou can use to easily add certain features to a model.
For example, if you wish to add CUDA functionality to a model this can be done using the ``CUDAModel`` mixin.


``models.torchbased.CUDAModel``
-------------------------------

Provides required additional settings for a model to run on a GPU using CUDA.
