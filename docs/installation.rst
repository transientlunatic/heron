.. highlight:: shell

============
Installation
============


Stable release
--------------

To install the latest stable release of Heron you can use ``pip``:

.. code-block:: console

   $ pip install heron

This should always install the latest stable release of heron, though it may still be sensible to run this command inside a virtual environment.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


Source installation
-------------------

Alternatively, if you want to make your own changes to the code, or test code between releases you can install from source.
The Heron source can be downloads from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/transientlunatic/heron

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/transientlunatic/heron/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install .


.. _Github repo: https://github.com/transientlunatic/heron
.. _tarball: https://github.com/transientlunatic/heron/tarball/master
