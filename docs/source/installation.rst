Installation Guide
========================

Getting ngehtsim
------------------------

The latest version of ngehtsim can be obtained from https://github.com/Smithsonian/ngehtsim.

Installation
------------------------

To install ngehtsim, run pip from the main directory:

.. code-block:: console

   $ pip install [--upgrade] . [--user]

If you would like to additionally install the calibration dependencies, run:

.. code-block:: console

   $ pip install [--upgrade] .[calib] [--user]

Dependencies
------------------------

ngehtsim uses the following packages:

* `python <https://www.python.org/downloads>`_ >=3.8
* `ehtim <https://github.com/achael/eht-imaging>`_
* `numpy <https://numpy.org>`_
* `matplotlib <https://matplotlib.org>`_
* `scipy <https://www.scipy.org>`_
* `astropy <https://www.astropy.org/>`_
* `ngEHTforecast <https://aeb.github.io/ngEHTforecast/html/docs/src/index.html>`_

ngehtsim also has the option to include additional calibration capabilities during installation, which further requires:

* `pandas <https://pandas.pydata.org/>`_
* `eat <https://github.com/sao-eht/eat.git>`_

Dependencies are specified in setup.py and will be handled via the pip install process. Generating a local version of the documentation for ngehtsim requires:

* `Sphinx <https://www.sphinx-doc.org>`_