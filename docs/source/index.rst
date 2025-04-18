
Welcome to ngehtsim's documentation!
====================================

ngehtsim is a set of Python_ tools for generating synthetic data appropriate for Very Long Baseline Interferometry (VLBI) facilities operating at (sub)millimeter wavelengths, such as the Event Horizon Telescope (EHT_), the next-generation Event Horizon Telescope (ngEHT_), and the Black Hole Explorer (BHEX_).  ngehtsim builds on synthetic data generation capabilities contained in the ehtim_ library, primarily by adding utilities for incorporating local weather effects, telescope properties, and fringe-finding proxies to the data generation procedure.  Atmospheric state data for ngehtsim comes from the `MERRA-2`_ database, processed through the am_ radiative transfer code to produce optical depth and brightness temperature information.

.. _Python: https://www.python.org/
.. _EHT: https://eventhorizontelescope.org/
.. _ngEHT: https://www.ngeht.org/
.. _ehtim: https://achael.github.io/eht-imaging/
.. _BHEX: https://www.blackholeexplorer.org/
.. _MERRA-2: https://gmao.gsfc.nasa.gov/reanalysis/merra-2/
.. _am: https://zenodo.org/records/13748391

Contents:
====================================

.. toctree::
    :maxdepth: 1

    ./installation
    ./introduction
    ./tutorials
    ./api

Indices and tables
====================================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Referencing
====================================

The ngehtsim codebase has been described and demonstrated in `Pesce et al. (2024)`_.  Those looking to use ngehtsim in their own work should please cite:

Pesce, D. W. et al. (2024) *Atmospheric Limitations for High-frequency Ground-based Very Long Baseline Interferometry*, ApJ, 968, 69, DOI: `10.3847/1538-4357/ad3961 <https://iopscience.iop.org/article/10.3847/1538-4357/ad3961>`_

.. _Pesce et al. (2024): https://iopscience.iop.org/article/10.3847/1538-4357/ad3961
