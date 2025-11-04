Exercise G
==========

Nonlinear problems.

.. rubric:: Available scripts

* ``compute_profiling.py`` – profiles RK3 and RK4 on a single-soliton test, producing
  timing summaries and ``cProfile`` dumps in ``data/A2/ex_g``.
* ``compute_work-precision.py`` – sweeps stable time-step fractions for RK3/RK4 and
  records work vs accuracy metrics in ``data/A2/ex_g``.
* ``plot_profiling.py`` / ``plot_work-precision.py`` – create the figures stored in
  ``figures/A2/ex_g``. Run the matching ``compute_*.py`` scripts first.
