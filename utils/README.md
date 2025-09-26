# Advanced Numerical Methods Utilities

Shared plotting helpers used throughout the assignment scripts.

## Plotting helper usage

Call `setup_assignment_plotting("assignment_1/Plots/<relative-folder>")` near the top of a script to configure styling and ensure the specified directory exists (paths are resolved relative to the repo root). For example, `setup_assignment_plotting("assignment_1/Plots/FourierSpectralMethods/exercise_a")` places output next to the corresponding exercise.

Use `save_figure("figure_name", fig=fig)` to export figures; the helper enforces PDF output, applies a tight layout by default, and returns the saved path. Pass `tight=False` or `tight_layout_kwargs={...}` if you need to customise the layout call before saving. Combine with `style_axes(...)` to apply consistent titles, labels, grids, and legends without repetitive code.
