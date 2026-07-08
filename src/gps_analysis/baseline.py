"""Reference-offset and windowing utilities for GNSS coordinate time series.

Planned surface (plan §10.2):

- ``estimate_offset`` / ``remove_offset`` — reference-epoch offset handling
  (Phase 1). Ancestors: ``find_exp_offsets`` / ``find_line_offsets`` in
  ``detrend_rnes.py`` (fit adjacent periods independently, difference at
  the midpoint) and the ``vshift`` usage pattern from gtimes.
- ``slice_window`` — time-window extraction shared by fitting, velocity and
  the precompute job (Phase 1); replaces the ad-hoc ``dPeriod`` slicing of
  ``(yearf, data, Ddata)`` triples.

Naming caveat (flagged in ``docs/CONSOLIDATION_MAP.md``): "baseline" in the
wider workspace also means RTK rover–base vectors (``pygmt/functions.py``
readers) and inter-station vectors. *This* module is only the
reference-offset/window meaning; if RTK-baseline math ever lands in this
package it gets its own, unambiguous module name.

Pure array utilities; no I/O, no unit assumptions.
"""

__all__: list[str] = []
