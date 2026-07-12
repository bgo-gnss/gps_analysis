"""gps_analysis — general GNSS time-series analysis for the IMO network.

Leaf math package (Tier 1) of the gpslibrary ecosystem: trajectory models,
robust detrending, velocity estimation (WLS → colored-noise MLE), baseline
utilities, and deformation-source inversion (Mogi → Okada → joint GPS+InSAR).

Consolidates the proven analysis code from ``~/work/projects/gps_data_analyses``
(``svartsengi-model``, the ``detrend-*`` family) into small, pure, unit-agnostic,
tested functions. Module plan: PLAN-postprocessing-revamp.md §10.2 in the
gpslibrary_new collection.

Dependency policy (hard rule, plan risk R6): numpy / scipy / gtimes only —
this package must never import geo_dataread, gps_parser, tostools, receivers,
gps_plot or gps_api.
"""

__version__ = "0.1.0"

from gps_analysis.baseline import (
    estimate_offset,
    estimate_step_offset,
    remove_offset,
    slice_window,
)
from gps_analysis.deformation import (
    MogiFit,
    MogiPosterior,
    MogiSource,
    OkadaFit,
    OkadaSource,
    halflife_days,
    local_coordinates,
    mogi_forward,
    mogi_invert,
    mogi_invert_bayes,
    mogi_mctigue,
    okada_forward,
    okada_invert,
    pressure_from_volume,
    rate_from_m3s,
    rate_to_m3s,
    time_for_rate,
    volume_from_pressure,
)
from gps_analysis.fitting import (
    ModelFunc,
    OutlierRejection,
    detrend_fit,
    fit_components,
    reject_outliers,
    remove_trend,
)
from gps_analysis.models import (
    FloatArray,
    TrajectoryParams,
    exp_linear,
    exp_linear_rate,
    linear,
    lineperiodic,
    periodic,
    poly2,
    poly2_peak_time,
    poly2_peak_value,
    poly2_rate,
)
from gps_analysis.preprocess import (
    prep_neu_series,
    prep_plot_series,
    screen_uncertainty,
)
from gps_analysis.transient import (
    BPD1Params,
    BPD2Params,
    InversionConfig,
    InversionResult,
    PriorBounds,
    bpd1_forward,
    bpd2_forward,
    detect_breakpoints,
    log_likelihood,
    noise_covariance,
    prepare_bounds,
    run_inversion,
)
from gps_analysis.velocity import (
    SlidingVelocity,
    VelocityEstimate,
    detectability_floor,
    estimate_velocity,
    horizontal_azimuth,
    horizontal_azimuth_sigma,
    horizontal_magnitude,
    horizontal_magnitude_sigma,
    sliding_velocity,
)

__all__ = [
    "__version__",
    # models
    "FloatArray",
    "TrajectoryParams",
    "linear",
    "periodic",
    "lineperiodic",
    "exp_linear",
    "exp_linear_rate",
    "poly2",
    "poly2_rate",
    "poly2_peak_time",
    "poly2_peak_value",
    # fitting
    "ModelFunc",
    "OutlierRejection",
    "fit_components",
    "detrend_fit",
    "remove_trend",
    "reject_outliers",
    # baseline
    "slice_window",
    "estimate_offset",
    "remove_offset",
    "estimate_step_offset",
    # preprocess
    "screen_uncertainty",
    "prep_plot_series",
    "prep_neu_series",
    # velocity
    "VelocityEstimate",
    "SlidingVelocity",
    "estimate_velocity",
    "sliding_velocity",
    "horizontal_magnitude",
    "horizontal_azimuth",
    "horizontal_magnitude_sigma",
    "horizontal_azimuth_sigma",
    "detectability_floor",
    # transient (GBIS4TS)
    "BPD1Params",
    "BPD2Params",
    "PriorBounds",
    "InversionConfig",
    "InversionResult",
    "bpd1_forward",
    "bpd2_forward",
    "noise_covariance",
    "log_likelihood",
    "prepare_bounds",
    "run_inversion",
    "detect_breakpoints",
    # deformation (Mogi / Okada)
    "MogiSource",
    "OkadaSource",
    "MogiFit",
    "OkadaFit",
    "MogiPosterior",
    "local_coordinates",
    "mogi_forward",
    "mogi_mctigue",
    "okada_forward",
    "mogi_invert",
    "mogi_invert_bayes",
    "okada_invert",
    "pressure_from_volume",
    "volume_from_pressure",
    "rate_to_m3s",
    "rate_from_m3s",
    "time_for_rate",
    "halflife_days",
]
