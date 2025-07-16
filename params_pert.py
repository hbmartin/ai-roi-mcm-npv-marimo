from marimo_utils.marimo_components import _DEFAULT_STEP

pert_descriptions = {
    "mini": "Min",
    "mode": "Mode",
    "maxi": "Max",
}


def same_pert_ranges(  # noqa: PLR0913
    mini: float,
    mini_value: float,
    value: float,
    maxi_value: float,
    maxi: float,
    step: float = _DEFAULT_STEP,
) -> dict[str, dict[str, float | int]]:
    return {
        "mini": {
            "lower": mini,
            "upper": maxi,
            "value": mini_value,
            "step": step,
        },
        "mode": {
            "lower": mini,
            "upper": maxi,
            "value": value,
            "step": step,
        },
        "maxi": {
            "lower": mini,
            "upper": maxi,
            "value": maxi_value,
            "step": step,
        },
    }


def pert_ranges(  # noqa: PLR0913
    mini_min: float,
    mini_value: float,
    mini_max: float,
    mode_min: float,
    mode_value: float,
    mode_max: float,
    maxi_min: float,
    maxi_value: float,
    maxi_max: float,
    step: float = _DEFAULT_STEP,
) -> dict[str, dict[str, float | int]]:
    return {
        "mini": {
            "lower": mini_min,
            "upper": mini_max,
            "value": mini_value,
            "step": step,
        },
        "mode": {
            "lower": mode_min,
            "upper": mode_max,
            "value": mode_value,
            "step": step,
        },
        "maxi": {
            "lower": maxi_min,
            "upper": maxi_max,
            "value": maxi_value,
            "step": step,
        },
    }
