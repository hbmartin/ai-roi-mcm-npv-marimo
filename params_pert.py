from marimo_components import _DEFAULT_STEP


pert_descriptions = {
    "mini": "Min",
    "mode": "Mode",
    "maxi": "Max",
}


def same_pert_ranges(
    min: float | int,
    mini_value: float | int,
    value: float | int,
    maxi_value: float | int,
    max: float | int,
    step: float | int = _DEFAULT_STEP,
) -> dict[str, dict[str, float | int]]:
    return {
        "mini": {
            "lower": min,
            "upper": max,
            "value": mini_value,
            "step": step,
        },
        "mode": {
            "lower": min,
            "upper": max,
            "value": value,
            "step": step,
        },
        "maxi": {
            "lower": min,
            "upper": max,
            "value": maxi_value,
            "step": step,
        },
    }


def pert_ranges(
    mini_min: float | int,
    mini_value: float | int,
    mini_max: float | int,
    mode_min: float | int,
    mode_value: float | int,
    mode_max: float | int,
    maxi_min: float | int,
    maxi_value: float | int,
    maxi_max: float | int,
    step: float | int = _DEFAULT_STEP,
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
