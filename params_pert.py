pert_metadata = {
    "mini": "Min",
    "mode": "Mode",
    "maxi": "Max",
}


def generate_pert_ranges(
    mini_min,
    mini_value,
    mini_max,
    mode_min,
    mode_value,
    mode_max,
    maxi_min,
    maxi_value,
    maxi_max,
):
    return {
        "mini": {
            "lower": mini_min,
            "upper": mini_max,
            "value": mini_value,
        },
        "mode": {
            "lower": mode_min,
            "upper": mode_max,
            "value": mode_value,
        },
        "maxi": {
            "lower": maxi_min,
            "upper": maxi_max,
            "value": maxi_value,
        },
    }
