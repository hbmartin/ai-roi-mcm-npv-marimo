from typing import Callable
import marimo as mo
import matplotlib.pyplot as plt
from scipy.stats import uniform, triang, beta, norm
import numpy as np

# Dictionary mapping distribution keys to scipy callables
SCIPY_DISTRIBUTIONS = {"uniform": uniform, "triang": triang, "beta": beta, "norm": norm}

_distributions = {
    "triang": {
        "c": {"description": "Center (% of width)", "lower": 0, "upper": 1},
        "loc": {
            "description": "Lower bound",
            "lower": None,
            "upper": None,
        },
        "scale": {
            "description": "Width",
            "lower": 0,
            "upper": None,
        },
    },
    "norm": {
        "loc": {
            "description": "Mean",
            "lower": None,
            "upper": None,
        },
        "scale": {
            "description": "Standard deviation",
            "lower": 0,
            "upper": None,
        },
    },
    "uniform": {
        "loc": {
            "description": "Lower bound",
            "lower": None,
            "upper": None,
        },
        "scale": {
            "description": "Width ",
            "lower": 0,
            "upper": None,
        },
    },
    "beta": {
        "a": {
            "description": "Alpha (a > 0)",
            "lower": 0,
            "upper": None,
        },
        "b": {
            "description": "Beta (b > 0)",
            "lower": 0,
            "upper": None,
        },
        "loc": {
            "description": "Lower bound",
            "lower": None,
            "upper": None,
        },
        "scale": {
            "description": "Width",
            "lower": 0,
            "upper": None,
        },
    },
}


def _deep_merge(result, dict2):
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def generate_ranged_distkwargs(distribution: str, ranged_distkwargs: dict) -> dict:
    ranged_copy = _distributions[distribution].copy()

    for p_name, ranges in ranged_copy.items():
        if (
            p_name not in ranged_distkwargs
            and not ranges["lower"]
            and not ranges["upper"]
        ):
            raise ValueError(
                f"Missing required parameter: {p_name}, must set any `None`s in {ranged_copy[p_name]}"
            )
        elif p_name not in ranged_distkwargs:
            continue

        if (
            "lower" in ranged_distkwargs[p_name]
            and ranges["lower"] is not None
            and ranged_distkwargs[p_name]["lower"] < ranges["lower"]
        ):
            raise ValueError(
                f"{p_name}: given lower bound {ranged_distkwargs[p_name]['lower']} is less than allowed: {ranges['lower']}"
            )
        elif "lower" not in ranged_distkwargs[p_name] and ranges["lower"] is None:
            raise ValueError(f"{p_name}: lower bound is not provided")

        if (
            "upper" in ranged_distkwargs[p_name]
            and ranges["upper"] is not None
            and ranged_distkwargs[p_name]["upper"] > ranges["upper"]
        ):
            raise ValueError(
                f"{p_name}: given upper bound {ranged_distkwargs[p_name]['upper']} is greater than allowed: {ranges['upper']}"
            )
        elif "upper" not in ranged_distkwargs[p_name] and ranges["upper"] is None:
            raise ValueError(f"{p_name}: upper bound is not provided")

    return _deep_merge(ranged_copy, ranged_distkwargs)


def _distribution_params(
    name: str,
    dist: str | Callable,
    ranged_distkwargs: dict,
):
    _dist = dist if isinstance(dist, Callable) else SCIPY_DISTRIBUTIONS[dist]

    params = mo.ui.dictionary(
        {
            p_name: mo.ui.slider(
                start=ranges["lower"],
                stop=ranges["upper"],
                step=0.01 if "step" not in ranges else ranges["step"],
                value=(
                    (ranges["lower"] + ranges["upper"]) / 2
                    if "value" not in ranges
                    else ranges["value"]
                ),
            )
            for p_name, ranges in ranged_distkwargs.items()
        }
    )

    html = mo.Html(
        f"<h2>{name}</h2><table>"
        + "\n".join(
            [
                f"<tr><td>{p_name if 'description' not in ranges else ranges['description']}</td><td>{params[p_name]}</td></tr>"
                for p_name, ranges in ranged_distkwargs.items()
            ]
        )
        + "</table>"
    )

    return {
        "name": name,
        "ui": html.batch(
            **{p_name: params[p_name] for p_name in ranged_distkwargs.keys()}
        ),
        "params": params,
        "dist": _dist,
    }


def distribution_params(
    ranged_distkwargs: dict,
):

    params = mo.ui.dictionary(
        {
            p_name: mo.ui.slider(
                start=ranges["lower"],
                stop=ranges["upper"],
                step=0.01 if "step" not in ranges else ranges["step"],
                value=(
                    (ranges["lower"] + ranges["upper"]) / 2
                    if "value" not in ranges
                    else ranges["value"]
                ),
            )
            for p_name, ranges in ranged_distkwargs.items()
        }
    )

    return params


def _dist_plot(params: dict, dist: Callable):
    _dist = dist(**params)
    x_min = _dist.ppf(0.0005)
    x_max = _dist.ppf(0.9995)
    x = np.linspace(x_min, x_max, 100)
    pdf_values = _dist.pdf(x)

    fig = plt.figure(figsize=(2, 2))
    plt.plot(x, pdf_values, "b-", linewidth=2, label=None)
    plt.fill_between(x, pdf_values, alpha=0.3)
    plt.ylabel("Probability Density")
    plt.grid(True, alpha=0.3)
    return mo.as_html(fig)


def display_params(
    name: str,
    params: mo.ui.dictionary,
    dist: str | Callable,
    descriptions: dict = {},
):
    _dist = dist if isinstance(dist, Callable) else SCIPY_DISTRIBUTIONS[dist]
    parameter_descriptions = (
        descriptions
        if descriptions or not isinstance(dist, str)
        else {
            k: (
                k
                if dist not in _distributions
                else _distributions[str(dist)][k].get("description")
            )
            for k, _ in params.items()
        }
    )
    html = mo.Html(
        "<table>"
        + "\n".join(
            [
                f"<tr><td>{parameter_descriptions[k]}</td><td>{v}</td></tr>"
                for k, v in params.items()
            ]
        )
        + "</table>"
    )
    return mo.hstack(
        [
            mo.vstack([mo.md(f"## {name}"), html]),
            _dist_plot({k: v.value for k, v in params.items()}, _dist),
        ],
        align="start",
        widths=[2, 1],
    )
