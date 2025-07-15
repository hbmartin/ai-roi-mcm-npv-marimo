import re
from typing import Callable
from monaco import Case, Sim

_CONST = "_constant"


def _key(k: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s_]", "", k).strip().lower().replace(" ", "_")


def params_to_sim(sim: Sim, invars: dict):
    for name, details in invars.items():
        if details["dist"] != _CONST:
            sim.addInVar(
                name=name,
                dist=details["dist"],
                distkwargs={k: v.value for k, v in details["params"].items()},
            )
        else:
            if hasattr(details["params"], "items") and callable(
                getattr(details["params"], "items")
            ):
                for k, v in details["params"].items():
                    sim.addConstVal(name=k, val=v.value)
            elif hasattr(details["params"], "value"):
                sim.addConstVal(name=name, val=details["params"].value)
            else:
                sim.addConstVal(name=name, val=details["params"])
    return sim


def case_vals_to_dict(case: Case):
    return (
        {_key(k): v.val for k, v in case.invals.items()}
        | {_key(k): v for k, v in case.constvals.items()},
    )


def output_to_case(case: Case, output: dict):
    for k, v in output.items():
        case.addOutVal(name="_".join([w.capitalize() for w in k.split("_")]), val=v)


def params_to_model(model_factory: Callable, factory_vars: dict) -> Callable:
    return model_factory(
        **{
            _key(k): v["params"].value if isinstance(v, dict) else v
            for k, v in factory_vars.items()
        }
    )


def outvals_to_dict(sim: Sim) -> dict:
    keys = sim.cases[0].outvals.keys()
    results = {k: [] for k in keys}
    for case in sim.cases:
        for k in keys:
            results[k].append(case.outvals[k].val)
    return results
