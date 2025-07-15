from typing import Callable
from monaco import Sim, SimFunctions
from monaco_utils.params_to_sim import (
    case_vals_to_dict,
    output_to_case,
    params_to_model,
    params_to_sim,
)


def sim_factory(
    name: str,
    model_factory: Callable,
    factory_vars: dict,
    invars: dict,
    ndraws: int,
    *,
    verbose: bool = True,
    debug: bool = False,
):
    model = params_to_model(model_factory, factory_vars)
    sim = Sim(
        name=name,
        ndraws=ndraws,
        fcns={
            SimFunctions.PREPROCESS: case_vals_to_dict,
            SimFunctions.RUN: lambda params: (model(**params),),
            SimFunctions.POSTPROCESS: output_to_case,
        },
        debug=debug,
        verbose=verbose,
    )
    params_to_sim(sim, invars)
    return sim
