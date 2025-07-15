from monaco import Sim


def params_to_sim(sim: Sim, invars: dict):
    for name, details in invars.items():
        sim.addInVar(
            name=name,
            dist=details["dist"],
            distkwargs={k: v.value for k, v in details["params"].items()},
        )
    return sim
