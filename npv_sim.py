from scipy.stats import uniform, triang, beta
from npv_model import npv_model


def npv_sim(
    bug_reduction,
    discount_rate,
    employees,
    hourly_rate,
    hours_saved,
    mo,
    monaco,
    n_samples,
    norm,
):
    # Define simulation functions
    def preprocess(case):
        print(case.invals["Hours Saved per Employee"].val)
        return (
            {
                "hours_saved": case.invals["Hours Saved per Employee"].val,
                "employees": case.constvals["employees"],
                "hourly_rate": case.invals["hourly_rate"].val,
                "bug_reduction": case.invals["bug_reduction"].val,
                "discount_rate": case.invals["discount_rate"].val,
                "productivity_rate": case.invals["productivity_rate"].val,
                "current_bug_cost": case.invals["current_bug_cost"].val,
                "delivery_improvement": case.invals["delivery_improvement"].val,
                "retention_improvement": case.invals["retention_improvement"].val,
            },
        )

    def run(params):
        result = npv_model(**params)
        return (result,)

    def postprocess(case, simulation_output):
        case.addOutVal(name="NPV", val=simulation_output["npv"])
        case.addOutVal(name="Annual_Benefits", val=simulation_output["annual_benefits"])
        case.addOutVal(name="Time_Savings", val=simulation_output["time_savings"])
        case.addOutVal(name="Quality_Savings", val=simulation_output["quality_savings"])
        case.addOutVal(name="Year_1_Net", val=simulation_output["year_1_net"])

    # Create monaco simulation
    sim = monaco.Sim(
        name="ai_roi_npv_analysis",
        ndraws=n_samples,
        fcns={"preprocess": preprocess, "run": run, "postprocess": postprocess},
        debug=True,
        verbose=True,
    )

    ## Constants
    sim.addConstVal(name="employees", val=employees)

    ## Time Savings Benefits (Internal)
    # sim.addInVar(
    #     name="hours_saved",
    #     dist=triang,
    #     distkwargs={"c": 0.5, "loc": hours_saved * 0.7, "scale": hours_saved * 0.6},
    # )
    sim.addInVar(
        name="hourly_rate",
        dist=norm,
        distkwargs={"loc": hourly_rate, "scale": hourly_rate * 0.1},
    )
    sim.addInVar(
        name="productivity_rate", dist=uniform, distkwargs={"loc": 0.55, "scale": 0.2}
    )

    ## Quality Improvement Benefits
    # Bug reduction (beta distribution)
    sim.addInVar(
        name="bug_reduction",
        dist=beta,
        distkwargs={
            "a": 3,
            "b": 5,
            "loc": bug_reduction * 0.7,
            "scale": bug_reduction * 0.6,
        },
    )

    # Discount rate (uniform distribution)
    sim.addInVar(
        name="discount_rate",
        dist=uniform,
        distkwargs={"loc": discount_rate * 0.8, "scale": discount_rate * 0.4},
    )

    sim.addInVar(
        name="current_bug_cost",
        dist=triang,
        distkwargs={"c": 0.3, "loc": 150000, "scale": 100000},
    )

    ## Product Delivery Benefits
    sim.addInVar(
        name="delivery_improvement",
        dist=beta,
        distkwargs={"a": 2, "b": 3, "loc": 0.1, "scale": 0.2},
    )

    ## Employee Retention Benefits
    sim.addInVar(
        name="retention_improvement",
        dist=beta,
        distkwargs={"a": 2, "b": 4, "loc": 0.1, "scale": 0.25},
    )

    return sim
