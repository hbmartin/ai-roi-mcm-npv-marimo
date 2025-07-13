# Project Overview

This is an AI ROI Monte Carlo NPV analysis project built with interactive marimo notebooks. The project provides a comprehensive simulation for analyzing the Net Present Value of AI implementation initiatives using statistical modeling and uncertainty analysis.

## Core Components

**NPV Model (`npv_model` function)**: 
- Calculates 3-year NPV based on 4 benefit categories: time savings, quality improvements, product delivery acceleration, employee retention
- Returns comprehensive financial metrics including annual benefits breakdown

**Monte Carlo Simulation**: 
- Built on monaco framework with preprocess/run/postprocess pattern
- Models uncertainty through probability distributions (triangular, normal, beta, uniform)
- Generates thousands of scenarios to assess NPV risk and variability

**Interactive Interface**: 
- marimo-based reactive notebook with parameter sliders
- Real-time visualization updates as parameters change
- Four-panel dashboard: NPV distribution, benefits breakdown, correlation analysis, risk assessment

## Data Flow

1. **Parameter Input**: Interactive sliders set base values for key business variables
2. **Distribution Modeling**: Base values inform probability distributions for uncertainty modeling
3. **Monte Carlo Execution**: monaco runs thousands of NPV calculations with sampled parameters
4. **Results Analysis**: Statistical analysis and visualization of NPV outcomes
5. **Risk Assessment**: Probability calculations and percentile analysis for decision support

## File Structure

- `ai_roi_mcm_npv.py`: Main interactive NPV analysis (marimo notebook)
- `pyproject.toml`: Dependencies managed via uv package manager

## Key Dependencies

- **marimo**: Interactive notebook environment with reactive cells - docs at https://docs.marimo.io/api/
- **monaco**: Monte Carlo simulation framework requiring preprocess/run/postprocess functions - docs at https://monaco.readthedocs.io/en/latest/api_reference.html
- **scipy.stats**: Probability distributions (norm, uniform, triang, beta)
- **matplotlib**: Visualization with 4-panel dashboard layout

## Monaco Simulation Pattern

The project follows monaco's required simulation pattern:
- `preprocess(case)`: Extract parameters from input variables and constants to structure randomized or constant inputs for the model
- `run(params)`: Execute NPV model with sampled parameters  
- `postprocess(case, output)`: Extract desired output values and return results for analysis

## Monaco Simulation best practicves

- Organize Code with Clear Separation of Concerns: serparate Input generation, Simulation core, Output processing, adn Configuration
- Leverage NumPy Vectorization: Vectorization is crucial for performance. Instead of loops, use NumPy's vectorized operations.
- Implement Efficient Memory Management: Pre-allocate arrays when possible, Use generators for large datasets, Clear unnecessary variables with `del`
- Implement Robust Input Validation: use `assert` to validate inputs, checking for values outside of expected ranges
- Use pytest cells in Marimo to comprehensively test the core model
- Comment key variables and key stages in model calculation. Use Marimo markdown when this might be useful to the end user (non-programmer).
- After simulation runs, perform sanity checks on output.
- Choose Appropriate Data Structures: e.g. Lists for small, dynamic collections and NumPy arrays for numerical computations
- Leverage SciPy's statistical distributions for input generation
- Constants vs Variables: Use `sim.addConstVal()` for fixed parameters and `sim.addInVar()` for uncertain parameters with distributions.

## Marimo notebook best practices

Marimo operates on a paradigm of reactive execution where cells automatically update when their dependencies change, creating a directed acyclic graph (DAG) based on variable references rather than cell order.

- Start with Library Imports: Begin notebooks with a standard import cell
- Use Global Variables Sparingly: Keep the number of global variables in your program small to avoid name collisions. Each global variable must be defined by only one cell in marimo, which encourages cleaner code organization.
- Use Descriptive Variable Names: Choose clear, descriptive names especially for global variables. 
- Handle Temporary Variables Properly: For intermediate calculations, use local variables prefixed with underscore (_) or encapsulate logic in functions.
- Encapsulate Logic in Functions: Use functions to avoid polluting the global namespace and enable code reuse.
- Avoid Cross-Cell Mutations: Marimo doesn't track object mutations, so avoid defining variables in one cell and mutating them in another.
- Write Idempotent Cells: Design cells to produce the same output when given the same inputs.
- Bind UI Elements to Global Variables: For UI interactions to work properly, elements must be assigned to global variables.
- Use Composite Elements for Complex UIs: Leverage mo.ui.array, mo.ui.dictionary, and mo.ui.form for structured interfaces.
- Use Pytest Integration: Marimo automatically runs pytest on on cells that consist exclusively of test code - functions whose names start with test_ or classes whose names start with Test.
- Multiple definitions: Each variable can only be defined in one cell
- Cycles: Avoid circular dependencies between cells
- Import restrictions: No import * statements allowed
- Design for Dual Usage: Structure notebooks to work both interactively and as scripts
- Clean Up Resources: Use `del` to explicitly manage memory for large datasets.
- Use Layout Functions: Organize outputs with marimo's layout system.
- Avoid Synchronous Assumptions: Don't assume execution order based on cell position.

## Marimo Data Flow
Marimo treats global variables and variables passed as function parameters within a cell very differently, and this distinction is central to how marimo achieves reproducibility and reactivity:

Global variables in marimo are shared across cells and form the backbone of the notebook's reactive dataflow. Each global variable must be defined in only one cell, and marimo uses static analysis to build a dependency graph based on which cells define and which cells read these globals. When a global variable is updated, marimo automatically re-executes all cells that depend on it.

Variables passed as function parameters or defined as local variables (including those prefixed with an underscore, such as _tmp) are only accessible within the cell or function where they are defined. They do not participate in marimo's global dependency tracking and cannot be accessed by other cells. This means they are not part of the reactive execution graph and changing them does not trigger updates in other cells