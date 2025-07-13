# AI ROI Monte Carlo NPV Analysis

This project provides a comprehensive Monte Carlo simulation for analyzing the Net Present Value (NPV) of AI implementation initiatives. The analysis is built using Python with interactive controls and advanced statistical modeling.

## Setup Instructions

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Clone or download this repository
2. Navigate to the project directory:
3. Install dependencies using uv:
   ```bash
   uv sync
   ```

4. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

5. Start the marimo notebook:
   ```bash
   marimo edit ai_roi_mcm_npv.py
   ```

The interactive analysis will open in your web browser.

## Key Features

- **NPV Model**: Based on Excel financial model with 4 benefit categories:
  - Time savings benefits from automation
  - Quality improvements through bug reduction
  - Product delivery acceleration
  - Employee retention improvements

- **Interactive Controls**: Real-time parameter adjustment through sliders:
  - Hours saved per employee per week
  - Number of employees
  - Fully-loaded hourly rate
  - Bug reduction percentage
  - Discount rate
  - Number of Monte Carlo simulations

- **Monte Carlo Simulation**: Uses appropriate probability distributions for uncertainty modeling with thousands of simulation runs

- **Comprehensive Visualization**: 
  - NPV distribution histogram with mean and break-even lines
  - Annual benefits breakdown
  - Risk assessment pie chart showing probability of positive NPV
  - Correlation analysis between benefit components
  - Statistical summary tables with percentiles

- **Risk Analysis**: 
  - Probability of positive NPV calculation
  - 90% confidence intervals
  - Comprehensive percentile analysis (5th, 10th, 25th, 50th, 75th, 90th, 95th)
  - Visual risk assessment tools

## Probability Distributions Used

The simulation incorporates realistic uncertainty modeling through various probability distributions:

- **Triangular Distribution**: 
  - Hours saved per employee (asymmetric uncertainty with most likely value)
  - Current bug costs (business estimates with skewed uncertainty)

- **Normal Distribution**: 
  - Hourly rates (symmetric uncertainty around market rates)

- **Beta Distribution**: 
  - Bug reduction effectiveness (bounded percentage with realistic shape)
  - Delivery improvement percentages (constrained business impact factors)
  - Employee retention improvements (realistic HR impact modeling)

- **Uniform Distribution**: 
  - Discount rates (range of acceptable cost of capital)
  - Productivity conversion factors (operational efficiency uncertainty)

## Usage

Run the interactive analysis:
```bash
python ai_roi_mcm_npv.py
```

Adjust the parameter sliders to see how different assumptions affect the NPV distribution and risk profile in real-time.
