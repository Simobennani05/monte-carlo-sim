# Portfolio Risk Simulation (GBM)

## Overview
This Python script (`portfolio_simulation.py`) performs a Monte Carlo simulation to estimate the risk and performance of a financial portfolio modeled using Geometric Brownian Motion (GBM). It calculates key risk metrics such as Value at Risk (VaR), Conditional Value at Risk (CVaR), probability of loss, and percentiles of terminal portfolio values. The script supports both CPU (NumPy) and GPU (CuPy/CUDA) computation for improved performance with large numbers of simulations.

## Features
- Simulates portfolio terminal values over a specified time horizon using GBM.
- Computes portfolio expected return and volatility based on asset weights, returns, volatilities, and correlations.
- Calculates risk metrics:
  - Mean and standard deviation of terminal values.
  - 10th, 50th (median), and 90th percentiles.
  - VaR (95%) and CVaR (95%) for loss assessment.
  - Probability of portfolio loss.
- Supports GPU acceleration with CuPy if available, falling back to NumPy otherwise.
- Configurable via environment variables for number of simulations and time horizon.

## Requirements
- **Python 3.6+**
- **Required Libraries**:
  - `numpy` (for CPU-based computation)
  - `cupy` (optional, for GPU acceleration with CUDA-compatible GPU)
- **Optional**: CUDA toolkit (for CuPy support)

Install dependencies using:
```bash
pip install numpy
pip install cupy-cudaXX  # Replace XX with your CUDA version (e.g., cupy-cuda11x)
```

## Usage
1. **Run the script**:
   ```bash
   python portfolio_simulation.py
   ```
   This uses default parameters:
   - Number of simulations: 10,000,000
   - Initial portfolio value: $100,000
   - Time horizon: 1 year
   - Random seed: 42

2. **Customize parameters** using environment variables:
   ```bash
   export N_PATHS=5000000
   export HORIZON_YEARS=2.0
   python portfolio_simulation.py
   ```

3. **Portfolio configuration**:
   - The portfolio consists of four assets with:
     - Weights: `[0.30, 0.25, 0.25, 0.20]` (summing to 1)
     - Annual expected returns: `[0.08, 0.05, 0.12, 0.03]`
     - Annual volatilities: `[0.20, 0.15, 0.25, 0.10]`
     - Correlation matrix:
       ```
       [[1.00, 0.40, 0.30, 0.10],
        [0.40, 1.00, 0.35, 0.15],
        [0.30, 0.35, 1.00, 0.05],
        [0.10, 0.15, 0.05, 1.00]]
       ```
   - Modify these directly in the `run_simulation` function if needed.

## Output
The script prints a summary of the simulation results, including:
- Device used (GPU or CPU)
- Number of simulation paths and time horizon
- Portfolio expected return and volatility
- Risk metrics (mean, standard deviation, percentiles, VaR, CVaR, loss probability)
- Computation time

### Example Output
```
=== Portfolio Risk Simulation (GBM) ===
Computation Device:   CPU (NumPy)
Number of Paths:     10,000,000
Time Horizon:        1.0 years
Initial Capital:     $100,000.00
Portfolio Return:    0.0700 (annual)
Portfolio Volatility: 0.1492 (annual)

--- Simulation Results ---
Mean Terminal Value: $107,010.12
Std Terminal Value:  $14,915.23
10th Percentile:     $87,123.45
Median Value:        $106,987.67
90th Percentile:     $127,456.89
VaR (95%):           $15,234.56 (loss)
CVaR (95%):          $19,876.32 (avg loss in worst 5%)
Loss Probability:    24.32%

Simulation Time:     2.45 seconds
```

## Notes
- **Performance**: Use a CUDA-compatible GPU with CuPy for faster simulations with large `N_PATHS`.
- **Customization**: Update asset weights, returns, volatilities, or correlations in the script to model different portfolios.
- **Limitations**: The simulation assumes a GBM model, which may not capture all real-world market dynamics (e.g., fat tails, jumps).

## License
This project is provided as-is for educational purposes. No warranty is implied.
