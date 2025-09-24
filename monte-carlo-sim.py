import os
import time
import math
import numpy as np

try:
    import cupy as cp
    array_module = cp
    use_gpu = True
except ImportError:
    array_module = np
    use_gpu = False

def compute_percentile(data, percentile):
    """Calculate the specified percentile of the data."""
    return array_module.percentile(data, percentile)

def to_numpy_array(data):
    """Convert array to NumPy for CPU-based processing."""
    return cp.asnumpy(data) if use_gpu else data

def calculate_portfolio_parameters(weights, returns, volatilities, corr_matrix):
    """Compute portfolio expected return and volatility."""
    diag_vol = array_module.diag(volatilities)
    cov_matrix = diag_vol @ corr_matrix @ diag_vol
    portfolio_return = float(weights @ returns)
    portfolio_variance = float(weights @ cov_matrix @ weights)
    portfolio_volatility = math.sqrt(portfolio_variance)
    return portfolio_return, portfolio_volatility, cov_matrix

def simulate_terminal_values(
    initial_capital: float,
    time_horizon: float,
    num_simulations: int,
    portfolio_return: float,
    portfolio_volatility: float,
    seed: int
):
    """Simulate terminal portfolio values using GBM."""
    if use_gpu:
        cp.random.seed(seed)
    else:
        np.random.seed(seed)

    drift = (portfolio_return - 0.5 * portfolio_volatility**2) * time_horizon
    diffusion = portfolio_volatility * math.sqrt(time_horizon)
    random_shocks = array_module.random.normal(0, 1, num_simulations, dtype=array_module.float32)
    terminal_values = initial_capital * array_module.exp(drift + diffusion * random_shocks)
    
    if use_gpu:
        cp.cuda.Stream.null.synchronize()
    
    return terminal_values

def calculate_risk_metrics(initial_capital, terminal_values):
    """Compute risk metrics for the portfolio."""
    losses = initial_capital - terminal_values
    mean_terminal = float(to_numpy_array(terminal_values.mean()))
    std_terminal = float(to_numpy_array(terminal_values.std()))
    var_95 = float(to_numpy_array(compute_percentile(losses, 95)))
    cvar_95 = float(to_numpy_array(losses[losses >= var_95].mean()))
    prob_loss = float(to_numpy_array((terminal_values < initial_capital).mean()))
    p10 = float(to_numpy_array(compute_percentile(terminal_values, 10)))
    p50 = float(to_numpy_array(compute_percentile(terminal_values, 50)))
    p90 = float(to_numpy_array(compute_percentile(terminal_values, 90)))
    
    return mean_terminal, std_terminal, var_95, cvar_95, prob_loss, p10, p50, p90

def print_simulation_results(
    device, num_simulations, time_horizon, initial_capital,
    portfolio_return, portfolio_volatility, sim_time,
    mean_terminal, std_terminal, var_95, cvar_95, prob_loss, p10, p50, p90
):
    """Print simulation results in a formatted manner."""
    print("\n=== Portfolio Risk Simulation (GBM) ===")
    print(f"Computation Device:   {device}")
    print(f"Number of Paths:     {num_simulations:,}")
    print(f"Time Horizon:        {time_horizon} years")
    print(f"Initial Capital:     ${initial_capital:,.2f}")
    print(f"Portfolio Return:    {portfolio_return:.4f} (annual)")
    print(f"Portfolio Volatility: {portfolio_volatility:.4f} (annual)")
    print("\n--- Simulation Results ---")
    print(f"Mean Terminal Value: ${mean_terminal:,.2f}")
    print(f"Std Terminal Value:  ${std_terminal:,.2f}")
    print(f"10th Percentile:     ${p10:,.2f}")
    print(f"Median Value:        ${p50:,.2f}")
    print(f"90th Percentile:     ${p90:,.2f}")
    print(f"VaR (95%):           ${var_95:,.2f} (loss)")
    print(f"CVaR (95%):          ${cvar_95:,.2f} (avg loss in worst 5%)")
    print(f"Loss Probability:    {prob_loss*100:.2f}%")
    print(f"\nSimulation Time:     {sim_time:.2f} seconds\n")

def run_simulation(
    num_simulations: int = 10_000_000,
    initial_capital: float = 100_000.0,
    time_horizon: float = 1.0,
    seed: int = 42
):
    """
    Run Monte Carlo simulation for portfolio risk using GBM.
    Portfolio consists of assets with specified weights, returns, volatilities, and correlations.
    """
    # Portfolio configuration
    weights = array_module.array([0.30, 0.25, 0.25, 0.20])  # Sums to 1
    returns = array_module.array([0.08, 0.05, 0.12, 0.03])   # Annual expected returns
    volatilities = array_module.array([0.20, 0.15, 0.25, 0.10])  # Annual volatilities
    corr_matrix = array_module.array([
        [1.00, 0.40, 0.30, 0.10],
        [0.40, 1.00, 0.35, 0.15],
        [0.30, 0.35, 1.00, 0.05],
        [0.10, 0.15, 0.05, 1.00],
    ])

    # Calculate portfolio parameters
    portfolio_return, portfolio_volatility, _ = calculate_portfolio_parameters(
        weights, returns, volatilities, corr_matrix
    )

    # Run simulation
    start_time = time.time()
    terminal_values = simulate_terminal_values(
        initial_capital, time_horizon, num_simulations,
        portfolio_return, portfolio_volatility, seed
    )
    sim_time = time.time() - start_time

    # Calculate risk metrics
    mean_terminal, std_terminal, var_95, cvar_95, prob_loss, p10, p50, p90 = calculate_risk_metrics(
        initial_capital, terminal_values
    )

    # Display results
    device = "GPU (CuPy/CUDA)" if use_gpu else "CPU (NumPy)"
    print_simulation_results(
        device, num_simulations, time_horizon, initial_capital,
        portfolio_return, portfolio_volatility, sim_time,
        mean_terminal, std_terminal, var_95, cvar_95, prob_loss, p10, p50, p90
    )

if __name__ == "__main__":
    num_simulations = int(os.getenv("N_PATHS", "10000000"))
    time_horizon = float(os.getenv("HORIZON_YEARS", "1.0"))
    run_simulation(num_simulations=num_simulations, time_horizon=time_horizon)