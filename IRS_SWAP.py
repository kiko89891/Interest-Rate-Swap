import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 0.1  # speed of mean reversion
b = 0.05  # long-term mean
sigma = 0.01  # volatility
r0 = 0.03  # initial interest rate
T = 3.0  # total time in years (now 3 years)
dt = 1 / 252  # daily steps (based on trading days)
N = int(T / dt)  # number of steps
n_sim = 10_000  # number of simulations

# Preallocate matrix to hold all simulations
rates = np.zeros((n_sim, N + 1))
rates[:, 0] = r0

# Simulate paths
np.random.seed(42)  # for reproducibility
for t in range(1, N + 1):
    Z = np.random.normal(0, 1, n_sim)
    rates[:, t] = rates[:, t - 1] + a * (b - rates[:, t - 1]) * dt + sigma * np.sqrt(dt) * Z

# Time grid for plotting
time_grid = np.linspace(0, T, N + 1)

# Plot 10 sample paths
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.plot(time_grid, rates[i, :], lw=0.8, alpha=0.6, label=f'Path {i + 1}' if i == 0 else None)

mean_path = np.mean(rates, axis=0)
plt.plot(time_grid, mean_path, color='black', lw=2, label='Mean Path')
plt.axhline(b, color='red', linestyle='--', lw=2, label='Long-Term Mean (b)')

plt.title('Vasicek Interest Rate Simulation (10,000 paths, 3 years)')
plt.xlabel('Time (Years)')
plt.ylabel('Interest Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Define quantiles to extract
quantiles_to_extract = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# Extract quantile info at each year end (t=1, t=2, t=3)
for year in range(1, 4):
    idx = int(year / dt)  # index corresponding to year in simulation steps
    rates_at_year = rates[:, idx]

    quantile_values = np.quantile(rates_at_year, quantiles_to_extract)

    print(f"\n--- Quantile Analysis of Interest Rates at Year {year} ---")
    print(f"Mean rate at Year {year}: {np.mean(rates_at_year):.4f}")
    print(f"Standard Deviation at Year {year}: {np.std(rates_at_year):.4f}")
    for q, val in zip(quantiles_to_extract, quantile_values):
        print(f"{q * 100:.1f}th percentile rate: {val:.4f}")

# Optional: Plot distribution of final year rates with quantiles
plt.figure(figsize=(10, 6))
final_year_rates = rates[:, -1]
plt.hist(final_year_rates, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(np.mean(final_year_rates), color='green', linestyle='dashed', linewidth=2,
            label=f'Mean: {np.mean(final_year_rates):.4f}')
for q, val in zip(quantiles_to_extract, np.quantile(final_year_rates, quantiles_to_extract)):
    plt.axvline(val, color='red', linestyle=':', linewidth=1,
                label=f'{q * 100:.1f}th Percentile' if q in [0.01, 0.5, 0.99] else None)
plt.title('Distribution of Vasicek Rates at Year 3')
plt.xlabel('Interest Rate')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
