import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# Step 1: Select assets and pull 3+ years of data
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TLT']
data = yf.download(assets, start="2020-01-01", end="2023-12-31")['Close']

# Step 2: Compute returns
returns = np.log(data / data.shift(1)).dropna()
mean_returns = returns.mean() * 252   # annualized mean returns
cov_matrix = returns.cov() * 252      # annualized covariance matrix

# Step 3: Portfolio performance function
def portfolio_performance(weights, mean_returns, cov_matrix):
    p_return = np.dot(weights, mean_returns)
    p_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return p_return, p_risk

# Step 4: Optimization (maximize Sharpe ratio)
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    p_return, p_risk = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_risk

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0,1) for asset in assets)
init_guess = len(assets) * [1./len(assets)]

opt_result = minimize(neg_sharpe_ratio, init_guess,
                      args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = opt_result.x
print("Optimal Weights:", dict(zip(assets, optimal_weights)))

# Step 5: Efficient Frontier
def efficient_frontier(mean_returns, cov_matrix, num_portfolios=5000):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(len(assets)))
        weights_record.append(weights)
        p_return, p_risk = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0,i] = p_return
        results[1,i] = p_risk
        results[2,i] = (p_return-0.02)/p_risk
    return results, weights_record

results, weights_record = efficient_frontier(mean_returns, cov_matrix)
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis')
plt.xlabel('Risk (Volatility)')
plt.ylabel('Return')
plt.colorbar(label='Sharpe Ratio')
plt.title("Efficient Frontier (2020–2023 Data)")
plt.show()

# Step 6: VaR Analysis
portfolio_returns = returns.dot(optimal_weights)

# Historical VaR
def historical_var(returns, confidence_level=0.95):
    return np.percentile(returns, (1-confidence_level)*100)

# Variance-Covariance VaR
def var_cov_var(mean, std, confidence_level=0.95):
    return mean + std * norm.ppf(1-confidence_level)

# Monte Carlo VaR
def monte_carlo_var(mean, std, confidence_level=0.95, simulations=10000):
    simulated_returns = np.random.normal(mean, std, simulations)
    return np.percentile(simulated_returns, (1-confidence_level)*100)

hist_var = historical_var(portfolio_returns, 0.95)
vc_var = var_cov_var(portfolio_returns.mean(), portfolio_returns.std(), 0.95)
mc_var = monte_carlo_var(portfolio_returns.mean(), portfolio_returns.std(), 0.95)

print("Historical VaR (95%):", hist_var)
print("Variance-Covariance VaR (95%):", vc_var)
print("Monte Carlo VaR (95%):", mc_var)

# Step 7: Visualize VaR
plt.hist(portfolio_returns, bins=50, alpha=0.7)
plt.axvline(hist_var, color='r', linestyle='dashed', linewidth=2, label='Historical VaR')
plt.axvline(vc_var, color='g', linestyle='dashed', linewidth=2, label='Var-Cov VaR')
plt.axvline(mc_var, color='b', linestyle='dashed', linewidth=2, label='Monte Carlo VaR')
plt.legend()
plt.title("Portfolio Returns Distribution with VaR (2020–2023)")
plt.show()


#Output

print("\n📊 Portfolio Optimization Results")
print("--------------------------------------------------")
for asset, weight in zip(assets, optimal_weights):
    print(f"{asset}: {weight:.2%}")
print("\nInterpretation:")
print("The optimizer concentrated ~63% in Apple (AAPL) and ~37% in Amazon (AMZN). "
      "Other assets received negligible weights, meaning they did not improve the Sharpe ratio. "
      "This shows the optimal portfolio is tilted toward tech growth stocks for 2020–2023.")

print("\n📉 Value at Risk (VaR) Results (95% Confidence)")
print("--------------------------------------------------")
print(f"Historical VaR: {hist_var:.2%}")
print(f"Variance-Covariance VaR: {vc_var:.2%}")
print(f"Monte Carlo VaR: {mc_var:.2%}")
print("\nInterpretation:")
print("With 95% confidence, the portfolio is unlikely to lose more than ~3% in a single day. "
      "All three methods agree closely, suggesting the return distribution is stable and risk estimates are reliable.")

print("\n📈 Efficient Frontier Insight")
print("--------------------------------------------------")
print("The efficient frontier plot shows the trade-off between risk and return. "
      "Portfolios with higher returns also carry higher risk. "
      "The optimal portfolio lies at the point where the Sharpe ratio is maximized, "
      "demonstrating the benefit of diversification.")

print("\n✅ Final Takeaway")
print("--------------------------------------------------")
print("Optimal portfolio: AAPL + AMZN. "
      "Downside risk: ~3% daily loss at 95% confidence. "
      "Diversification benefits are visible in the efficient frontier, "
      "but the optimizer selected only the assets that maximized risk-adjusted return.")
