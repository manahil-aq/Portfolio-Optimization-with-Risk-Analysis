1. Introduction 

This project explores Portfolio Optimization with Risk Analysis using Python. The goal is to apply Modern Portfolio Theory (MPT) to construct an efficient portfolio and evaluate downside risk using Value at Risk (VaR). By combining optimization techniques with risk metrics, the project demonstrates how investors can balance return expectations with volatility and potential losses. 

2. Problem Statement 

Investors face the challenge of allocating capital across multiple assets to maximize returns while minimizing risk. Traditional approaches often rely on intuition or single-asset performance, which can lead to inefficient portfolios. This project addresses the problem by applying quantitative optimization and risk analysis, ensuring decisions are data-driven and theoretically sound. The importance lies in demonstrating how computational finance tools can guide real-world investment strategies. 

3. Data Description 

Source: Daily stock prices retrieved from Yahoo Finance via the yfinance library. 

Assets: Apple (AAPL), Microsoft (MSFT), Google (GOOGL), Amazon (AMZN), and iShares 20+ Year Treasury Bond ETF (TLT). 

Features: Closing prices, log returns, mean returns, covariance matrix. 

Challenges: Handling missing data, ensuring consistent timeframes, and annualizing daily statistics for comparability. 

4. Methodology 

Techniques Used: Modern Portfolio Theory (MPT), Sharpe ratio optimization, Efficient Frontier simulation, and Value at Risk (VaR). 

Step-by-Step Approach:  

Collect 3+ years of historical price data. 

Compute log returns, mean returns, and covariance matrix. 

Optimize portfolio weights using scipy.optimize to maximize Sharpe ratio. 

Simulate thousands of random portfolios to plot the efficient frontier. 

Calculate VaR using Historical, Variance-Covariance, and Monte Carlo methods. 

Justification: MPT provides a theoretical foundation for diversification, while VaR quantifies downside risk, making the analysis both rigorous and practical. 

5. Implementation 

Tools: Python, NumPy, Pandas, yfinance, Matplotlib, SciPy. 

Technologies: Open-source libraries for data handling, optimization, and visualization. 

Workflow: End-to-end pipeline from data collection to risk visualization, ensuring reproducibility and clarity. 

6. Results 

Optimal Weights: ~63% AAPL, ~37% AMZN, negligible in other assets. 

Efficient Frontier Visualization: Demonstrates the trade-off between risk and return, with the optimal portfolio lying at the maximum Sharpe ratio point. 

VaR Results (95%): Portfolio unlikely to lose more than ~3% in a single day across all three methods. 

Interpretation: The optimizer concentrated on Apple and Amazon due to superior risk-adjusted returns. VaR results confirm stable downside risk estimates. 

7. Insights & Applications 

Business/Financial Relevance: Shows how investors can use quantitative methods to design efficient portfolios. 

Real-World Use Case: Asset managers can apply similar workflows to construct portfolios, assess risk, and communicate findings to clients. 

Key Insight: Diversification is not about equal allocation but about selecting assets that maximize efficiency. 

8. Conclusion 

Key Takeaways: The optimal portfolio is concentrated in AAPL and AMZN, with ~3% daily downside risk at 95% confidence. The efficient frontier demonstrates diversification benefits, and VaR provides reliable risk estimates. 
