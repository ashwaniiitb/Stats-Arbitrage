"""
Statistical Arbitrage: Optimal Hedge Ratio Analysis
====================================================
This script estimates the optimal long-short position sizing between S&P 500 (MESZ5) 
and NASDAQ (MNQZ5) futures using rolling returns, beta estimation, and econometric tests.

Methodology:
1. Rolling beta estimation via OLS regression
2. Rolling correlation analysis
3. Optimal hedge ratio via variance minimization
4. Cointegration testing (Engle-Granger)
5. Stationarity testing (ADF, KPSS)
6. Mean reversion testing (Half-life)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Statistical and econometric libraries
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class StatisticalArbitrage:
    """
    A class for conducting statistical arbitrage analysis between two cointegrated assets.
    """
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.spx_data = None
        self.ndx_data = None
        self.combined_data = None
        
    def load_data(self):
        """Load and combine all CSV files for both instruments."""
        print("Loading data from directory...")
        
        spx_files = []
        ndx_files = []
        
        # Iterate through all subdirectories
        for subdir in sorted(self.data_dir.iterdir()):
            if subdir.is_dir():
                # S&P 500 files
                spx_file = list(subdir.glob("MESZ5_*.csv"))
                if spx_file:
                    df = pd.read_csv(spx_file[0])
                    # Clean time format: "09:30:00:000:000:000" -> "09:30:00"
                    df['Time'] = df['Time'].str.split(':').str[:3].str.join(':')
                    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                    df = df[['DateTime', 'close']].rename(columns={'close': 'SPX'})
                    spx_files.append(df)
                
                # NASDAQ files
                ndx_file = list(subdir.glob("MNQZ5_*.csv"))
                if ndx_file:
                    df = pd.read_csv(ndx_file[0])
                    # Clean time format: "09:30:00:000:000:000" -> "09:30:00"
                    df['Time'] = df['Time'].str.split(':').str[:3].str.join(':')
                    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                    df = df[['DateTime', 'close']].rename(columns={'close': 'NDX'})
                    ndx_files.append(df)
        
        # Concatenate all files
        self.spx_data = pd.concat(spx_files, ignore_index=True).sort_values('DateTime')
        self.ndx_data = pd.concat(ndx_files, ignore_index=True).sort_values('DateTime')
        
        # Merge on timestamp
        self.combined_data = pd.merge(self.spx_data, self.ndx_data, on='DateTime', how='inner')
        self.combined_data.set_index('DateTime', inplace=True)
        
        print(f"Loaded {len(self.combined_data)} observations")
        print(f"Date range: {self.combined_data.index.min()} to {self.combined_data.index.max()}")
        print(f"SPX range: {self.combined_data['SPX'].min():.2f} to {self.combined_data['SPX'].max():.2f}")
        print(f"NDX range: {self.combined_data['NDX'].min():.2f} to {self.combined_data['NDX'].max():.2f}")
        
        return self.combined_data
    
    def calculate_returns(self, window='60s'):
        """
        Calculate returns at specified frequency.
        Default: 60-second returns to reduce noise in high-frequency data.
        """
        print(f"\nCalculating returns with {window} resampling...")
        
        # Resample to reduce noise
        resampled = self.combined_data.resample(window).last().dropna()
        
        # Calculate log returns
        resampled['SPX_Return'] = np.log(resampled['SPX'] / resampled['SPX'].shift(1))
        resampled['NDX_Return'] = np.log(resampled['NDX'] / resampled['NDX'].shift(1))
        
        # Calculate simple returns for comparison
        resampled['SPX_Return_Simple'] = resampled['SPX'].pct_change()
        resampled['NDX_Return_Simple'] = resampled['NDX'].pct_change()
        
        self.combined_data = resampled.dropna()
        
        print(f"Resampled to {len(self.combined_data)} observations")
        print(f"Mean SPX return: {self.combined_data['SPX_Return'].mean()*10000:.4f} bps")
        print(f"Mean NDX return: {self.combined_data['NDX_Return'].mean()*10000:.4f} bps")
        print(f"Std SPX return: {self.combined_data['SPX_Return'].std()*10000:.4f} bps")
        print(f"Std NDX return: {self.combined_data['NDX_Return'].std()*10000:.4f} bps")
        
        return self.combined_data
    
    def calculate_rolling_statistics(self, window=120):
        """
        Calculate rolling beta, correlation, and optimal hedge ratio.
        
        Parameters:
        -----------
        window : int
            Rolling window size in observations
        """
        print(f"\nCalculating rolling statistics with window={window}...")
        
        df = self.combined_data.copy()
        
        # Rolling correlation
        df['Rolling_Corr'] = df['SPX_Return'].rolling(window).corr(df['NDX_Return'])
        
        # Rolling beta (SPX = dependent, NDX = independent)
        # Beta = Cov(SPX, NDX) / Var(NDX)
        rolling_cov = df['SPX_Return'].rolling(window).cov(df['NDX_Return'])
        rolling_var_ndx = df['NDX_Return'].rolling(window).var()
        df['Rolling_Beta'] = rolling_cov / rolling_var_ndx
        
        # Optimal hedge ratio via variance minimization
        # h* = Cov(SPX, NDX) / Var(NDX) = Beta
        df['Hedge_Ratio_MinVar'] = df['Rolling_Beta']
        
        # Alternative: Variance ratio method
        rolling_var_spx = df['SPX_Return'].rolling(window).var()
        df['Hedge_Ratio_VarRatio'] = np.sqrt(rolling_var_spx / rolling_var_ndx)
        
        # Rolling R-squared
        df['Rolling_R2'] = df['Rolling_Corr'] ** 2
        
        self.combined_data = df
        
        print(f"Mean rolling correlation: {df['Rolling_Corr'].mean():.4f}")
        print(f"Mean rolling beta: {df['Rolling_Beta'].mean():.4f}")
        print(f"Mean hedge ratio (MinVar): {df['Hedge_Ratio_MinVar'].mean():.4f}")
        print(f"Mean R-squared: {df['Rolling_R2'].mean():.4f}")
        
        return df
    
    def estimate_optimal_hedge_ratio_ols(self):
        """
        Estimate optimal hedge ratio using full-sample OLS regression.
        SPX = alpha + beta * NDX + epsilon
        """
        print("\n" + "="*70)
        print("OPTIMAL HEDGE RATIO ESTIMATION - OLS REGRESSION")
        print("="*70)
        
        df = self.combined_data.dropna()
        
        # Method 1: Returns-based regression
        X = df['NDX_Return'].values.reshape(-1, 1)
        y = df['SPX_Return'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        beta_returns = model.coef_[0]
        alpha_returns = model.intercept_
        r2_returns = model.score(X, y)
        
        print("\n1. RETURNS-BASED REGRESSION:")
        print(f"   SPX_Return = {alpha_returns:.6f} + {beta_returns:.4f} * NDX_Return")
        print(f"   R-squared: {r2_returns:.4f}")
        print(f"   Interpretation: For 1% move in NDX, SPX moves {beta_returns:.4f}%")
        
        # Method 2: Price-based regression (levels)
        X_price = df['NDX'].values.reshape(-1, 1)
        y_price = df['SPX'].values
        
        model_price = LinearRegression()
        model_price.fit(X_price, y_price)
        
        beta_price = model_price.coef_[0]
        alpha_price = model_price.intercept_
        r2_price = model_price.score(X_price, y_price)
        
        print("\n2. PRICE-BASED REGRESSION:")
        print(f"   SPX = {alpha_price:.2f} + {beta_price:.4f} * NDX")
        print(f"   R-squared: {r2_price:.4f}")
        print(f"   Interpretation: Hedge ratio = {beta_price:.4f}")
        
        # Store results
        self.ols_results = {
            'beta_returns': beta_returns,
            'alpha_returns': alpha_returns,
            'r2_returns': r2_returns,
            'beta_price': beta_price,
            'alpha_price': alpha_price,
            'r2_price': r2_price
        }
        
        return self.ols_results
    
    def construct_spread(self, method='ols_returns'):
        """
        Construct the mean-reverting spread using optimal hedge ratio.
        
        Parameters:
        -----------
        method : str
            'ols_returns': Use beta from returns regression
            'ols_price': Use beta from price regression
            'rolling': Use rolling hedge ratio
        """
        print(f"\nConstructing spread using method: {method}")
        
        df = self.combined_data.copy()
        
        if method == 'ols_returns':
            beta = self.ols_results['beta_returns']
            df['Spread'] = df['SPX_Return'] - beta * df['NDX_Return']
            spread_name = f"Spread (β_ret={beta:.4f})"
            
        elif method == 'ols_price':
            beta = self.ols_results['beta_price']
            alpha = self.ols_results['alpha_price']
            df['Spread'] = df['SPX'] - (alpha + beta * df['NDX'])
            spread_name = f"Spread (β_price={beta:.4f})"
            
        elif method == 'rolling':
            df['Spread'] = df['SPX_Return'] - df['Hedge_Ratio_MinVar'] * df['NDX_Return']
            spread_name = "Spread (Rolling β)"
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.combined_data = df
        self.spread_method = method
        
        print(f"Spread constructed: {spread_name}")
        print(f"Mean: {df['Spread'].mean():.6f}")
        print(f"Std: {df['Spread'].std():.6f}")
        print(f"Min: {df['Spread'].min():.6f}")
        print(f"Max: {df['Spread'].max():.6f}")
        
        return df
    
    def test_stationarity(self):
        """
        Test spread for stationarity using ADF and KPSS tests.
        """
        print("\n" + "="*70)
        print("STATIONARITY TESTS")
        print("="*70)
        
        spread = self.combined_data['Spread'].dropna()
        
        # Augmented Dickey-Fuller Test
        print("\n1. AUGMENTED DICKEY-FULLER TEST (ADF)")
        print("   H0: Series has a unit root (non-stationary)")
        print("   H1: Series is stationary")
        
        adf_result = adfuller(spread, autolag='AIC')
        
        print(f"\n   ADF Statistic: {adf_result[0]:.6f}")
        print(f"   p-value: {adf_result[1]:.6f}")
        print(f"   Critical Values:")
        for key, value in adf_result[4].items():
            print(f"      {key}: {value:.4f}")
        
        if adf_result[1] < 0.05:
            print(f"\n   ✓ REJECT H0: Spread IS stationary at 5% level (p={adf_result[1]:.4f})")
        else:
            print(f"\n   ✗ FAIL TO REJECT H0: Spread may NOT be stationary (p={adf_result[1]:.4f})")
        
        # KPSS Test
        print("\n2. KPSS TEST")
        print("   H0: Series is stationary")
        print("   H1: Series has a unit root")
        
        kpss_result = kpss(spread, regression='c', nlags='auto')
        
        print(f"\n   KPSS Statistic: {kpss_result[0]:.6f}")
        print(f"   p-value: {kpss_result[1]:.6f}")
        print(f"   Critical Values:")
        for key, value in kpss_result[3].items():
            print(f"      {key}: {value:.4f}")
        
        if kpss_result[1] > 0.05:
            print(f"\n   ✓ FAIL TO REJECT H0: Spread IS stationary at 5% level (p={kpss_result[1]:.4f})")
        else:
            print(f"\n   ✗ REJECT H0: Spread may NOT be stationary (p={kpss_result[1]:.4f})")
        
        # Store results
        self.stationarity_results = {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'adf_critical_values': adf_result[4],
            'kpss_statistic': kpss_result[0],
            'kpss_pvalue': kpss_result[1],
            'kpss_critical_values': kpss_result[3]
        }
        
        return self.stationarity_results
    
    def test_cointegration(self):
        """
        Test for cointegration between SPX and NDX using Engle-Granger test.
        """
        print("\n" + "="*70)
        print("COINTEGRATION TEST (ENGLE-GRANGER)")
        print("="*70)
        
        spx = self.combined_data['SPX'].dropna().values
        ndx = self.combined_data['NDX'].dropna().values
        
        # Engle-Granger cointegration test
        coint_t, pvalue, crit_vals = coint(spx, ndx)
        
        print("\n   H0: No cointegration")
        print("   H1: Series are cointegrated")
        print(f"\n   Test Statistic: {coint_t:.6f}")
        print(f"   p-value: {pvalue:.6f}")
        print(f"   Critical Values:")
        print(f"      1%: {crit_vals[0]:.4f}")
        print(f"      5%: {crit_vals[1]:.4f}")
        print(f"      10%: {crit_vals[2]:.4f}")
        
        if pvalue < 0.05:
            print(f"\n   ✓ REJECT H0: SPX and NDX ARE cointegrated at 5% level (p={pvalue:.4f})")
        else:
            print(f"\n   ✗ FAIL TO REJECT H0: No evidence of cointegration (p={pvalue:.4f})")
        
        self.cointegration_results = {
            'test_statistic': coint_t,
            'pvalue': pvalue,
            'critical_values': crit_vals
        }
        
        return self.cointegration_results
    
    def calculate_half_life(self):
        """
        Calculate half-life of mean reversion for the spread.
        Uses AR(1) model: spread(t) = α + φ * spread(t-1) + ε
        Half-life = -log(2) / log(φ)
        """
        print("\n" + "="*70)
        print("MEAN REVERSION ANALYSIS (HALF-LIFE)")
        print("="*70)
        
        spread = self.combined_data['Spread'].dropna().values
        spread_lag = np.roll(spread, 1)[1:]
        spread = spread[1:]
        
        # Fit AR(1) model
        X = add_constant(spread_lag)
        model = OLS(spread, X).fit()
        
        phi = model.params[1]
        alpha = model.params[0]
        
        print(f"\n   AR(1) Model: spread(t) = {alpha:.6f} + {phi:.6f} * spread(t-1)")
        print(f"   R-squared: {model.rsquared:.4f}")
        print(f"   φ coefficient: {phi:.6f}")
        
        if phi >= 1:
            print(f"\n   ✗ φ >= 1: Spread does NOT mean-revert (random walk or explosive)")
            half_life = np.inf
        elif phi <= 0:
            print(f"\n   ✗ φ <= 0: Spread shows oscillatory behavior")
            half_life = -np.log(2) / np.log(abs(phi))
        else:
            half_life = -np.log(2) / np.log(phi)
            print(f"\n   ✓ Half-life of mean reversion: {half_life:.2f} periods")
            print(f"   Interpretation: Spread reverts to mean in ~{half_life:.0f} observations")
        
        self.half_life_results = {
            'phi': phi,
            'alpha': alpha,
            'r2': model.rsquared,
            'half_life': half_life
        }
        
        return self.half_life_results
    
    def determine_position_sizing(self, initial_capital=100000):
        """
        Determine optimal long-short position sizing based on hedge ratio.
        
        Position sizing logic:
        - If beta < 1: Go LONG α% SPX, SHORT β% NDX (where β = hedge_ratio)
        - If beta > 1: Go LONG α% NDX, SHORT β% SPX (inverted)
        """
        print("\n" + "="*70)
        print("OPTIMAL POSITION SIZING")
        print("="*70)
        
        beta = self.ols_results['beta_returns']
        
        print(f"\n   Initial Capital: ${initial_capital:,.2f}")
        print(f"   Hedge Ratio (β): {beta:.4f}")
        
        # Current prices
        current_spx = self.combined_data['SPX'].iloc[-1]
        current_ndx = self.combined_data['NDX'].iloc[-1]
        
        print(f"\n   Current Prices:")
        print(f"      SPX (MESZ5): {current_spx:.2f}")
        print(f"      NDX (MNQZ5): {current_ndx:.2f}")
        
        # Position sizing
        # For mean-reverting spread: SPX_Return - β * NDX_Return
        # We want to maintain dollar neutrality with hedge ratio adjustment
        
        if beta < 1:
            # SPX is less volatile, use it as base
            spx_allocation = initial_capital / (1 + beta)
            ndx_allocation = beta * spx_allocation
            
            spx_position = spx_allocation / current_spx  # Long
            ndx_position = -ndx_allocation / current_ndx  # Short
            
            print(f"\n   Strategy: LONG SPX, SHORT NDX")
            print(f"      LONG ${spx_allocation:,.2f} in SPX ({spx_position:.4f} contracts)")
            print(f"      SHORT ${ndx_allocation:,.2f} in NDX ({abs(ndx_position):.4f} contracts)")
            
        else:
            # NDX is less volatile, use it as base
            ndx_allocation = initial_capital / (1 + 1/beta)
            spx_allocation = ndx_allocation / beta
            
            spx_position = spx_allocation / current_spx  # Long
            ndx_position = -ndx_allocation / current_ndx  # Short
            
            print(f"\n   Strategy: LONG SPX, SHORT NDX (adjusted)")
            print(f"      LONG ${spx_allocation:,.2f} in SPX ({spx_position:.4f} contracts)")
            print(f"      SHORT ${ndx_allocation:,.2f} in NDX ({abs(ndx_position):.4f} contracts)")
        
        # Calculate spread value
        spread_value = spx_position * current_spx + ndx_position * current_ndx
        
        print(f"\n   Net Position Value: ${spread_value:,.2f}")
        print(f"   Leverage: {(spx_allocation + abs(ndx_allocation))/initial_capital:.2f}x")
        
        self.position_sizing = {
            'initial_capital': initial_capital,
            'beta': beta,
            'spx_allocation': spx_allocation,
            'ndx_allocation': ndx_allocation,
            'spx_position': spx_position,
            'ndx_position': ndx_position,
            'current_spx': current_spx,
            'current_ndx': current_ndx,
            'spread_value': spread_value
        }
        
        return self.position_sizing
    
    def backtest_strategy(self, initial_capital=100000, z_entry=2.0, z_exit=0.5):
        """
        Simple backtest of the statistical arbitrage strategy.
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital
        z_entry : float
            Z-score threshold for entry
        z_exit : float
            Z-score threshold for exit
        """
        print("\n" + "="*70)
        print("STRATEGY BACKTEST")
        print("="*70)
        
        df = self.combined_data.copy()
        
        # Calculate z-score of spread
        spread = df['Spread']
        spread_mean = spread.rolling(120).mean()
        spread_std = spread.rolling(120).std()
        df['Z_Score'] = (spread - spread_mean) / spread_std
        
        # Generate signals
        df['Signal'] = 0
        df.loc[df['Z_Score'] > z_entry, 'Signal'] = -1  # Short spread (overbought)
        df.loc[df['Z_Score'] < -z_entry, 'Signal'] = 1   # Long spread (oversold)
        df.loc[abs(df['Z_Score']) < z_exit, 'Signal'] = 0  # Exit
        
        # Calculate strategy returns
        df['Strategy_Return'] = df['Signal'].shift(1) * df['Spread']
        df['Cumulative_Return'] = df['Strategy_Return'].cumsum()
        df['Portfolio_Value'] = initial_capital * (1 + df['Cumulative_Return'])
        
        # Calculate performance metrics
        total_return = df['Portfolio_Value'].iloc[-1] / initial_capital - 1
        sharpe_ratio = df['Strategy_Return'].mean() / df['Strategy_Return'].std() * np.sqrt(252 * 6.5 * 60)  # Annualized
        max_drawdown = (df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1).min()
        
        num_trades = (df['Signal'].diff() != 0).sum()
        
        print(f"\n   Backtest Parameters:")
        print(f"      Initial Capital: ${initial_capital:,.2f}")
        print(f"      Entry Threshold: ±{z_entry} std devs")
        print(f"      Exit Threshold: ±{z_exit} std devs")
        
        print(f"\n   Performance Metrics:")
        print(f"      Total Return: {total_return*100:.2f}%")
        print(f"      Final Portfolio Value: ${df['Portfolio_Value'].iloc[-1]:,.2f}")
        print(f"      Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"      Max Drawdown: {max_drawdown*100:.2f}%")
        print(f"      Number of Trades: {num_trades}")
        
        self.backtest_results = {
            'initial_capital': initial_capital,
            'total_return': total_return,
            'final_value': df['Portfolio_Value'].iloc[-1],
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades
        }
        
        self.combined_data = df
        
        return self.backtest_results
    
    def create_visualizations(self, save_path='optimal_hedge_analysis.png'):
        """
        Create comprehensive visualizations of the analysis.
        """
        print(f"\nCreating visualizations...")
        
        df = self.combined_data
        
        fig = plt.figure(figsize=(20, 14))
        
        # 1. Price series
        ax1 = plt.subplot(4, 3, 1)
        ax1_twin = ax1.twinx()
        ax1.plot(df.index, df['SPX'], 'b-', label='SPX (MESZ5)', linewidth=1)
        ax1_twin.plot(df.index, df['NDX'], 'r-', label='NDX (MNQZ5)', linewidth=1)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('SPX Price', color='b')
        ax1_twin.set_ylabel('NDX Price', color='r')
        ax1.set_title('Price Series: S&P 500 vs NASDAQ Futures')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. Returns scatter plot
        ax2 = plt.subplot(4, 3, 2)
        ax2.scatter(df['NDX_Return'], df['SPX_Return'], alpha=0.3, s=1)
        
        # Add regression line
        z = np.polyfit(df['NDX_Return'].dropna(), df['SPX_Return'].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['NDX_Return'].min(), df['NDX_Return'].max(), 100)
        ax2.plot(x_line, p(x_line), "r-", linewidth=2, 
                label=f'β = {self.ols_results["beta_returns"]:.4f}')
        ax2.set_xlabel('NDX Returns')
        ax2.set_ylabel('SPX Returns')
        ax2.set_title(f'Returns Relationship (R² = {self.ols_results["r2_returns"]:.4f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling correlation
        ax3 = plt.subplot(4, 3, 3)
        ax3.plot(df.index, df['Rolling_Corr'], 'g-', linewidth=1)
        ax3.axhline(df['Rolling_Corr'].mean(), color='r', linestyle='--', 
                   label=f'Mean = {df["Rolling_Corr"].mean():.4f}')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Correlation')
        ax3.set_title('Rolling Correlation (120-period)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0.5, 1.0])
        
        # 4. Rolling beta
        ax4 = plt.subplot(4, 3, 4)
        ax4.plot(df.index, df['Rolling_Beta'], 'b-', linewidth=1)
        ax4.axhline(df['Rolling_Beta'].mean(), color='r', linestyle='--',
                   label=f'Mean = {df["Rolling_Beta"].mean():.4f}')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Beta')
        ax4.set_title('Rolling Beta (SPX vs NDX)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Hedge ratio
        ax5 = plt.subplot(4, 3, 5)
        ax5.plot(df.index, df['Hedge_Ratio_MinVar'], 'purple', linewidth=1, label='Min Variance')
        ax5.axhline(self.ols_results['beta_returns'], color='r', linestyle='--',
                   label=f'OLS β = {self.ols_results["beta_returns"]:.4f}')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Hedge Ratio')
        ax5.set_title('Optimal Hedge Ratio (Dynamic)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. R-squared
        ax6 = plt.subplot(4, 3, 6)
        ax6.plot(df.index, df['Rolling_R2'], 'orange', linewidth=1)
        ax6.axhline(df['Rolling_R2'].mean(), color='r', linestyle='--',
                   label=f'Mean = {df["Rolling_R2"].mean():.4f}')
        ax6.set_xlabel('Time')
        ax6.set_ylabel('R²')
        ax6.set_title('Rolling R-squared')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, 1])
        
        # 7. Spread time series
        ax7 = plt.subplot(4, 3, 7)
        ax7.plot(df.index, df['Spread'], 'k-', linewidth=0.5, alpha=0.7)
        spread_mean = df['Spread'].mean()
        spread_std = df['Spread'].std()
        ax7.axhline(spread_mean, color='r', linestyle='--', label='Mean')
        ax7.axhline(spread_mean + 2*spread_std, color='g', linestyle=':', label='±2σ')
        ax7.axhline(spread_mean - 2*spread_std, color='g', linestyle=':')
        ax7.set_xlabel('Time')
        ax7.set_ylabel('Spread Value')
        ax7.set_title(f'Mean-Reverting Spread (β={self.ols_results["beta_returns"]:.4f})')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Spread distribution
        ax8 = plt.subplot(4, 3, 8)
        ax8.hist(df['Spread'].dropna(), bins=100, density=True, alpha=0.7, color='blue')
        
        # Overlay normal distribution
        mu, sigma = df['Spread'].mean(), df['Spread'].std()
        x = np.linspace(df['Spread'].min(), df['Spread'].max(), 100)
        ax8.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
        ax8.set_xlabel('Spread Value')
        ax8.set_ylabel('Density')
        ax8.set_title('Spread Distribution')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Z-score of spread
        ax9 = plt.subplot(4, 3, 9)
        if 'Z_Score' in df.columns:
            ax9.plot(df.index, df['Z_Score'], 'purple', linewidth=0.5, alpha=0.7)
            ax9.axhline(0, color='black', linestyle='-', linewidth=1)
            ax9.axhline(2, color='r', linestyle='--', label='Entry (±2σ)')
            ax9.axhline(-2, color='r', linestyle='--')
            ax9.axhline(0.5, color='g', linestyle=':', label='Exit (±0.5σ)')
            ax9.axhline(-0.5, color='g', linestyle=':')
            ax9.set_xlabel('Time')
            ax9.set_ylabel('Z-Score')
            ax9.set_title('Spread Z-Score (Standardized)')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
        
        # 10. Portfolio value
        ax10 = plt.subplot(4, 3, 10)
        if 'Portfolio_Value' in df.columns:
            ax10.plot(df.index, df['Portfolio_Value'], 'darkgreen', linewidth=1.5)
            ax10.axhline(self.backtest_results['initial_capital'], color='r', 
                        linestyle='--', label='Initial Capital')
            ax10.set_xlabel('Time')
            ax10.set_ylabel('Portfolio Value ($)')
            ax10.set_title(f'Strategy Performance (Return: {self.backtest_results["total_return"]*100:.2f}%)')
            ax10.legend()
            ax10.grid(True, alpha=0.3)
        
        # 11. Drawdown
        ax11 = plt.subplot(4, 3, 11)
        if 'Portfolio_Value' in df.columns:
            drawdown = (df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1) * 100
            ax11.fill_between(df.index, drawdown, 0, color='red', alpha=0.5)
            ax11.set_xlabel('Time')
            ax11.set_ylabel('Drawdown (%)')
            ax11.set_title(f'Underwater Plot (Max DD: {self.backtest_results["max_drawdown"]*100:.2f}%)')
            ax11.grid(True, alpha=0.3)
        
        # 12. Trading signals
        ax12 = plt.subplot(4, 3, 12)
        if 'Signal' in df.columns:
            signals = df['Signal'].copy()
            ax12.plot(df.index, signals, 'b-', linewidth=0.8, alpha=0.7)
            ax12.fill_between(df.index, signals, 0, where=(signals > 0), 
                             color='green', alpha=0.3, label='Long Spread')
            ax12.fill_between(df.index, signals, 0, where=(signals < 0), 
                             color='red', alpha=0.3, label='Short Spread')
            ax12.set_xlabel('Time')
            ax12.set_ylabel('Signal')
            ax12.set_title('Trading Signals')
            ax12.legend()
            ax12.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
        plt.close()
    
    def generate_report(self, save_path='optimal_hedge_report.txt'):
        """
        Generate a comprehensive text report.
        """
        print(f"\nGenerating report...")
        
        report = []
        report.append("="*80)
        report.append("STATISTICAL ARBITRAGE: OPTIMAL HEDGE RATIO ANALYSIS")
        report.append("S&P 500 (MESZ5) vs NASDAQ (MNQZ5) Futures")
        report.append("="*80)
        report.append("")
        
        # Data summary
        report.append("1. DATA SUMMARY")
        report.append("-" * 80)
        report.append(f"   Total Observations: {len(self.combined_data):,}")
        report.append(f"   Date Range: {self.combined_data.index.min()} to {self.combined_data.index.max()}")
        report.append(f"   SPX Price Range: {self.combined_data['SPX'].min():.2f} - {self.combined_data['SPX'].max():.2f}")
        report.append(f"   NDX Price Range: {self.combined_data['NDX'].min():.2f} - {self.combined_data['NDX'].max():.2f}")
        report.append("")
        
        # OLS results
        report.append("2. OPTIMAL HEDGE RATIO (OLS REGRESSION)")
        report.append("-" * 80)
        report.append(f"   Returns-Based Beta: {self.ols_results['beta_returns']:.6f}")
        report.append(f"   Returns-Based R²: {self.ols_results['r2_returns']:.6f}")
        report.append(f"   Price-Based Beta: {self.ols_results['beta_price']:.6f}")
        report.append(f"   Price-Based R²: {self.ols_results['r2_price']:.6f}")
        report.append("")
        report.append(f"   Interpretation:")
        report.append(f"      - For every 1% move in NDX, SPX moves {self.ols_results['beta_returns']:.4f}%")
        report.append(f"      - Optimal hedge: {self.ols_results['beta_returns']:.4f} units of NDX per unit of SPX")
        report.append("")
        
        # Rolling statistics
        report.append("3. ROLLING STATISTICS (120-period window)")
        report.append("-" * 80)
        report.append(f"   Mean Rolling Correlation: {self.combined_data['Rolling_Corr'].mean():.6f}")
        report.append(f"   Mean Rolling Beta: {self.combined_data['Rolling_Beta'].mean():.6f}")
        report.append(f"   Mean Rolling R²: {self.combined_data['Rolling_R2'].mean():.6f}")
        report.append("")
        
        # Cointegration
        report.append("4. COINTEGRATION TEST")
        report.append("-" * 80)
        report.append(f"   Test Statistic: {self.cointegration_results['test_statistic']:.6f}")
        report.append(f"   P-value: {self.cointegration_results['pvalue']:.6f}")
        if self.cointegration_results['pvalue'] < 0.05:
            report.append(f"   Result: ✓ COINTEGRATED (p < 0.05)")
        else:
            report.append(f"   Result: ✗ NOT COINTEGRATED (p >= 0.05)")
        report.append("")
        
        # Stationarity
        report.append("5. STATIONARITY TESTS (Spread)")
        report.append("-" * 80)
        report.append(f"   ADF Test:")
        report.append(f"      Statistic: {self.stationarity_results['adf_statistic']:.6f}")
        report.append(f"      P-value: {self.stationarity_results['adf_pvalue']:.6f}")
        if self.stationarity_results['adf_pvalue'] < 0.05:
            report.append(f"      Result: ✓ STATIONARY (p < 0.05)")
        else:
            report.append(f"      Result: ✗ NON-STATIONARY (p >= 0.05)")
        report.append("")
        report.append(f"   KPSS Test:")
        report.append(f"      Statistic: {self.stationarity_results['kpss_statistic']:.6f}")
        report.append(f"      P-value: {self.stationarity_results['kpss_pvalue']:.6f}")
        if self.stationarity_results['kpss_pvalue'] > 0.05:
            report.append(f"      Result: ✓ STATIONARY (p > 0.05)")
        else:
            report.append(f"      Result: ✗ NON-STATIONARY (p <= 0.05)")
        report.append("")
        
        # Mean reversion
        report.append("6. MEAN REVERSION (Half-Life)")
        report.append("-" * 80)
        report.append(f"   AR(1) Coefficient (φ): {self.half_life_results['phi']:.6f}")
        report.append(f"   Half-Life: {self.half_life_results['half_life']:.2f} periods")
        report.append(f"   Interpretation: Spread reverts 50% to mean in ~{self.half_life_results['half_life']:.0f} observations")
        report.append("")
        
        # Position sizing
        report.append("7. OPTIMAL POSITION SIZING")
        report.append("-" * 80)
        report.append(f"   Initial Capital: ${self.position_sizing['initial_capital']:,.2f}")
        report.append(f"   Hedge Ratio: {self.position_sizing['beta']:.6f}")
        report.append(f"   SPX Allocation: ${self.position_sizing['spx_allocation']:,.2f} (LONG)")
        report.append(f"   NDX Allocation: ${self.position_sizing['ndx_allocation']:,.2f} (SHORT)")
        report.append(f"   SPX Position: {self.position_sizing['spx_position']:.4f} contracts")
        report.append(f"   NDX Position: {self.position_sizing['ndx_position']:.4f} contracts")
        report.append("")
        
        # Backtest results
        report.append("8. BACKTEST PERFORMANCE")
        report.append("-" * 80)
        report.append(f"   Total Return: {self.backtest_results['total_return']*100:.2f}%")
        report.append(f"   Final Portfolio Value: ${self.backtest_results['final_value']:,.2f}")
        report.append(f"   Sharpe Ratio: {self.backtest_results['sharpe_ratio']:.4f}")
        report.append(f"   Maximum Drawdown: {self.backtest_results['max_drawdown']*100:.2f}%")
        report.append(f"   Number of Trades: {self.backtest_results['num_trades']}")
        report.append("")
        
        # Trading strategy
        report.append("9. TRADING STRATEGY")
        report.append("-" * 80)
        report.append(f"   Entry Signal: When Z-score > ±2σ")
        report.append(f"   Exit Signal: When Z-score < ±0.5σ")
        report.append(f"   Position: SPX_Return - {self.ols_results['beta_returns']:.4f} × NDX_Return")
        report.append(f"   Type: Market-neutral, mean-reverting spread")
        report.append("")
        
        # Conclusions
        report.append("10. CONCLUSIONS")
        report.append("-" * 80)
        report.append(f"   • SPX and NDX show strong correlation ({self.combined_data['Rolling_Corr'].mean():.4f})")
        
        if self.cointegration_results['pvalue'] < 0.05:
            report.append(f"   • ✓ Assets are cointegrated (suitable for pairs trading)")
        else:
            report.append(f"   • ✗ Weak cointegration evidence")
        
        if self.stationarity_results['adf_pvalue'] < 0.05:
            report.append(f"   • ✓ Spread is stationary (mean-reverting)")
        else:
            report.append(f"   • ✗ Spread may not be stationary")
        
        report.append(f"   • Optimal hedge ratio: {self.ols_results['beta_returns']:.4f}")
        report.append(f"   • Half-life of mean reversion: {self.half_life_results['half_life']:.0f} periods")
        report.append(f"   • Strategy Sharpe ratio: {self.backtest_results['sharpe_ratio']:.2f}")
        report.append("")
        
        report.append("="*80)
        
        # Write report
        report_text = "\n".join(report)
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"Saved report to: {save_path}")
        
        # Also print to console
        print("\n" + report_text)


def main():
    """Main execution function."""
    
    # Configuration
    DATA_DIR = "/Users/ashwanidubey/Desktop/mid freq/statistical arbitrage/01-Oct to 24-Oct"
    INITIAL_CAPITAL = 100000
    ROLLING_WINDOW = 120  # 2 hours at 60s frequency
    
    print("="*80)
    print("STATISTICAL ARBITRAGE: OPTIMAL HEDGE RATIO ANALYSIS")
    print("="*80)
    print(f"\nData Directory: {DATA_DIR}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"Rolling Window: {ROLLING_WINDOW} periods")
    
    # Initialize analysis
    stat_arb = StatisticalArbitrage(DATA_DIR)
    
    # Step 1: Load data
    stat_arb.load_data()
    
    # Step 2: Calculate returns (resample to 60s)
    stat_arb.calculate_returns(window='60s')
    
    # Step 3: Calculate rolling statistics
    stat_arb.calculate_rolling_statistics(window=ROLLING_WINDOW)
    
    # Step 4: Estimate optimal hedge ratio
    stat_arb.estimate_optimal_hedge_ratio_ols()
    
    # Step 5: Test for cointegration
    stat_arb.test_cointegration()
    
    # Step 6: Construct spread
    stat_arb.construct_spread(method='ols_returns')
    
    # Step 7: Test stationarity
    stat_arb.test_stationarity()
    
    # Step 8: Calculate half-life
    stat_arb.calculate_half_life()
    
    # Step 9: Determine position sizing
    stat_arb.determine_position_sizing(initial_capital=INITIAL_CAPITAL)
    
    # Step 10: Backtest strategy
    stat_arb.backtest_strategy(initial_capital=INITIAL_CAPITAL, z_entry=2.0, z_exit=0.5)
    
    # Step 11: Create visualizations
    stat_arb.create_visualizations(save_path='optimal_hedge_analysis.png')
    
    # Step 12: Generate report
    stat_arb.generate_report(save_path='optimal_hedge_report.txt')
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

