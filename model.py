import numpy as np
from scipy.special import factorial
from math import gamma
from scipy.optimize import root_scalar
from analysis import *

class OU_Trading_Model:
    """
    Ornstein-Uhlenbeck Trading Model (OU Model)

    This class implements a trading model based on the Ornstein-Uhlenbeck process. It is designed for mean-reversion 
    strategies using pairs of assets. The model identifies trading thresholds and executes trades based on deviations 
    from the mean.

    Attributes:
        capital (float): Initial capital for the trading model.
        update_capital (bool): Indicates whether to update the capital automatically(for test mode).
        dt (float): Time step size (default is 1/252, representing daily data in a trading year).
        c (float): Transaction cost (default is 0.007).
        is_long (bool): Indicates the current trade direction (long/short).
        beta (float): Hedge ratio between the two assets.
        amount_a, amount_b (float): Amounts of each asset held in the portfolio.
        p_a, p_b (float): Current value of the portfolio in terms of each asset.
        u_threshold, l_threshold (float): Upper and lower thresholds for trading signals.
    """

    def __init__(self,capital,update_capital=False,dt=1/252,cost=0.007):
        """
        Initialize the trading model with the given parameters.
        
        Args:
            capital (float): Initial capital.
            dt (float, optional): Time step size. Default is 1/252.
            cost (float, optional): Transaction cost per trade. Default is 0.007.
        """
        self.c = cost  # Transaction cost
        self.is_long = None  # Trade direction (long/short)
        self.capital = capital  # Initial capital
        self.p_a = None  # Portfolio value in asset A
        self.p_b = None  # Portfolio value in asset B
        self.amount_a = None  # Quantity of asset A
        self.amount_b = None  # Quantity of asset B
        self.dt = dt  # Time step size
        self.update_capital = update_capital # Automatic capital updates

    def setup(self, price_data):
        """
        Initialize the model parameters and thresholds based on historical price data.

        Args:
            price_data (list): Historical price data for two assets, where each element is a list of prices.
        """
        self.init_prices = price_data[0][0], price_data[1][0]  # Initial prices of both assets
        params = self.calculate_likelihoods(price_data)  # Compute OU process parameters
        self.beta = params['beta']  # Hedge ratio
        self._set_thresholds(params['mu'], params['sigma'], params['theta'], self.c)

    def trade(self, prices):
        """
        Execute trading logic based on the current prices of the assets.

        Args:
            prices (list): Current prices of the two assets.
        
        Raises:
            Exception: If the model is not set up with initial prices.
        """
        signal = False 
        if self.init_prices is None:
            raise Exception("Model not setup")

        # Calculate the spread index based on initial prices and hedge ratio
        self.index = prices[0] / self.init_prices[0] - self.beta * prices[1] / self.init_prices[1]

        # Update portfolio value based on price changes and current positions
        if self.amount_a is not None and self.amount_b is not None:
            if self.is_long:  # Long position
                self.p_a += (prices[0] - self._last_a) * self.amount_a
                self.p_b += (self._last_b - prices[1]) * self.amount_b
            else:  # Short position
                self.p_a += (self._last_a - prices[0]) * self.amount_a
                self.p_b += (prices[1] - self._last_b) * self.amount_b

        # Update total capital 
        if self.update_capital and self.p_a is not None and self.p_b is not None:
            self.capital = self.p_a + self.p_b

        # Check if trading thresholds are crossed and adjust positions
        if self.index <= self.l_threshold and (self.is_long is None or not self.is_long):
            self.recalculate_amounts(prices)
            signal = True
            self.is_long = True
        if self.index >= self.u_threshold and (self.is_long is None or self.is_long):
            self.recalculate_amounts(prices)
            signal = True
            self.is_long = False

        # Store the last prices
        self._last_a, self._last_b = prices

        # New variable to track order signals 
        return signal 

    def recalculate_amounts(self, prices):
        """
        Recalculate the amounts of each asset to be held based on the current prices.

        Args:
            prices (list): Current prices of the two assets.
        """
        self.capital -= self.c if self.update_capital else 0 # Deduct transaction cost 
        self.amount_a = self.capital / (prices[0] + self.beta * prices[1])
        self.amount_b = self.beta * self.amount_a
        if self.is_long is None:  # Initialize portfolio values
            self.p_a = self.amount_a * prices[0]
            self.p_b = self.amount_b * prices[1]

    def _set_thresholds(self, mu, sigma, theta, c):
        """
        Compute and set the upper and lower thresholds for trading.

        Args:
            mu (float): Mean reversion level.
            sigma (float): Volatility of the spread.
            theta (float): Speed of mean reversion.
            c (float): Transaction cost.
        """
        def solve_for_a(c, N=100):
            """
            Solve for the threshold scaling factor 'a' using numerical methods.
            """
            def lhs(a):
                s = [(np.sqrt(2) * a)**(2 * n + 1) / factorial(2 * n + 1) * gamma((2 * n + 1) / 2) for n in range(N)]
                return np.sum(s) / 2

            def rhs(a):
                s = [(np.sqrt(2) * a)**(2 * n) / factorial(2 * n) * gamma((2 * n + 1) / 2) for n in range(N)]
                return np.sum(s) * np.sqrt(2) / 2 * (a - c / 2)

            def equation(a):
                return lhs(a) - rhs(a)

            res = root_scalar(equation, bracket=[1e-2, 10], method='brentq')  # Numerical solver
            return res.root

        a_star = solve_for_a(c)
        self.u_threshold = mu + a_star * sigma / np.sqrt(2 * theta)
        self.l_threshold = mu - a_star * sigma / np.sqrt(2 * theta)

    def calculate_likelihoods(self, price_data, dt=None):
        """
        Calculate the optimal parameters (beta, theta, mu, sigma) based on historical price data.

        Args:
            price_data (list): Historical price data for two assets.
            dt (float, optional): Time step size. If not provided, uses the model's default.

        Returns:
            dict: Contains the optimal beta, theta, mu, sigma, and log-likelihood.
        """
        if dt is None:
            dt = self.dt

        # B/A Ratios to Test
        B_over_A = np.linspace(0.001, 1, 1000)

        # likelihood and param dictionaries
        results = {}

        likelihoods = []

        # Calculate likelihood for each B/A ratio
        for value in B_over_A:
            alpha = 1 / price_data[0][0] # calculate the weight of the first asset
            beta = value / price_data[1][0] # weight of second asset
            spread = alpha * price_data[0] - beta * price_data[1] #s pread claculation
            parameters = calculate_parameters(spread, dt) # get the ou paprams - theta , mu sigma
            likelihood = log_likelihood(parameters, spread, dt)

            likelihoods.append(likelihood)

        # Find the optimal B for this pair

        ind = np.argmax(likelihoods)
        max_likelihood = likelihoods[ind]

        optimal_B = B_over_A[ind] # get corresponding beta value

        # Recalculate parameters for the optimal B
        alpha = 1 / price_data[0][0]
        beta = optimal_B / price_data[1][0]
        spread = alpha * price_data[0] - beta * price_data[1]
        mu, theta, sigma = calculate_parameters(spread, dt)

        # Store results
        results = {
            'beta': optimal_B,
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'log_l': max_likelihood,
        }
        return results