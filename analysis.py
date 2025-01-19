import numpy as np
from itertools import combinations

def log_likelihood(parameters, S, dt):
    theta = parameters[0]
    mu = parameters[1]
    sigma = parameters[2]

    sigma0 = sigma**2 * (1 - np.exp(-2*mu*dt)) / (2 * mu)
    sigma0 = np.sqrt( sigma0 )

    N = S.size

    term1 = -0.5 * np.log(2 * np.pi)
    term2 = -np.log(sigma0)

    prefactor = -1 / (2 * N * sigma0**2)
    sum_term = 0
    for i in range( 1, N ):
        x2 = S[i]
        x1 = S[i-1]

        sum_term = sum_term + (x2 - x1 * np.exp(-mu*dt) - \
                   theta * (1-np.exp(-mu*dt)))**2

    f = (term1 + term2 + prefactor * sum_term)

    return f

def calculate_parameters(x, dt):

    N =x.size

    Xx  = np.sum(x[0:-1])
    Xy  = np.sum(x[1:])
    Xxx = np.sum(x[0:-1]**2)
    Xxy = np.sum(x[0:-1] * x[1:])
    Xyy = np.sum(x[1:]**2)

    mu = (Xy * Xxx - Xx * Xxy) / (N * (Xxx - Xxy) - (Xx**2 - Xx * Xy) )

    theta = (Xxy - mu * Xx - mu * Xy + N * mu**2) / \
        (Xxx - 2 * mu * Xx + N * mu**2)
    theta = -1 / dt * np.log(theta)

    prefactor = 2 * theta / (N*(1-np.exp(-2*theta*dt)))
    term = Xyy - 2*np.exp(-theta*dt) * Xxy + np.exp(-2*theta*dt) * Xxx - 2*mu*(1-np.exp(-theta*dt)) * (Xy - Xx * np.exp(-theta*dt)) + N * mu**2 * ( 1-np.exp(-theta * dt))**2

    sigma02 = prefactor * term

    sigma02 = max(sigma02, 1e-10)   
    sigma = np.sqrt(sigma02)

    return mu, theta, sigma

def analyze_tickers(price_data,dt):
    tickers = price_data.keys()

    #B/A Ratios to Test
    B_over_A = np.linspace(0.001, 1, 1000)

    #likelihood and param dictionaries
    results = []

    #calc Optimal B and params each pair
    for ticker_1, ticker_2 in combinations(tickers, 2):
        likelihoods = []

        #Calculate likelihood for each B/A ratio
        for value in B_over_A:
            alpha = 1 / price_data[ticker_1][0] #calculate the weight of the first asset
            beta = value / price_data[ticker_2][0] #weight of second asset
            spread = alpha * price_data[ticker_1] - beta * price_data[ticker_2] #spread claculation
            parameters = calculate_parameters(spread, dt) # get the ou paprams - theta , mu sigma
            likelihood = log_likelihood(parameters, spread, dt)

            likelihoods.append(likelihood)

        # Find the optimal B for this pair

        ind = np.argmax(likelihoods)
        max_likelihood = likelihoods[ind]
        optimal_B = B_over_A[ind] # get corresponding beta value

        # Recalculate parameters for the optimal B
        alpha = 1 / price_data[ticker_1][0]
        beta = optimal_B / price_data[ticker_2][0]
        spread = alpha * price_data[ticker_1] - beta * price_data[ticker_2]
        mu, theta, sigma = calculate_parameters(spread, dt)

        # Store results
        results.append({        
            "t1": ticker_1,
            "t2": ticker_2,
            "beta": optimal_B,
            "theta": theta,
            "mu": mu,
            "sigma": sigma,
            "likl": max_likelihood,                 
        })
    return sorted(results, key=lambda x: x['likl'],reverse=True)