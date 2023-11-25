import numpy as np

np.random.seed(0)

# Millikan's 27 measurements used to calculate the charge of the electron
data = [
    159.3347108,
    159.8040323,
    158.9439612,
    159.7257801,
    159.2956215,
    159.139296,
    159.0611525,
    159.6866588,
    159.3738034,
    158.8658497,
    158.7877509,
    159.1783726,
    159.0220855,
    159.6084258,
    159.0611525,
    159.6084258,
    158.7877509,
    159.1002227,
    159.2565353,
    159.6866588,
    159.2174523,
    158.8267987,
    159.4911002
]


true_value = 160.2176634

def gibbs(omega, data, true_value, alpha):
    data_train = data[:len(data)//2]
    data_test = data[len(data)//2:]
    def risk(theta, data):
        return sum([(theta - X)**2 for X in data])/len(data)

    ratio =  -omega * len(data_test) * (risk(np.mean(data_train), data_test) - risk(true_value, data_test))
    return ratio < np.log(1/alpha)

for alpha in [.05, .1, .2, .3, .4, .5]:
    bootstrap_iters = 100
    coverages = []
    omegas = np.linspace(0, 1, 500)[1:]
    for omega in omegas:
        coverage = 0
        for boot_iter in range(bootstrap_iters):
            coverage += gibbs(omega, np.random.choice(data, size = len(data), replace = True), true_value, alpha)
        coverage /= bootstrap_iters
        coverages.append(coverage)

    omega = omegas[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])]

    result = gibbs(omega, data, true_value, alpha)
    print(alpha, omega, result)
