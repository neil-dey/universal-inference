import numpy as np
import scipy.stats as st


def emp_risk(theta, data):
    return np.mean([(theta - X)**2 for X in data])

def log_gue_over_omega_fn(data, true_value):
    data_train = data[:len(data)//2]
    data_test = data[len(data)//2:]
    return -1 * len(data_test) * (emp_risk(np.mean(data_train), data_test) - emp_risk(true_value, data_test))

def offline_gue(data, true_value, alpha):
    bootstrap_iters = 100
    omegas = np.linspace(0, 5, 100)
    coverages = np.zeros(len(omegas))

    data_train = data[:len(data)//2]
    data_test = data[len(data)//2:]

    for boot_iter in range(bootstrap_iters):
        boot_data = data_train[np.random.choice(len(data_train), size = len(data_train), replace = True)]
        lgoo = log_gue_over_omega_fn(boot_data, np.mean(data_train))
        for idx, omega in enumerate(omegas):
            coverages[idx] += (omega * lgoo < np.log(1/alpha))
    coverages /= bootstrap_iters

    omega = omegas[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])]

    return omega * log_gue_over_omega_fn(data, true_value) < np.log(1/alpha)

def online_gue(data, true_value, alpha):
    boot_iters = 100
    coverages = []
    omegas = np.linspace(0, 5, num = 100)

    omega_hats = np.zeros(len(data))
    for n in range(0, len(data)):
        coverages = np.zeros(len(omegas))
        for _ in range(boot_iters):
            boot_data = data[np.random.choice(n+1, n+1, replace = True)]


            boot_erms = [np.mean(boot_data[0: i+1]) for i in range(len(boot_data))]
            boot_erms = [159] + boot_erms
            boot_mu = np.mean(data[:n+1])

            excess_losses = [(thetahat - x)**2 - (boot_mu - x)**2 for (thetahat, x) in zip(boot_erms, boot_data)]
            log_gue_over_omega = sum([-1*excess_losses[i] for i in range(n-1)])
            for idx, omega in enumerate(omegas):
                log_gue = omega*log_gue_over_omega
                coverages[idx] += log_gue < np.log(1/alpha)
        coverages /= boot_iters
        omega_hats[n] = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]

    erms = [np.mean(data[0: i+1]) for i in range(len(data))]
    erms = [159] + erms
    excess_losses = [(thetahat - x)**2 - (true_value - x)**2 for (thetahat, x) in zip(erms, data)]
    return sum([-1*omega_hat * excess_loss for (omega_hat, excess_loss) in zip(omega_hats, excess_losses)]) < np.log(1/alpha)


np.random.seed(0)
data = np.array([159.3347108, 159.8040323, 158.9439612, 159.7257801, 159.2956215, 159.139296, 159.0611525, 159.6866588, 159.3738034, 158.8658497, 158.7877509, 159.1783726, 159.0220855, 159.6084258, 159.0611525, 159.6084258, 158.7877509, 159.1002227, 159.2565353, 159.6866588, 159.2174523, 158.8267987, 159.4911002])
true_value = 160.2176634

alphas = np.linspace(0, 0.5, 100)[1:]
for alpha in alphas:
    off = offline_gue(data, true_value, alpha)
    on = online_gue(data, true_value, alpha)
    print(alpha, off, on)
