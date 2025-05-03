import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import sys
import multiprocessing as mp

np.random.seed(0)

modes = ["online", "offline"]
mode = modes[int(sys.argv[1])]

omega_modes = ["parambootstrap_normal", "parambootstrap_beta", "nonparametric"]
omega_mode = omega_modes[int(sys.argv[2])]

print(mode, omega_mode)

def ramdas(mu, data, alpha, c = 1/2):
    mu_hat = lambda t: (sum([X_i for (i, X_i) in enumerate(data) if i <= t]) + 1/2)/(t+2)
    sigma_sq_hat = lambda t: (sum([(X_i - mu_hat(i))**2 for (i, X_i) in enumerate(data) if i <= t]) + 1/4)/(t+2)
    lam = lambda t: min(c, (2*np.log(2/alpha)/(sigma_sq_hat(t - 1) * (t+1) * np.log(t+2)))**0.5)
    v = lambda t: 4 * (data[t] - mu_hat(t-1))**2
    psi = lambda t: (-np.log(1-t) - t)/4

    t = len(data)
    center = sum([lam(i) * X_i for (i, X_i) in enumerate(data)])/sum([lam(i) for i in range(t)])

    width = (np.log(2/alpha)  + sum([v(i) * psi(lam(i)) for i in range(t)]))/sum([lam(i) for i in range(t)])

    return center - width <= mu and mu <= center+width

def online_gue(mu, data, alpha):
    boot_iters = 100
    coverages = []
    omegas = np.linspace(0.1, 100, num = 1000)
    omegas = omegas[::-1]

    omega_hats = np.zeros(len(data))
    for n in range(0, len(data)):
        coverages = np.zeros(len(omegas))
        for _ in range(boot_iters):
            # Bootstrap the first n data points (as long as an estimate can still be made from it)
            if omega_mode == "parambootstrap_normal":
                boot_data = st.norm(loc = np.mean(data), scale = np.var(data, ddof=1)**0.5).rvs(size = len(data))
            elif omega_mode == "parambootstrap_beta":
                boot_a, boot_b, _, _ = st.fit(st.beta, data, bounds = [(0, 10), (0, 10)]).params
                boot_data = st.beta(a = boot_a, b = boot_b).rvs(size = len(data))
            elif omega_mode == "nonparametric":
                boot_data = data[np.random.choice(n+1, n+1, replace = True)]
            else:
                print("Unsupported Mode")
                exit()


            boot_erms = [np.mean(boot_data[0: i+1]) for i in range(len(boot_data))]
            boot_erms = [0.5] + boot_erms
            boot_mu = np.mean(data[:n+1])

            excess_losses = [(thetahat - x)**2 - (boot_mu - x)**2 for (thetahat, x) in zip(boot_erms, boot_data)]
            log_gue_over_omega = sum([-1*excess_losses[i] for i in range(n-1)])
            for idx, omega in enumerate(omegas):
                log_gue = omega*log_gue_over_omega
                coverages[idx] += log_gue < np.log(1/alpha)
        coverages /= boot_iters
        omega_hats[n] = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]

    erms = [np.mean(data[0: i+1]) for i in range(len(data))]
    erms = [0.5] + erms
    excess_losses = [(thetahat - x)**2 - (mu - x)**2 for (thetahat, x) in zip(erms, data)]
    return sum([-1*omega_hat * excess_loss for (omega_hat, excess_loss) in zip(omega_hats, excess_losses)]) < np.log(1/alpha)

def offline_gue(mu, data, alpha):
    boot_iters = 100
    omegas = np.linspace(0.1, 100, num = 1000)

    training = data[:len(data)//2]
    validation = data[len(data)//2:]

    coverages = np.zeros(len(omegas))
    for _ in range(boot_iters):
        # Bootstrap the first n data points (as long as an estimate can still be made from it)
        if omega_mode == "parambootstrap_normal":
            boot_data = st.norm(loc = np.mean(training), scale = np.var(data, ddof=1)**0.5).rvs(size = len(data))
        elif omega_mode == "parambootstrap_beta":
            boot_a, boot_b, _, _ = st.fit(st.beta, training, bounds = [(0, 10), (0, 10)]).params
            boot_data = st.beta(a = boot_a, b = boot_b).rvs(size = len(data))
        elif omega_mode == "nonparametric":
            n = len(training)
            boot_data = training[np.random.choice(n, len(data), replace = True)]
        else:
            print("Unsupported Mode")
            exit()


        thetahat = np.mean(boot_data[:len(boot_data)//2])
        boot_mu = np.mean(training)
        excess_losses = [(thetahat - x)**2 - (boot_mu - x)**2 for x in boot_data[len(boot_data)//2:]]
        log_gue_over_omega = -1*sum(excess_losses)
        for idx, omega in enumerate(omegas):
            log_gue = omega*log_gue_over_omega
            coverages[idx] += log_gue < np.log(1/alpha)
    coverages /= boot_iters
    omega = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]

    thetahat = np.mean(training)
    excess_losses = [(thetahat - x)**2 - (mu - x)**2 for x in validation]
    return -1*omega* sum(excess_losses) < np.log(1/alpha)

def gibbs(mu, data, alpha):
    if mode == "online":
        return online_gue(mu, data, alpha)

    return offline_gue(mu, data, alpha)



P = st.beta(a=5, b=2)
mu = P.stats()[0]

ramdas_coverages = []
gibbs_coverages = []
nom_coverages = np.linspace(0, 1, num = 100)[95:-1]
for nom_coverage in nom_coverages:
    print(nom_coverage)
    ramdas_coverage = 0
    gibbs_coverage = 0
    mc_iters = 100
    with mp.Pool(4) as p:
        results = p.starmap(gibbs, [(mu, P.rvs(size = 10), 1-nom_coverage) for it in range(mc_iters)])
        gibbs_coverage += sum(results)
    ramdas_coverage /= mc_iters
    gibbs_coverage /= mc_iters
    ramdas_coverages.append(ramdas_coverage)
    gibbs_coverages.append(gibbs_coverage)
    print("Ramdas coverage:", ramdas_coverages)
    print("GUe coverage:", gibbs_coverages)


exit()
# New results
ramdas_coverages = [0.998, 0.996, 1.0, 0.995, 0.998, 0.999, 0.996, 0.998, 1.0, 0.999, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0][-19:]
online_nonparametric = [np.float64(0.9586), np.float64(0.9518), np.float64(0.9578), np.float64(0.9596), np.float64(0.9602), np.float64(0.9606), np.float64(0.9606), np.float64(0.9592), np.float64(0.956), np.float64(0.9564), np.float64(0.9646), np.float64(0.9626), np.float64(0.968), np.float64(0.9692), np.float64(0.973), np.float64(0.9696), np.float64(0.976), np.float64(0.98), np.float64(0.9836)]

offline_nonparametric = [np.float64(0.8846), np.float64(0.8822), np.float64(0.8948), np.float64(0.8858), np.float64(0.909), np.float64(0.9026), np.float64(0.9162), np.float64(0.9268), np.float64(0.9194), np.float64(0.9296), np.float64(0.9366), np.float64(0.9458), np.float64(0.9442), np.float64(0.9528), np.float64(0.9622), np.float64(0.9666), np.float64(0.9778), np.float64(0.9858), np.float64(0.9934)]

plt.plot(nom_coverages, nom_coverages, color = 'black')
plt.scatter(nom_coverages, online_nonparametric, color = 'blue', label = "Online GUe")
plt.scatter(nom_coverages, offline_nonparametric, color = 'red', marker = "^", label = "Offline GUe")
plt.scatter(nom_coverages, ramdas_coverages, color = 'purple', marker = "x", label = "PrPl-EB")
plt.legend()
plt.savefig("new_ramdas.svg")
exit()
