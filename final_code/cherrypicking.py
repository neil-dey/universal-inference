import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import sys
import multiprocessing as mp

np.random.seed(0)

dist = int(sys.argv[1])

def emp_risk(theta, data):
    return np.mean([(theta - X)**2 for X in data])

def log_gue_over_omega(data, true_value):
    data_train = data[:len(data)//2]
    data_test = data[len(data)//2:]
    return -1 * len(data_test) * (emp_risk(np.mean(data_train), data_test) - emp_risk(true_value, data_test))

def offline_gue(data, true_value, alpha):
    bootstrap_iters = 100
    omegas = np.linspace(0, 10, 100)[1:]
    coverages = np.zeros(len(omegas))

    data_train = data[:len(data)//2]
    data_test = data[len(data)//2:]

    for boot_iter in range(bootstrap_iters):
        boot_data = data_train[np.random.choice(len(data_train), size = len(data_train), replace = True)]
        lgoo = log_gue_over_omega(boot_data, np.mean(data_train))
        for idx, omega in enumerate(omegas):
            coverages[idx] += (omega * lgoo < np.log(1/alpha))
    coverages /= bootstrap_iters

    omega = omegas[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])]

    #print(coverages)
    #print("    ", omega, coverages[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])])
    return omega * log_gue_over_omega(data, true_value) < np.log(1/alpha)

def online_gue(data, true_value, alpha):
    boot_iters = 100
    coverages = []
    omegas = np.linspace(0, 10, num = 100)[1:]

    omega_hats = np.zeros(len(data))
    for n in range(0, len(data)):
        coverages = np.zeros(len(omegas))
        for _ in range(boot_iters):
            boot_data = data[np.random.choice(n+1, n+1, replace = True)]


            boot_erms = [np.mean(boot_data[0: i+1]) for i in range(len(boot_data))]
            boot_erms = [1] + boot_erms
            boot_mu = np.mean(data[:n+1])

            excess_losses = [(thetahat - x)**2 - (boot_mu - x)**2 for (thetahat, x) in zip(boot_erms, boot_data)]
            log_gue_over_omega = sum([-1*excess_losses[i] for i in range(n-1)])
            for idx, omega in enumerate(omegas):
                log_gue = omega*log_gue_over_omega
                coverages[idx] += log_gue < np.log(1/alpha)
        coverages /= boot_iters
        omega_hats[n] = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]

    erms = [np.mean(data[0: i+1]) for i in range(len(data))]
    erms = [1] + erms
    excess_losses = [(thetahat - x)**2 - (true_value - x)**2 for (thetahat, x) in zip(erms, data)]
    return sum([-1*omega_hat * excess_loss for (omega_hat, excess_loss) in zip(omega_hats, excess_losses)]) < np.log(1/alpha)

def mc_iter(iternum):
    np.random.seed(iternum)

    data_size = 60
    data = P.rvs(size = data_size)

    sorted_data = sorted(data)
    Q1 = sorted_data[len(data)//4]
    Q3 = sorted_data[3*len(data)//4]
    k = 1

    cherrypicked_data = np.array([x for x in data if x > Q1 - k*(Q3 - Q1) and x < Q3 + k * (Q3 - Q1)])

    exact_coverage = 0
    offline_coverage = 0
    online_coverage = 0

    if P.stats()[0] > np.mean(cherrypicked_data) -  st.t.interval(nom_coverage, len(cherrypicked_data))[1]*(np.var(cherrypicked_data, ddof=1)/len(cherrypicked_data))**0.5 and P.stats()[0] < np.mean(cherrypicked_data) + st.t.interval(nom_coverage, len(cherrypicked_data))[1]*(np.var(cherrypicked_data, ddof=1)/len(cherrypicked_data))**0.5:
        exact_coverage += 1

    #offline_coverage += offline_gue(cherrypicked_data, P.stats()[0], 1-nom_coverage)
    online_coverage += online_gue(cherrypicked_data, P.stats()[0], 1-nom_coverage)

    return (exact_coverage, offline_coverage, online_coverage)

if dist == 0:
    P = st.triang(c=0.5, scale = 2)
else:
    P = st.beta(a=5, b=2)
nom_coverages = np.linspace(0.8, 0.99, num=20)
exact_coverages = []
offline_coverages = []
online_coverages = []
for nom_coverage in nom_coverages:
    print(nom_coverage)
    exact_coverage = 0
    offline_coverage = 0
    online_coverage = 0
    mc_iters = 100

    with mp.Pool(4) as p:
        coverages = p.map(mc_iter, [i for i in range(mc_iters)])

    exact_coverage = np.mean([e for (e, off, on) in coverages])
    offline_coverage = np.mean([off for (e, off, on) in coverages])
    online_coverage = np.mean([on for (e, off, on) in coverages])

    exact_coverages.append(exact_coverage)
    offline_coverages.append(offline_coverage)
    online_coverages.append(online_coverage)
    print(exact_coverages)
    print(offline_coverages)
    print(online_coverages)

# Final results
plt.scatter(nom_coverages, exact_coverages, color = "blue", label = "Exact CI")
plt.scatter(nom_coverages, online_coverages, color = "red", marker = "^", label = "Online GUI CS")
plt.scatter(nom_coverages, gibbs_coverages, color = "gold", marker = "+", label = "GUI CS")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.legend()
if dist == 0:
    plt.title("Outliers Removed: Triangular Sample")
    plt.savefig("cherrypicked_triang.svg")
else:
    plt.title("Outliers Removed: Beta Sample")
    plt.savefig("cherrypicked_beta.svg")
