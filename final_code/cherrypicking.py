import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import sys
import multiprocessing as mp

np.random.seed(0)

dist = int(sys.argv[1])

def emp_risk(theta, data):
    return np.mean([(theta - X)**2 for X in data])

def log_gue_over_omega_fn(data, true_value):
    data_train = data[:len(data)//2]
    data_test = data[len(data)//2:]
    return -1 * len(data_test) * (emp_risk(np.mean(data_train), data_test) - emp_risk(true_value, data_test))

def offline_gue(data, true_value, alpha):
    bootstrap_iters = 100
    omegas = np.linspace(0, 10, 1000)
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

    #print(coverages)
    #print("    ", omega, coverages[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])])
    return omega * log_gue_over_omega_fn(data, true_value) < np.log(1/alpha)

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
    Q1 = P.ppf(0.25)#sorted_data[len(data)//4]
    Q3 = P.ppf(0.75)#sorted_data[3*len(data)//4]
    k = 1

    cherrypicked_data = np.array([x for x in data if x > Q1 - k*(Q3 - Q1) and x < Q3 + k * (Q3 - Q1)])

    exact_coverage = 0
    offline_coverage = 0
    online_coverage = 0

    if P.stats()[0] > np.mean(cherrypicked_data) -  st.t.interval(nom_coverage, len(cherrypicked_data))[1]*(np.var(cherrypicked_data, ddof=1)/len(cherrypicked_data))**0.5 and P.stats()[0] < np.mean(cherrypicked_data) + st.t.interval(nom_coverage, len(cherrypicked_data))[1]*(np.var(cherrypicked_data, ddof=1)/len(cherrypicked_data))**0.5:
        exact_coverage += 1

    offline_coverage += offline_gue(cherrypicked_data, P.stats()[0], 1-nom_coverage)
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
    break
    print(nom_coverage)
    exact_coverage = 0
    offline_coverage = 0
    online_coverage = 0
    mc_iters = 1000

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

# Triangular distribution
if dist == 0:
    """
    exact_coverages = [np.float64(0.674), np.float64(0.687), np.float64(0.7), np.float64(0.712), np.float64(0.721), np.float64(0.734), np.float64(0.749), np.float64(0.757), np.float64(0.768), np.float64(0.777), np.float64(0.788), np.float64(0.8), np.float64(0.808), np.float64(0.823), np.float64(0.839), np.float64(0.857), np.float64(0.874), np.float64(0.896), np.float64(0.918), np.float64(0.942)]
    offline_coverages = [np.float64(0.835), np.float64(0.838), np.float64(0.842), np.float64(0.847), np.float64(0.851), np.float64(0.859), np.float64(0.866), np.float64(0.873), np.float64(0.88), np.float64(0.885), np.float64(0.889), np.float64(0.894), np.float64(0.904), np.float64(0.918), np.float64(0.929), np.float64(0.939), np.float64(0.947), np.float64(0.956), np.float64(0.975), np.float64(0.988)]
    online_coverages = [np.float64(0.928), np.float64(0.92), np.float64(0.924), np.float64(0.93), np.float64(0.938), np.float64(0.939), np.float64(0.94), np.float64(0.939), np.float64(0.94), np.float64(0.946), np.float64(0.948), np.float64(0.948), np.float64(0.954), np.float64(0.958), np.float64(0.965), np.float64(0.968), np.float64(0.965), np.float64(0.972), np.float64(0.981), np.float64(0.992)]
    """
    exact_coverages = [np.float64(0.767), np.float64(0.783), np.float64(0.796), np.float64(0.808), np.float64(0.818), np.float64(0.825), np.float64(0.836), np.float64(0.848), np.float64(0.855), np.float64(0.861), np.float64(0.875), np.float64(0.884), np.float64(0.895), np.float64(0.908), np.float64(0.921), np.float64(0.932), np.float64(0.951), np.float64(0.969), np.float64(0.983), np.float64(0.99)]
    offline_coverages = [np.float64(0.884), np.float64(0.888), np.float64(0.892), np.float64(0.9), np.float64(0.905), np.float64(0.909), np.float64(0.913), np.float64(0.919), np.float64(0.925), np.float64(0.929), np.float64(0.934), np.float64(0.941), np.float64(0.946), np.float64(0.953), np.float64(0.965), np.float64(0.973), np.float64(0.975), np.float64(0.98), np.float64(0.987), np.float64(0.995)]
    online_coverages = [np.float64(0.967), np.float64(0.967), np.float64(0.964), np.float64(0.968), np.float64(0.969), np.float64(0.97), np.float64(0.968), np.float64(0.97), np.float64(0.968), np.float64(0.972), np.float64(0.975), np.float64(0.976), np.float64(0.979), np.float64(0.976), np.float64(0.98), np.float64(0.978), np.float64(0.983), np.float64(0.987), np.float64(0.993), np.float64(0.996)]


# Beta distribution
else:
    #exact_coverages = [np.float64(0.587), np.float64(0.593), np.float64(0.601), np.float64(0.615), np.float64(0.631), np.float64(0.641), np.float64(0.649), np.float64(0.666), np.float64(0.682), np.float64(0.694), np.float64(0.706), np.float64(0.714), np.float64(0.725), np.float64(0.742), np.float64(0.761), np.float64(0.776), np.float64(0.799), np.float64(0.814), np.float64(0.841), np.float64(0.879)]
    exact_coverages = [np.float64(0.692), np.float64(0.7), np.float64(0.712), np.float64(0.726), np.float64(0.739), np.float64(0.753), np.float64(0.761), np.float64(0.775), np.float64(0.79), np.float64(0.806), np.float64(0.812), np.float64(0.82), np.float64(0.829), np.float64(0.845), np.float64(0.867), np.float64(0.886), np.float64(0.906), np.float64(0.92), np.float64(0.936), np.float64(0.965)]
    offline_coverages = [np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0)]
    online_coverages = [np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0)]

plt.scatter(nom_coverages, exact_coverages, color = "blue", label = "Exact CI")
plt.scatter(nom_coverages, online_coverages, color = "red", marker = "^", label = "Online GUI CS")
plt.scatter(nom_coverages, offline_coverages, color = "gold", marker = "+", label = "Offline GUI CS")
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
