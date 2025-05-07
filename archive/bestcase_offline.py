import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import multiprocessing as mp

np.random.seed(0)

# 1 dimensional
mu0 = 5
mu1 = 10
cov = 10000
c = (mu0 + mu1)/2


def _gibbs(mu, data, add_to_training):
    data0_train = [x for (x, y) in data[:len(data)//2+add_to_training] if y == 0]
    data0_test = sorted([x for (x, y) in data[len(data)//2+add_to_training:] if y == 0])
    data1_train = [x for (x, y) in data[:len(data)//2+add_to_training] if y == 1]
    data1_test = sorted([x for (x, y) in data[len(data)//2+add_to_training:] if y == 1])

    raw_data = np.array([])
    train_data = np.empty(shape=(0, 2))
    for theta in data0_train:
        idx = np.searchsorted(raw_data, theta)
        train_data = np.insert(train_data, idx, [theta, 0], axis = 0)
        raw_data = np.insert(raw_data, idx, theta)

    for theta in data1_train:
        idx = np.searchsorted(raw_data, theta)
        train_data = np.insert(train_data, idx, [theta, 1], axis = 0)
        raw_data = np.insert(raw_data, idx, theta)

    num_1s = 0
    c_est = train_data[0][0]
    for (theta, label) in train_data:
        if label == 1:
            num_1s += 1
        else:
            num_1s -= 1
            if num_1s <= 0:
                c_est = theta
                num_1s = 0

    def emp_risk_test(theta):
        err_count = np.searchsorted(data1_test, theta, side="right")
        err_count += len(data0_test) - np.searchsorted(data0_test, theta, side="left")
        return err_count
        #return len([x for x in data1_test if x <= theta]) + len([x for x in data0_test if x > theta])

    # Check that confidence set contains mu
    return emp_risk_test(c_est) - emp_risk_test(theta)
    #return T(mu, omega) >= np.log(1 - nom_coverage)

def gibbs(mu, data, nom_coverage):
    boot_iters = 100
    omegas = np.linspace(0.001, 10, num = 10000)
    coverages = np.zeros(len(omegas))
    i = 0
    add_to_training = 0
    num_fails = 0
    while i <= boot_iters:
        train_data = np.array(data[0:len(data)//2 + add_to_training])
        indices = np.random.choice(range(len(train_data)), size = len(train_data), replace = True)
        boot_data = train_data[indices]

        if not [x for (x, y) in boot_data if y == 0] or not [x for (x, y) in boot_data if y == 1]:
            num_fails += 1
            if num_fails > 10:
                num_fails = 0
                add_to_training += 1
            continue

        boot_c = (np.mean([x for (x, y) in boot_data if y == 0]) + np.mean([x for (x, y) in boot_data if y == 1]))/2
        log_gue = _gibbs(boot_c, boot_data, add_to_training)
        for idx, omega in enumerate(omegas):
            if omega*log_gue >= np.log(1-nom_coverage):
                coverages[idx] += 1

        num_fails = 0
        i += 1
    coverages /= boot_iters

    omega = omegas[np.argmin([abs(nom_coverage - coverage) for coverage in coverages])]
    #print("  omega:", omega)
    #print(add_to_training, len(data))
    return omega * _gibbs(mu, data, add_to_training) >= np.log(1-nom_coverage)

def mc_iteration(nom_coverage, mc_iter):
    np.random.seed(mc_iter)
    # Exact confidence interval
    exact_coverage = 0
    universal_coverage = 0
    data = []
    n0 = 0
    n1 = 0
    while True:
        # Generate a new data point
        if np.random.rand() < 0.5:
            data.append((st.norm.rvs(loc = mu0, scale = cov**0.5, size = 1)[0], 0))
            n0 += 1
        else:
            data.append((st.norm.rvs(loc = mu1, scale = cov**0.5, size = 1)[0], 1))
            n1 += 1

        if n0 <= 2 or n1 <= 2:
            continue

        # Point estimate for (mu0 + mu1)/2
        c_est = (np.mean([x for (x, y) in data if y == 0]) + np.mean([x for (x, y) in data if y == 1]))/2

        # (1-\alpha)100% confidence interval for (mu0 + mu1)/2
        z_alpha = st.norm.interval(nom_coverage)[1]
        CI = [c_est - z_alpha* ((n0 + n1)/(4*n0*n1) * cov)**0.5, c_est + z_alpha * ((n0 + n1)/(4*n0*n1) * cov)**0.5]

        # If the confidence interval still includes the origin, collect more data and try again
        if CI[0] <= 0 and 0 <= CI[1]:
            continue

        # Otherwise, we stop collecting data and check if the confidence interval contains the true (mu0+mu1)/2
        if CI[0] <= c and c <= CI[1]:
            exact_coverage += 1
        break

    # Universal interval
    universal_coverage = gibbs(c, data, nom_coverage)
    print(universal_coverage)
    return (exact_coverage, universal_coverage)


nom_coverages = np.linspace(0, 1, num=100)[82:-1]
exact_coverages = []
universal_coverages = []
for nom_coverage in nom_coverages:
    print("Nominal coverage:", round(nom_coverage, 2))
    mc_iters = 100#300
    with mp.Pool(4) as p:
        output = p.starmap(mc_iteration, [(nom_coverage, it) for it in range(mc_iters)])
        #output = [mc_iteration(nom_coverage) for _ in range(mc_iters)]

    exact_coverages.append(np.mean([ec for (ec, uc) in output]))
    universal_coverages.append(np.mean([uc for (ec, uc) in output]))
    print(exact_coverages, universal_coverages)

# Final results
"""
exact_coverages = [0.6066666666666667, 0.56, 0.5566666666666666, 0.6433333333333333, 0.64, 0.6233333333333333, 0.64, 0.6733333333333333, 0.7166666666666667, 0.75, 0.7433333333333333, 0.78, 0.8133333333333334, 0.8166666666666667, 0.8733333333333333, 0.8733333333333333, 0.8966666666666666, 0.9466666666666667, 1.0]
universal_coverages = [0.89, 0.9, 0.9166666666666666, 0.91, 0.8933333333333333, 0.91, 0.9266666666666666, 0.9366666666666666, 0.9433333333333334, 0.9533333333333334, 0.94, 0.9733333333333334, 0.9733333333333334, 0.98, 0.9833333333333333, 0.9833333333333333, 0.9933333333333333, 0.99, 1.0]

plt.rcParams['text.usetex'] = True
plt.title(r"Stopping Rule: 0 $\not\in\bar{X} \pm z_\alpha \sqrt{s^2/n}$")
plt.scatter(nom_coverages, exact_coverages, color="blue", label = "Exact CI")
plt.scatter(nom_coverages, universal_coverages, color = "red", marker = "^", label = "GUI CS")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.legend()
plt.savefig("exact_cropped.png")
"""
