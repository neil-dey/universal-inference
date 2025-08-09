import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import multiprocessing as mp
import sys

ALTERNATIVE = float(sys.argv[1])

np.random.seed(1)

def loss(theta, x):
    return abs(theta - x)

def erm(data):
    if not data.size:
        return np.log(2)
    return np.median(data)

def online_gue(data, seed, alpha = 0.05):
    np.random.seed(seed)
    thetahats = np.zeros(len(data)+1)
    omega_hats = np.zeros(len(data))
    omegas = np.linspace(0, 10, 1000)
    boot_iters = 100

    for n in range(len(data)):
        coverages = np.zeros(len(omegas))
        thetahats[n+1] = np.median(data[:n+1])
        for _ in range(boot_iters):
            # Bootstrap the first n data points
            choices = np.random.choice(range(n+1), size = n+1, replace = True)
            boot_data = data[choices]

            # Estimate the bootstrapped thetahats
            boot_thetahats = np.zeros(len(boot_data)+1)
            for i in range(len(boot_data) + 1):
                boot_thetahats[i] = erm(boot_data[:i])

            log_gue_over_omega = sum([omega*(loss(boot_thetahats[i], boot_data[i]) - loss(thetahats[n], boot_data[i])) for i in range(n)])
            for idx, omega in enumerate(omegas):
                log_gue = omega * log_gue_over_omega
                coverages[idx] += log_gue < np.log(1/alpha)
        coverages /= boot_iters
        best_omega = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]
        omega_hats[n] = best_omega
        #print(n, omega_hats[n])


    return omega_hats

try:
    with open('vectors' + str(ALTERNATIVE) + '.npy', 'rb') as f:
        omega_hat_vectors = np.load(f)
except:
    with mp.Pool(mp.cpu_count() - 1) as p:
        omega_hat_vectors = p.starmap(online_gue, [(st.expon.rvs(size = 100, loc = ALTERNATIVE), x) for x in range(100)])

    with open('vectors' + str(ALTERNATIVE) + '.npy', 'wb') as f:
        np.save(f, omega_hat_vectors)

transpose = [[x[i] for x in omega_hat_vectors] for i in range(len(omega_hat_vectors[0]))][2::10]

plt.violinplot(transpose, positions = [i*10+2 for i in range(10)], widths = 5, showextrema=False)
plt.ylim(0, 0.25)
plt.xlabel("n")
plt.ylabel("Learning Rate")
#plt.show()
plt.savefig("learning_rates_vs_n_alt" + str(ALTERNATIVE) + ".png")
