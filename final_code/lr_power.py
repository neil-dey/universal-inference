import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import multiprocessing as mp

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
    rejections = []

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

        log_gue = sum([-1*omega_hats[i]*(loss(thetahats[i], data[i]) - loss(np.log(2), boot_data[i])) for i in range(n)])
        rejections.append(log_gue >= -np.log(alpha))

    return rejections

alternatives = [1, 1.5, 2]
colors = ["blue", "red", "gold"]
markers = ['.', '^', '+']
for idx, alternative in enumerate(alternatives):
    print(alternative)
    try:
        with open('rejections' + str(idx) + '.npy', 'rb') as f:
            rejections = np.load(f)
    except:
        with mp.Pool(mp.cpu_count() - 1) as p:
            rejections = p.starmap(online_gue, [(st.expon.rvs(loc = alternative, size = 100), x) for x in range(1000)])

        with open('rejections' + str(idx) + '.npy', 'wb') as f:
            np.save(f, rejections)

    power = [1-np.mean([x[i] for x in rejections]) for i in range(len(rejections[0]))]
    plt.scatter([i for i in range(100)], power, color = colors[idx], marker = markers[idx], label = 'log(2) + ' + str(alternative))

plt.ylabel("Type II Error Rate")
plt.xlabel("n")
plt.ylim(0, 1)
plt.legend(title = "Alternative")
#plt.show()
plt.savefig("power_curves.svg")
