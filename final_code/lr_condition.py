import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import multiprocessing as mp

np.random.seed(1)

def loss(theta, x):
    #return (theta - x)**2
    return abs(theta - x)

def erm(data):
    if not data.size:
        #return 0
        return np.log(2)
    #return np.mean(data)
    return np.median(data)

def online_gue(data, alpha):
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


    return omega_hats

def get_conditional_path(sample_num, conf_quantile = 0.95):
    #print(sample_num)
    alpha = 0.05
    sample_size = 100

    data = st.expon.rvs(size=sample_size)
    omega_hats = online_gue(data, alpha)
    thetahats = np.zeros(len(data)+1)
    conditional_path = np.zeros(len(data))
    residuals = []
    conditional_vars = np.zeros(len(data))
    differences = np.zeros((1000, len(data)))
    for n in range(len(data)):
        thetahats[n+1] = np.median(data[:n+1])
        conditionals = np.zeros(1000)
        for i in range(1000):
            new_data = st.expon.rvs()
            #conditionals[i] = np.exp(-1*omega_hats[n]*(loss(thetahats[n], new_data) - loss(0, new_data)))
            conditionals[i] = np.exp(-1*omega_hats[n]*(loss(thetahats[n], new_data) - loss(np.log(2), new_data)))
        conditional_path[n] = np.mean(conditionals)
        conditional_vars[n] = np.var(conditionals, ddof=1)

        # CLT-based bootstrap for simultaneous confidence band
        for i in range(1000):
            muhat_b = st.norm.rvs(conditional_path[n], (conditional_vars[n]/1000)**0.5)
            differences[i][n] = muhat_b - conditional_path[n]

    band_sizes = []
    for i in range(len(data)):
        delta_bootstraps = np.zeros(1000)

        for j in range(1000):
            delta_bootstraps[j] = max([abs(a) for a in differences[j]])
        band_sizes.append(np.quantile(delta_bootstraps, conf_quantile))

    return conditional_path, band_sizes

"""
with mp.Pool(mp.cpu_count() - 1) as p:
    conditional_paths = p.map(get_conditional_path, [x for x in range(10)])
"""

fig, ax = plt.subplots(2, 3)
for sample_num in range(100):
    conditional_path, band_sizes = get_conditional_path(sample_num)
    y_upper = None
    y_lower = None
    if max([c + b for (c, b) in zip(conditional_path, band_sizes)]) > 3:
        y_upper = 3
    if min([c - b for (c, b) in zip(conditional_path, band_sizes)]) < 0:
        y_lower = 0
    print(max([(c - b).item() for (c, b) in zip(conditional_path, band_sizes)]))

    if sample_num < 6:
        if sample_num == 5:
            ax[sample_num//3, sample_num%3].plot([n for n in range(len(conditional_path))], conditional_path, label = "Est. Cond. Expectation")
            ax[sample_num//3, sample_num%3].plot([n for n in range(len(conditional_path))], [c - b for (c, b) in zip(conditional_path, band_sizes)], color = 'red', linestyle = 'dashed', label = "95% CI Lower Bound")
            #ax[sample_num//3, sample_num%3].legend()
        else:
            ax[sample_num//3, sample_num%3].plot([n for n in range(len(conditional_path))], conditional_path)
            ax[sample_num//3, sample_num%3].plot([n for n in range(len(conditional_path))], [c - b for (c, b) in zip(conditional_path, band_sizes)], color = 'red', linestyle = 'dashed')
        #ax[sample_num//3, sample_num%3].plot([n for n in range(len(conditional_path))], [c + b for (c, b) in zip(conditional_path, band_sizes)], color = 'red', linestyle = 'dashed')
        ax[sample_num//3, sample_num%3].set_title("Sample #" + str(sample_num))


        ax[sample_num//3, sample_num%3].set(ylim = (y_lower, y_upper))
        #plt.errorbar([n for n in range(len(data))], conditional_path, yerr=2*(conditional_vars/1000)**0.5, label = "Sample #" + str(_))

for axs in ax.flat:
    axs.set(xlabel = "n", ylabel = "Cond. Expectation")
    #axs.label_outer()
#plt.xlim(500, 1000)
#plt.ylim(0,1.5)
#plt.yscale('log')
fig.tight_layout()
plt.savefig("conditional_expectations_median.svg", bbox_inches='tight')
