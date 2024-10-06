import numpy as np
import scipy.stats as st
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import sys
import time
from itertools import compress

np.random.seed(0)


n = 10
boot_iters = 100
mc_iters = 10

theta = 10
sigma_alpha = 1/n
sigma_beta = 1/(2*n)
sigma = 1

nom_coverage = int(sys.argv[1])

def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.6+
    count = len(it)
    start = time.time() # time estimate start
    def show(j):
        x = int(size*j/count)
        # time estimate calculation and string
        remaining = ((time.time() - start) / j) * (count - j)
        mins, sec = divmod(remaining, 60) # limited to minutes
        time_str = f"{int(mins):02}:{sec:03.1f}"
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}", end='\r', file=out, flush=True)
    show(0.1) # avoid div/0
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def loss(x, theta):
    return (x-theta)**2

def emprisk(theta, xs):
    return sum([loss(x, theta) for x in xs])

def _gibbs(true_theta, data, alpha, omega, mode):
    if mode == "off":
        train_data = data[0:len(data)//2]
        test_data = data[len(data)//2:]
        thetahat = np.mean(train_data)
        log_gue = -1*omega * (emprisk(thetahat, test_data) - emprisk(true_theta, test_data))

    elif mode == "on":
        thetahats = np.zeros(len(data) + 1)
        running_sum = 0
        for idx, x in enumerate(data, start = 1):
            running_sum += x
            thetahats[idx] = running_sum/idx

        log_gue = -1*omega*sum([loss(x, thetahat) - loss(x, true_theta) for (x, thetahat) in zip(data, thetahats)])

    return log_gue < np.log(1/alpha)

def gibbs(true_theta, data, alpha, mode):
    thetahat = np.mean(data)
    coverages = []
    omegas = np.linspace(0, 3, num=100)[1:]
    for omega in omegas:
        coverage = 0
        for _ in range(boot_iters):
            boot_data = np.random.choice(data, size = len(data), replace = True)
            if _gibbs(thetahat, boot_data, alpha, omega, mode):
                coverage += 1
        coverage /= boot_iters
        coverages.append(coverage)

    omega = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]
    #print([(np.round(omega, 2), coverage) for (omega, coverage) in zip(omegas, coverages)])
    #print("   " , omega, coverages[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])])
    return _gibbs(true_theta, data, alpha, omega, mode)




#for mc_iter in progressbar(range(mc_iters), str(np.round(nom_coverage, 0)) + " "):
for mc_iter in range(mc_iters):
    continue
    candidate_thetas = np.linspace(5.5, 15.5, num=100)
    bootstrap_inclusions = []
    online_inclusions = []
    offline_inclusions = []

    xs = np.array([theta + st.gamma.rvs(a = 1, loc = -sigma_alpha, scale = sigma_alpha) + st.gamma.rvs(a = 1, loc = -sigma_beta, scale = sigma_beta) + st.gamma.rvs(a = 1, loc = -sigma, scale = sigma) for _ in range(n)])
    thetahat = np.mean(xs)

    for c_t in candidate_thetas:
        distances = []
        for _ in range(boot_iters):
            indices = np.random.choice(n, n)
            boot_xs = xs[indices]
            boot_thetahat = np.mean(boot_xs)
            distances.append(abs(boot_thetahat - thetahat))

        bootstrap_inclusions.append(abs(thetahat - c_t) < np.percentile(distances, nom_coverage))

        online_inclusions.append(gibbs(c_t, xs, 1-nom_coverage/100, "on"))

        offline_inclusions.append(gibbs(c_t, xs, 1-nom_coverage/100, "off"))


    print()
    l = list(compress(candidate_thetas, bootstrap_inclusions))
    print("Bootstrap:", min(l), max(l))
    l = list(compress(candidate_thetas, online_inclusions))
    print("Online:", min(l), max(l))
    l = list(compress(candidate_thetas, offline_inclusions))
    print("Offline:", min(l), max(l))


bootstrap_intervals = [(9.843434343434343, 10.954545454545453), (9.237373737373737, 10.247474747474747), (9.237373737373737, 10.44949494949495), (9.540404040404042, 10.146464646464647), (9.43939393939394, 10.44949494949495),(9.843434343434343, 10.651515151515152),(9.338383838383837, 9.641414141414142),(9.742424242424242, 10.348484848484848),(9.43939393939394, 10.247474747474747),(9.944444444444445, 10.853535353535353)]

online_intervals = [(6.207070707070707, 14.59090909090909),(5.601010101010101, 14.085858585858587),(5.5, 14.48989898989899),(5.702020202020202, 13.883838383838384),(5.904040404040404, 13.984848484848484),(5.803030303030303, 14.59090909090909),(5.5, 13.47979797979798),(6.005050505050505, 14.186868686868687),(5.904040404040404, 13.681818181818182),(6.106060606060606, 14.691919191919192)]

offline_intervals = [(9.843434343434343, 11.257575757575758),(9.338383838383837, 11.257575757575758),(7.722222222222222, 10.55050505050505),(9.43939393939394, 10.954545454545453),(8.934343434343434, 10.44949494949495),(9.742424242424242, 11.358585858585858),(9.136363636363637, 10.348484848484848),(9.641414141414142, 10.348484848484848),(9.338383838383837, 10.954545454545453),(9.944444444444445, 10.853535353535353)]

plt.title("Visualizations of Confidence Intervals for Two-Way Mean")

"""
x = 0.9
legend_flag = True
for (bi, oni, ofi) in zip(bootstrap_intervals, online_intervals, offline_intervals):
    plt.vlines(x, bi[0], bi[1], color = 'blue', label = "Bootstrapped CI" if legend_flag else "")
    x += 0.1
    plt.vlines(x, ofi[0], ofi[1], color = 'gold', label = "Offline CI" if legend_flag else "")
    x += 0.1
    plt.vlines(x, oni[0], oni[1], color = 'red', label = "Online CI" if legend_flag else "")
    x += 0.8
    legend_flag = False
"""
x = 0.8
legend_flag = True
for (bi, oni, ofi) in zip(bootstrap_intervals, online_intervals, offline_intervals):
    plt.scatter([x, x], [bi[0], bi[1]], color = 'blue', marker = "o", label = "Bootstrapped CI" if legend_flag else "")
    plt.vlines(x, bi[0], bi[1], color = 'blue')
    x += 0.2
    plt.scatter([x, x], [ofi[0], ofi[1]], color = 'gold', marker = "x", label = "Offline CI" if legend_flag else "")
    plt.vlines(x, ofi[0], ofi[1], color = 'gold')
    x += 0.2
    plt.scatter([x, x], [oni[0], oni[1]], color = 'red', marker = "^", label = "Online CI" if legend_flag else "")
    plt.vlines(x, oni[0], oni[1], color = 'red')
    x += 0.6
    legend_flag = False

plt.hlines(theta, 0.8, 10.2, color = 'black', linestyle='dashed')
plt.ylim((2, 17))
plt.legend(loc = 'lower right')
plt.xlabel("Trial #")
plt.savefig("anova_ci_width.svg")
plt.show()
