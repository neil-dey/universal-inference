import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import sys

np.random.seed(0)

dist = int(sys.argv[1])

def _gibbs(omega, data, true_value, alpha):
    thetahats = [np.mean(data[0: i+1]) for i in range(len(data))]
    thetahats = [0] + thetahats

    ratio =  -omega * sum([(thetahat - x)**2 - (true_value - x)**2 for thetahat, x in zip(thetahats, data)])
    return ratio < np.log(1/alpha)

def gibbs(data, true_value, alpha):
    bootstrap_iters = 100
    coverages = []
    omegas = np.linspace(0, 100, 100)[1:]
    for omega in omegas:
        coverage = 0
        for boot_iter in range(bootstrap_iters):
            coverage += _gibbs(omega, np.random.choice(data, size = len(data), replace = True), true_value, alpha)
        coverage /= bootstrap_iters
        coverages.append(coverage)

    omega = omegas[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])]

    #print("    ", omega)
    return _gibbs(omega, data, true_value, alpha)


if dist == 0:
    P = st.norm()
else:
    P = st.beta(a=5, b=2)
nom_coverages = np.linspace(0.8, 0.99, num=20)[:-1]
exact_coverages = []
gibbs_coverages = []
for nom_coverage in nom_coverages:
    print(nom_coverage)
    exact_coverage = 0
    gibbs_coverage = 0
    mc_iters = 100
    for _ in range(mc_iters):
        data_size = 30
        data = P.rvs(size = data_size)
        sorted_data = sorted(data)
        Q1 = sorted_data[len(data)//4]
        Q3 = sorted_data[3*len(data)//4]
        k = 1

        cherrypicked_data = [x for x in data if x > Q1 - k*(Q3 - Q1) and x < Q3 + k * (Q3 - Q1)]
        #print(len(cherrypicked_data))

        if P.stats()[0] > np.mean(cherrypicked_data) -  st.t.interval(nom_coverage, len(cherrypicked_data))[1]*(np.var(cherrypicked_data, ddof=1)/len(cherrypicked_data))**0.5 and P.stats()[0] < np.mean(cherrypicked_data) + st.t.interval(nom_coverage, len(cherrypicked_data))[1]*(np.var(cherrypicked_data, ddof=1)/len(cherrypicked_data))**0.5:
            exact_coverage += 1

        gibbs_coverage += gibbs(cherrypicked_data, P.stats()[0], 1-nom_coverage)
    exact_coverage /= mc_iters
    gibbs_coverage /= mc_iters
    exact_coverages.append(exact_coverage)
    gibbs_coverages.append(gibbs_coverage)
    print(exact_coverages)
    print(gibbs_coverages)


# Final results
"""
# Normal
exact_coverages = [0.72, 0.73, 0.73, 0.73, 0.73, 0.76, 0.78, 0.79, 0.79, 0.79, 0.81, 0.81, 0.81, 0.83, 0.85, 0.87, 0.9, 0.93, 0.94]
gibbs_coverages = [0.94, 0.94, 0.94, 0.94, 0.95, 0.95, 0.95, 0.96, 0.96, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.99, 0.99] # N(0, 1)
offline_coverages = [0.916, 0.922, 0.918, 0.931, 0.936, 0.94, 0.938, 0.944, 0.96, 0.954, 0.955, 0.96, 0.961, 0.957, 0.977, 0.978, 0.981, 0.984, 0.991] # N(0, 1)

# Beta
exact_coverages = [0.655, 0.674, 0.624, 0.645, 0.68, 0.671, 0.681, 0.708, 0.717, 0.744, 0.741, 0.755, 0.765, 0.781, 0.797, 0.824, 0.841, 0.841, 0.868, 0.915][:-1] # Beta(5, 2)
offline_coverages = [0.999, 0.999, 0.999, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0][:-1] # Beta(5, 2)
gibbs_coverages = [1.0]*19

plt.scatter(nom_coverages, exact_coverages, color = "blue", label = "Exact CI")
plt.scatter(nom_coverages, gibbs_coverages, color = "red", marker = "^", label = "Online GUe CS")
plt.scatter(nom_coverages, offline_coverages, color = "gold", marker = "+", label = "Offline GUe CS")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")

plt.title("Effects of Outlier Removal in a Normal Sample")
plt.title("Effects of Outlier Removal in a Beta Sample")
plt.legend()
#plt.show()

plt.savefig("./cherrypicked_normal.png")
plt.savefig("./cherrypicked_beta.png")
"""
