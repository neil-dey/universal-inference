import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

np.random.seed(0)

def _gibbs(omega, data, true_value, alpha):
    data_train = data[:len(data)//2]
    data_test = data[len(data)//2:]
    def risk(theta, data):
        return sum([(theta - X)**2 for X in data])/len(data)

    ratio =  -omega * len(data_test) * (risk(np.mean(data_train), data_test) - risk(true_value, data_test))
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


P = st.beta(a=5, b=2)
nom_coverages = np.linspace(0.8, 0.99, num=20)
exact_coverages = []
gibbs_coverages = []
for nom_coverage in nom_coverages:
    continue
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

#exact_coverages = [0.696, 0.704, 0.711, 0.73, 0.755, 0.74, 0.756, 0.773, 0.799, 0.79, 0.782, 0.846, 0.815, 0.85, 0.864, 0.883, 0.881, 0.925, 0.929, 0.953] # N(0, 1)
#gibbs_coverages = [0.916, 0.922, 0.918, 0.931, 0.936, 0.94, 0.938, 0.944, 0.96, 0.954, 0.955, 0.96, 0.961, 0.957, 0.977, 0.978, 0.981, 0.984, 0.991, 0.995] # N(0, 1)

exact_coverages = [0.655, 0.674, 0.624, 0.645, 0.68, 0.671, 0.681, 0.708, 0.717, 0.744, 0.741, 0.755, 0.765, 0.781, 0.797, 0.824, 0.841, 0.841, 0.868, 0.915] # Beta(5, 2)
gibbs_coverages = [0.999, 0.999, 0.999, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # Beta(5, 2)

plt.scatter(nom_coverages, exact_coverages, color = "blue", label = "Exact CI")
plt.scatter(nom_coverages, gibbs_coverages, color = "red", marker = "^", label = "GUI CS")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.title("Outliers Removed: Beta Sample")
plt.legend()
#plt.show()
plt.savefig("./cherrypicked_beta.png")
