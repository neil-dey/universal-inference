import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import sys

np.random.seed(0)

dist = int(sys.argv[1])

def online_gue(data, true_value, alpha):
    boot_iters = 100
    coverages = []
    omegas = np.linspace(1, 100, num = 1000)
    omegas = omegas[::-1]

    omega_hats = np.zeros(len(data))
    for n in range(1, len(data)):
        coverages = np.zeros(len(omegas))
        for _ in range(boot_iters):
            boot_data = data[np.random.choice(len(data), len(data), replace = True)]

            boot_erms = [np.mean(boot_data[0: i+1]) for i in range(len(boot_data))]
            boot_erms = [1] + boot_erms
            boot_mu = boot_erms[-1]

            excess_losses = [(thetahat - x)**2 - (boot_mu - x)**2 for (thetahat, x) in zip(boot_erms, boot_data)]
            log_gue_over_omega = sum([-1*excess_losses[i] for i in range(n)])
            for idx, omega in enumerate(omegas):
                log_gue = omega*log_gue_over_omega
                coverages[idx] += log_gue < np.log(1/alpha)
        coverages /= boot_iters
        omega_hats[n - 1] = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]

    erms = [np.mean(data[0: i+1]) for i in range(len(data))]
    erms = [0.5] + erms
    excess_losses = [(thetahat - x)**2 - (mu - x)**2 for (thetahat, x) in zip(erms, data)]
    return sum([-1*omega_hat * excess_loss for (omega_hat, excess_loss) in zip(omega_hats, excess_losses)]) < np.log(1/alpha)

if dist == 0:
    P = st.triang(c=0.5, scale = 2)
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
