import numpy as np
import scipy.stats as st
from scipy.optimize import root_scalar
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(0)

# 1 dimensional
mu0 = 5
mu1 = 10
cov = 10000
c = (mu0 + mu1)/2


omega = 0.05

nom_coverages = np.linspace(0, 1, num=10)[5:-1]
exact_coverages = []
universal_coverages = []

import sys
def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, "#"*x, "."*(size-x), j, count),
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

for nom_coverage in nom_coverages:
    print("Nominal coverage:", round(nom_coverage, 2))
    exact_coverage = 0
    universal_coverage = 0
    mc_iters = 100
    exact_sample_sizes = []
    universal_sample_sizes = []
    for _ in range(mc_iters):
        # Exact confidence interval
        data0 = st.norm.rvs(loc = mu0, scale = cov**0.5, size = 2)
        data1 = st.norm.rvs(loc = mu1, scale = cov**0.5, size = 2)
        n0 = 2
        n1 = 2
        while True:
            # Point estimate for (mu0 + mu1)/2
            c_est = (np.mean(data0) + np.mean(data1))/2

            # (1-\alpha)100% confidence interval for (mu0 + mu1)/2
            z_alpha = st.norm.interval(nom_coverage)[1]
            CI = [c_est - z_alpha* ((n0 + n1)/(4*n0*n1) * cov)**0.5, c_est + z_alpha * ((n0 + n1)/(4*n0*n1) * cov)**0.5]

            # If the confidence interval still includes the origin, collect more data and try again
            if CI[0] <= 0 and 0 <= CI[1]:
                if np.random.rand() < 0.5:
                    data0 = np.append(data0, st.norm.rvs(loc = mu0, scale = cov**0.5, size = 1))
                    n0 += 1
                else:
                    data1 = np.append(data1, st.norm.rvs(loc = mu1, scale = cov**0.5, size = 1))
                    n1 += 1
                continue

            # Otherwise, we stop collecting data and check if the confidence interval contains the true (mu0+mu1)/2
            if CI[0] <= c and c <= CI[1]:
                exact_coverage += 1
            break
        exact_sample_sizes.append(n0+n1)

        # Universal interval
        # First, get 1 point each of training and testing data from each distribution
        data0_init_train = st.norm.rvs(loc = mu0, scale = cov**0.5, size = 1)[0]
        data0_test = st.norm.rvs(loc = mu0, scale = cov**0.5, size = 1)
        data1_init_train = st.norm.rvs(loc = mu1, scale = cov**0.5, size = 1)[0]
        data1_test = st.norm.rvs(loc = mu1, scale = cov**0.5, size = 1)

        # train_data consists of tuples (theta, label, risk), with "label" being 0 or 1 whether theta came N(mu0, cov) or N(mu1, cov), and "risk" being the number of errors theta makes on the training data. It is sorted in order of ascending theta.
        # train_data_raw only contains the theta values to speed up some operations
        c_est = data0_init_train
        if data0_init_train < data1_init_train:
            train_data = np.array([[data0_init_train, 0, 0], [data1_init_train, 1, 1]])
            train_data_raw = np.array([data0_init_train, data1_init_train])
        else:
            train_data = np.array([[data1_init_train, 1, 2], [data0_init_train, 0, 1]])
            train_data_raw = np.array([data1_init_train, data0_init_train])

        n0 = 2
        n1 = 2
        while True:
            def emp_risk_test(theta):
                return len([x for x in data1_test if x <= theta]) + len([x for x in data0_test if x > theta])

            def T(theta, omega):
                return (omega * (emp_risk_test(c_est) - emp_risk_test(theta)))

            # Check that confidence set contains zero
            if T(0, omega) >= np.log(1 - nom_coverage):
                # With 50% probability, generate a new data point for the test set. Each distribution then gets another 50/50 shot.
                if np.random.rand() < 0.5:
                    if np.random.rand() < 0.5:
                        new_point = st.norm.rvs(loc = mu0, scale = cov**0.5, size = 1)[0]
                        idx = np.searchsorted(data0_test, new_point)
                        data0_test = np.insert(data0_test, idx, new_point)
                        n0 += 1
                    else:
                        new_point = st.norm.rvs(loc = mu1, scale = cov**0.5, size = 1)[0]
                        idx = np.searchsorted(data1_test, new_point)
                        data1_test = np.insert(data1_test, idx, new_point)
                        n1 += 1
                # With 50% probability we instead generate a new data point for the training set
                else:
                    if np.random.rand() < 0.5:
                        new_point = st.norm.rvs(loc = mu0, scale = cov**0.5, size = 1)[0]
                        idx = np.searchsorted(train_data_raw, new_point)
                        if idx == len(train_data):
                            train_data = np.insert(train_data, idx, np.array([new_point, 0, train_data[-1][2]]), axis = 0)
                        else:
                            error_count = train_data[idx][2] + 1 if train_data[idx][1] == 0 else train_data[idx][2] - 1
                            train_data = np.insert(train_data, idx, np.array([new_point, 0, error_count]), axis = 0)
                        for i in range(idx):
                            train_data[i][2] += 1
                        train_data_raw = np.insert(train_data_raw, idx, new_point)
                        n0 += 1
                    else:
                        new_point = st.norm.rvs(loc = mu1, scale = cov**0.5, size = 1)[0]
                        idx = np.searchsorted(train_data_raw, new_point)
                        if idx == 0:
                            error_count = train_data[idx][2] if train_data[idx][1] == 1 else train_data[idx][2] + 2
                            train_data = np.insert(train_data, idx, np.array([new_point, 1, error_count]), axis = 0)
                        else:
                            train_data = np.insert(train_data, idx, np.array([new_point, 1, train_data[idx-1][2] + 1]), axis = 0)
                        for i in range(idx + 1, len(train_data)):
                            train_data[i][2] += 1
                        train_data_raw = np.insert(train_data_raw, idx, new_point)
                        n1 += 1
                c_est = train_data_raw[np.argmin([x[2] for x in train_data])]
                continue

            # Check that confidence set contains c
            if T(c, omega) >= np.log(1 - nom_coverage):
                universal_coverage += 1
            universal_sample_sizes.append(n0+n1)
            break

    universal_coverages.append(universal_coverage/mc_iters)
    exact_coverages.append(exact_coverage/mc_iters)
    print("Average sample sizes:", np.mean(exact_sample_sizes), np.mean(universal_sample_sizes))
    print("Exact coverages:")
    print(exact_coverages)
    print("Universal coverages")
    print(universal_coverages)

plt.title("Normally Distributed Data")
plt.scatter(nom_coverages, exact_coverages, color="blue", label = "Exact CI")
#plt.scatter(nom_coverages, universal_coverages, color = "red", label = "Universal CI")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.legend()
plt.show()
