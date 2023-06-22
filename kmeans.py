from sklearn.cluster import KMeans
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

mu1s = []
mu2s = []
mu3s = []
for mc_iter in range(100):
    sample_size = 100
    cov = 0.01
    """
    x1 = st.multivariate_normal.rvs(mean = [1, 0], cov = cov, size = sample_size*96//100)
    x2 = st.multivariate_normal.rvs(mean = [np.cos(2*np.pi/3), np.sin(2*np.pi/3)], cov = cov, size = sample_size*3//100)
    x3 = st.multivariate_normal.rvs(mean = [np.cos(4*np.pi/3), np.sin(4*np.pi/3)], cov = cov, size = sample_size*1//100)
    x = np.vstack([x1, x2, x3])
    mu = KMeans(n_clusters = 3, random_state = 0).fit(x).cluster_centers_
    print(mu)
    plt.scatter(x1[:,0], x1[:,1], color = "red")
    plt.scatter(x2[:,0], x2[:,1], color = "blue")
    plt.scatter(x3[:,0], x3[:,1], color = "green")
    plt.show()
    exit()
    """

    x = []
    for _ in range(sample_size):
        if np.random.rand() < .96:
            mu = [1, 0]
        elif np.random.rand() < .75:
            mu = [np.cos(2*np.pi/3), np.sin(2*np.pi/3)]
        else:
            mu = [np.cos(4*np.pi/3), np.sin(4*np.pi/3)]
        x.append(st.multivariate_normal.rvs(mean = mu, cov = cov))


    mu = KMeans(n_clusters = 3, random_state = 0).fit(x).cluster_centers_

    mu1 = mu[np.argmax(mu, axis = 0)[0]]
    mu = np.delete(mu, np.argmax(mu, axis = 0)[0], axis = 0)
    mu2 = mu[np.argmax(mu, axis = 0)[1]]
    mu3 = mu[np.argmin(mu, axis = 0)[1]]


    mu1s.append(mu1)
    mu2s.append(mu2)
    mu3s.append(mu3)

mu1s = np.array(mu1s)
mu2s = np.array(mu2s)
mu3s = np.array(mu3s)
print(np.mean(mu1s, axis = 0), np.cov(mu1s, rowvar = False))
print(np.mean(mu2s, axis = 0), np.cov(mu2s, rowvar = False))
print(np.mean(mu3s, axis = 0), np.cov(mu3s, rowvar = False))
plt.scatter(mu3s[:,0], mu3s[:,1], color = "cyan")
plt.scatter(mu2s[:,0], mu2s[:,1], color = "green")
plt.scatter(mu1s[:,0], mu1s[:,1], color = "red")
plt.scatter([1, -0.5, -0.5], [0, 3**0.5/2, -3**0.5/2], color = "black")
plt.show()
