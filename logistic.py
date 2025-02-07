import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d # 3d plots

xs = st.norm.rvs(0, 1, 10000)
def p(x, theta):
    return 1/(1+np.exp(-theta[0] - theta[1]*x))
ys = [1 if np.random.rand() < p(x, [1, 1]) else -1 for x in xs]

theta0s = np.linspace(-100, 100, num = 100)
theta1s = np.linspace(-100, 100, num = 100)

omega = 1.1
expectations = np.zeros((len(theta0s), len(theta1s)))

precomputes = [(1 + np.exp(y*(1 + x)))**-2 for (x, y) in zip(xs, ys)]
max_val = 0
max_index = (0, 0)
for i, theta0 in enumerate(theta0s):
    print(i)
    for j, theta1 in enumerate(theta1s):
        e = np.mean([np.exp(-omega * ((1+np.exp(y*(theta0 + theta1*x)))**-2 - precompute)) for (x, y, precompute) in zip(xs, ys, precomputes)])
        #print(e)
        expectations[i][j] = e
        if e > max_val:
            max_val = e
            max_index = (i, j)

(i, j) = max_index
print(max_val, theta0s[i], theta1s[j])
ax = plt.figure().add_subplot(projection='3d')
X, Y = np.meshgrid(theta0s, theta1s)
ax.plot_surface(X, Y, expectations, edgecolor='royalblue', lw=0.5, alpha = 0.3)
plt.show()
