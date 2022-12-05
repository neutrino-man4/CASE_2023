import numpy as np
import matplotlib.pyplot as plt

mi = np.random.exponential(scale=500, size=1000000)
mi = mi+1450
print(mi+1450)

rng = np.random.default_rng(12345)
rints = rng.integers(low=1450, high=7000, size=1000)

ys = []
for i in rints:
    ys.append((1./500)*np.exp(-(i-1450)/500.))

fig,ax = plt.subplots()
ax.set_yscale('log')
#plt.hist(mi,density=True)
plt.plot(rints,ys)
#ax.set_ylim([1e-2,1])
plt.savefig("/home/tier3/bmaier/public_html/figs/test.png")


