import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
R = 1000
N_target = 10000

C = np.ones(R)
N = 3
while N < N_target:
    p = C / N
    C = C + (rng.random(R) < p)
    N += 1

z = C / N_target

mean_emp, sd_emp = z.mean(), z.std()
mean_th, sd_th = 1/3, 1/(3*np.sqrt(2))

print(f"empirical mean = {mean_emp:.5f}, theory = {mean_th:.5f}")
print(f"empirical SD/mean = {sd_emp/mean_emp:.5f}, theory = {sd_th/mean_th:.5f}")
print(f"smallest core = {C.min():.0f}, largest core = {C.max():.0f}")

zz = np.linspace(0, 1, 500)
plt.hist(z, bins=50, density=True, alpha=0.6, label='Simulated z=C/N')
plt.plot(zz, 2*(1-zz), 'r-', label='h(z)=2(1-z)')
plt.xlabel('z = C/N'); plt.ylabel('density'); plt.legend()
plt.savefig('HW2 problem5c.png', dpi=140)