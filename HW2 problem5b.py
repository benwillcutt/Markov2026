import numpy as np

rng = np.random.default_rng(1)
U = rng.random(100000)
Z = 1 - np.sqrt(1 - U)

print("sample mean:", Z.mean(), "theory:", 1/3)
print("sample sd:", Z.std(), "theory:", 1/(3*np.sqrt(2)))