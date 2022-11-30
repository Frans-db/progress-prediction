import math
import random
import matplotlib.pyplot as plt

l,u = 0,1
m = (l + u) / 2
r = (u - l) / 2

def energy(p: float, p_hat: float) -> float:
    real = (p - m) / (r * math.sqrt(2))
    predicted = (p_hat - m) / (r * math.sqrt(2))
    return min(1, real**2 + predicted**2)

reals = [i / 100 for i in range(1, 101)]
predictions = [(i / 100) + (random.random() - 0.5) for i in range(1, 101)]
energies = [energy(real, prediction) for (real, prediction) in zip(reals, predictions)]

losses = [e * abs(r - p) for (e,r,p) in zip(energies, reals, predictions)]

avg_loss = sum(losses) / len(losses)
print('Average Loss:', avg_loss)

plt.plot(reals, label='Actual')
plt.plot(predictions, label='Predicted')
plt.plot(energies, label='Energies')
plt.plot(losses, label='Losses')
plt.legend(loc='best')
plt.show()
