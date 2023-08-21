import matplotlib.pyplot as plt
import numpy as np

def normalize(vector):
    vector /= np.sqrt(np.std(vector))
    vector -= np.mean(vector)
    return vector

size = 10000
t = np.linspace(0, size - 1, size)
print(t)
dummy_data_flat = np.zeros(size)
dummy_data_flat = dummy_data_flat + np.random.normal(0, 0.01, 10000)
dummy_data_rising = t * 0.00001
dummy_data_rising = dummy_data_rising + np.random.normal(0, 0.01, 10000)
dummy_data_falling = t * -0.00001
dummy_data_falling = dummy_data_falling + np.random.normal(0, 0.01, 10000)

plt.plot(dummy_data_falling)
plt.plot(dummy_data_flat)
plt.plot(dummy_data_rising)
plt.show()


norm_ris = normalize(dummy_data_rising)
norm_fal = normalize(dummy_data_falling)
norm_flat = normalize(dummy_data_flat)

plt.plot(norm_ris)
plt.plot(norm_flat)
plt.plot(norm_fal)

plt.show()

