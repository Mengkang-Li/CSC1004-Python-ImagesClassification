import matplotlib.pyplot as plt
import numpy as np

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
performances = [0.9200, 0.9280, 0.9320, 0.9400, 0.9450, 0.9520, 0.9570, 0.9650, 0.9720, 0.9760, 0.9780, 0.9798, 0.9799,
                0.9800, 0.9800]
plt.figure(figsize=(8, 6))  # 一定要放在plot之前
epochs = np.array(epochs)
performances = np.array(performances)
picture = plt.plot(epochs, performances)

plt.xlabel("epochs")
plt.ylabel("performances")
plt.title("performances-epochs picture", fontsize=20)
plt.show()