import matplotlib.pyplot as plt
import numpy as np

x = [1,2,3,4,5]
y = [1,2,3,2,1]

plt.figure(figsize=(10,4))
plt.plot(x, y)
plt.title('Title', y=1.20)
plt.tight_layout()
plt.show()