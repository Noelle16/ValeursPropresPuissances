import matplotlib.pyplot as plt
import numpy as np
X, Y = [], []
for line in open('errors.txt', 'r'):
  values = [float(s) for s in line.split()]
  Y.append(values[0])
  X = list(range(0, len(Y)))
plt.title('Erreurs MCCA k = 10')
plt.plot(X, Y)
plt.show()