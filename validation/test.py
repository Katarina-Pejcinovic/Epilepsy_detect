import numpy as np
pred = np.ones([32, 2])
pred = np.amax(pred, axis=1)
print(pred.shape)
