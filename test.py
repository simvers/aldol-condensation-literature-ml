import numpy as np
import pandas as pd
from mendeleev import element


a = {"A": 1, "B": 2, "C": 3}
print(a.get('A'))
x = pd.NA
y = a.get(x, 1)
print(5/y)

exit()

b = np.empty((0, 3))

a = np.array([1, 2, 3])
c = np.array([4, 5, 6])
print(a)

b = np.vstack((b, a))
print(np.vstack((b, c)))

b = np.empty((0, 3, 3))

a = np.array(
    [[1, 2, 3],
     [4, 5, 6],
     [6, 7, 8]]
)
print(a)

b = np.vstack((b, a[None, :, :]))
print(np.vstack((b, a[None, :, :])))

