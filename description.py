import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Import clean data
data = pd.read_excel('data/data_clean.xlsx')
composition = pd.read_excel('data/composition_clean.xlsx')
print(data.head(10))


