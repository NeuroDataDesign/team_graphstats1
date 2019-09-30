import numpy as np
from graspy.plot.plot import heatmap
import matplotlib.pyplot as plt

arr = np.array([[1,2,3],[4,5,6],[7,8,9]])

title = 'test annotated heatmap'

fig = heatmap(arr, title=title, annot=True)
plt.savefig("annotated.png")