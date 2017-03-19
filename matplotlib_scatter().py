import numpy as np
import matplotlib .pyplot as plt
N=50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N)) **2  # area of circle
plt.scatter(x,y,s=area,c=colors,alpha=0.8) # (x co-ordi,y co-ordi,size,area,alpha)

plt.plot(x,y,color='blue',linewidth=1)   # (x co-ordi,y co-ordi,color of line,width of line)
plt.show()
#plt.show()