# Creating line y=x+2 and seperating the points.
#Taking arbitrary point a,b and plotting left or right side


import numpy as np
import matplotlib .pyplot as plt
N = 50
x = 50*np.random.rand(N)
y = 2 +50* np.random.rand(N)

plt.scatter(x,y,) # (x co-ordi,y co-ordi,size,area,alpha)

plt.plot(x,x+2,color='blue',linewidth=1)
a=50*np.random.rand(1)
b=2 + 50*np.random.rand(1)
plt.scatter(a,b,c='red',s=50)
plt.show()
#plt.scatter(x,y<x+2,s=area,c=colors,alpha=0.5) # (x co-ordi,y co-ordi,size,area,alpha)

#plt.show()

