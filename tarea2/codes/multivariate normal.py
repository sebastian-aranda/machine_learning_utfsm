import matplotlib.pyplot as plt
import numpy

mean = [0,0]
cov = [[1,0],[0,100]]
cov2 = [[1,0],[0,1]]
cov3 = [[30,0],[0,100]]

x,y = numpy.random.multivariate_normal(mean,cov3,5000).T
plt.plot(x,y,'bo')
plt.title("Cov1")
plt.axis('equal')
plt.show()

