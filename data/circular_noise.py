from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from numpy.random import randint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

s = 100
nx = np.linspace(-100,100,s)
ny = np.linspace(-100,100,s)
init_1 = 0
init_2 = 0

radius = 75

prob_inside = 0.9
prob_outside = 0.1

labels = np.zeros(s**2)
data = np.zeros((s**2,2))

counter1 = 0
for i in nx:
    counter2 = 0
    for j in ny:
        data[(counter1)*s + counter2] = np.array([[i,j]])
        number = randint(1,100,1)
        if(i**2 + j**2 > 75**2):
            if(number >= 15):
                labels[(counter1)*s + counter2] = -1
            else:
                labels[(counter1)*s + counter2] = 1
        else:
            if(number >= 85):
                labels[(counter1)*s + counter2] = -1
            else:
                labels[(counter1)*s + counter2] = 1
                
        counter2 += 1
    counter1 += 1

labels = np.array([labels])
labels = np.transpose(labels)

print(labels.shape)
print(data.shape)

data = np.concatenate((data,labels),axis=1)
np.savetxt("circle_noise.csv",data,delimiter=",")

plt.scatter(data[:,0],data[:,1],labels)
plt.show()