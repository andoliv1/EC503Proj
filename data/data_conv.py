import scipy.io
import numpy as np

mat = scipy.io.loadmat('kernel-svm-2rings.mat')
x = np.transpose(mat['x'])
y = mat['y']
data = np.hstack((x,y))
np.save('kernel-svm-2-rings', data)

mat = scipy.io.loadmat('iris.mat')
X_train = mat['X_data_train']
X_test = mat['X_data_test']
X = np.vstack((X_train, X_test))
print(X.shape)
print(mat)
Y_train = mat['Y_label_train']
Y_test = mat['Y_label_test']
Y = np.vstack((Y_train, Y_test))
data = np.hstack((X,Y))
np.save('iris', data)


