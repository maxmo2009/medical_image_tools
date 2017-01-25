import numpy as np 




A = np.array([1,2,3])

B = np.array([[1,1,1],
	          [1,1,1],
	          [1,1,1]])

C = np.array([[1,1,1],
	          [1,2,1],
	          [1,1,1]])


BC = np.dot(B,C)

print np.dot(A,BC)