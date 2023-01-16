import numpy as np

A = np.array(((0,0,1),
              (1,0,0),
              (0,1,0)))

B = np.array(((1,2,3),
              (4,5,6),
              (7,8,9)))

def isRotationMatrix(R):
    Rt = np.transpose(R)
    print(Rt)
    shouldBeIdentity = np.dot(Rt, R)
    
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    print(n)
    return n < 1e-6

print(isRotationMatrix(A))