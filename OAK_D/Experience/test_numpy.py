import numpy as np
from scipy.sparse import csr_matrix
import time

# Capture program start time
start_time = time.perf_counter()


def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.indices, data.shape) for row in M]

def get_indice_dense(data):
    return np.where(data > 1)

N = 2
shape = (1000,1000)
data = np.random.randint(0,N+1,shape)
indx = get_indice_dense(data)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(indx)
print(execution_time, " secondes")

