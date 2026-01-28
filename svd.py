import numpy as np

def eigen_decomp(A,symmetric=False):
    A=np.array(A)
    if symmetric:
        vals,vecs = np.linalg.eigh(A)
    else:
        vals,vecs = np.linalg.eig(A)
        
    idx = np.argsort(-np.abs(vals))

    return vals[idx],vecs[:,idx]

def normalise(v):
    norm = np.linalg.norm(v)
    if norm==0:
        return v
    else:
        return v/norm

def SVD(A):
    A=np.array(A)
    A_transpose_A = A.T @ A

    eign_values,eign_vectors = eigen_decomp(A_transpose_A)
    singular_values = np.sqrt(eign_values)

    sigma = np.diag(singular_values)

    vecs = [normalise(i) for i in eign_vectors]


    