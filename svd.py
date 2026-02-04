import numpy as np

def eigen_decomp(A,symmetric=False):
    A=np.array(A)
    if symmetric:
        vals,vecs = np.linalg.eigh(A)
    else:
        vals,vecs = np.linalg.eig(A)
        
    idx = np.argsort(-np.abs(vals))

    return vals[idx],vecs[:,idx]

def SVD(A):
    A=np.array(A)
    A_transpose_A = A.T @ A

    eign_values,eign_vectors = eigen_decomp(A_transpose_A)
    singular_values = np.sqrt(eign_values)
    sigma=np.zeros((A.shape[0],A.shape[1]))
    sigma = np.diag(singular_values)
    
    V = eign_vectors.T
    U=np.zeros((A.shape[0],len(singular_values)))
    for i,s in enumerate(singular_values):
        U[:,i]= (A @ V[:,i])/s

    return U,sigma,V.T

m=int(input("enter the number of rows of the matrix:"))
n=int(input("enter the number of columns of the matrix"))
A = np.zeros((m,n), dtype=float)
for i in range(m):
    for j in range(n):
        A[i,j] = float(input("enter the element rowwise :"))

U,sigma,V_transpose = SVD(A)

print(U)
print(sigma)
print(V_transpose)






    