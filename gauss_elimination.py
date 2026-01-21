import numpy as np

n=int(input("enter the size of the matrix"))


A = np.zeros((n,n), dtype=float)
b=np.zeros((1,n),dtype=float)
x=np.zeros((1,n),dtype=float)

for i in range(n):
    for j in range(n):
        A[i,j] = float(input("enter the element rowwise:"))
        
for i in range(n):
    b[1,i]=input("enter the values for the b matrix rowwise: ")

for k in range(n):
    pivot = A[k,k]
    
    for i in range(k+1,n):
        m = A[i,k]/pivot
        A[i,k:n] = A[i,k:n]-m*A[k,k:n]  

for i in range(n-1,-1,-1):
    x[i]=(b[i]-np.dot(A[i,i+1:],x[i+1:]))/A[i,i]


print(x)