import numpy as np

n=int(input("enter the size of the matrix"))


A = np.zeros((n,n), dtype=float)
b=np.zeros((n),dtype=float)
x=np.zeros((n),dtype=float)
l=np.eye(n)
for i in range(n):
    for j in range(n):
        A[i,j] = float(input("enter the element rowwise:"))

'''for i in range(n):
    b[i]=float(input("enter the values for the b matrix rowwise: "))'''

for k in range(n):
    pivot = A[k,k]
    if pivot==0:
        print("error")
        break
    for i in range(k+1,n):
        m = A[i,k]/pivot
        A[i,k:n] = A[i,k:n]-m*A[k,k:n]  
        b[i]=b[i]-m*b[k]
        l[i,k]=m

print(A)
print(l)
