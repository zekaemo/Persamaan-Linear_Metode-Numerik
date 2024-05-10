import numpy as np

def inversmax(A,b):
    
    A_inv = np.linalg.inv(A)
    solution = np.dot(A_inv,b)
    return solution


A = np.array([[1,1,-1],
              [2,2,1],
              [-1,1,1]])

b = np.array([1,5,1])

# Compute the inverse of the coefficient matrix A
solution=inversmax(A,b)
print(solution)