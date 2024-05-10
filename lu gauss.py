import numpy as np

def lu_solve(A, b):
    """
    Solves the linear equation Ax = b using LU decomposition with partial pivoting.

    Args:
    A (numpy array): The coefficient matrix of the linear equation.
    b (numpy array): The constant vector of the linear equation.

    Returns:
    numpy array: The solution vector x of the linear equation.
    """

    # Get the shape of A
    n = A.shape[0]

    # Initialize L and U matrices
    L = np.identity(n)
    U = A.copy()

    # Perform LU decomposition with partial pivoting
    for i in range(n):
        # Find the index of the largest element in the current column
        pivot_index = np.argmax(np.abs(U[i:, i])) + i

        # Swap rows if necessary
        if pivot_index != i:
            U[[i, pivot_index]] = U[[pivot_index, i]]
            L[[i, pivot_index]] = L[[pivot_index, i]]

        # Update the U matrix
        for j in range(i+1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i] = 0

        # Update the U matrix
        for j in range(i+1, n):
            for k in range(i+1, n):
                U[j, k] -= L[j, i] * U[i, k]

    # Solve the lower and upper triangular systems
    y = np.zeros(n)
    for i in range(n-1, -1, -1):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    x = np.zeros(n)
    for i in range(n):
        x[i] = y[i] - np.dot(U[i, i+1:], x[i+1:])
        x[i] /= U[i, i]

    return x

A = np.array([[1,1,-1],
              [2,2,1],
              [-1,1,1]])

b = np.array([1,5,1])

x = lu_solve(A, b)
print(x)