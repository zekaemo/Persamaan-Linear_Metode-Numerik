import numpy as np

def crout_decomposition(A):
    """
    Performs Crout decomposition on the given matrix A.

    Args:
    A (numpy array): The matrix to decompose.

    Returns:
    tuple: A tuple containing the L and U matrices.
    """

    # Get the shape of A
    n = A.shape[0]

    # Initialize L and U matrices
    L = np.identity(n)
    U = np.zeros(A.shape)

    # Perform Crout decomposition
    for i in range(n):
        for j in range(i, n):
            # Compute the diagonal element of U
            if i == j:
                U[i, i] = A[i, i] - np.dot(L[i, :i], U[:i, i])

            # Compute the subdiagonal element of U
            if i < j:
                U[j, i] = A[j, i] - np.dot(L[j, :i], U[:i, i])

        # Compute the subdiagonal element of L
        for j in range(i+1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

    return L, U

def crout_solve(A, b):
    """
    Solves the linear equation Ax = b using Crout decomposition.

    Args:
    A (numpy array): The coefficient matrix of the linear equation.
    b (numpy array): The constant vector of the linear equation.

    Returns:
    numpy array: The solution vector x of the linear equation.
    """

    # Perform Crout decomposition
    L, U = crout_decomposition(A)

    # Solve the lower and upper triangular systems
    y = np.zeros(L.shape[0])
    for i in range(L.shape[0]):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    x = np.zeros(U.shape[0])
    for i in range(U.shape[0]-1, -1, -1):
        x[i] = y[i] - np.dot(U[i, i+1:], x[i+1:])
        x[i] /= U[i, i]

    return x

A = np.array([[1,1,-1],
            [2,2,1],
            [-1,1,1]])

b = np.array([1, 5,1])
x = crout_solve(A, b)
print(x)

