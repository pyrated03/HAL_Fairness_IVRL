import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv as sparse_inv


def admm_maximize_dep(Z, Y, S, H, L_Y, L_S, epsilon, p = 0, rho=1.0, max_iter=1000, tol=1e-4, sigma = 1):
    n = Z.shape[0]
    q = K_Z.shape[1]
    m = L_Y.shape[0]

    if p == 0:
        p = q-1
    # Initialize variables
    theta = np.zeros((p, q))

    # Create auxiliary variables
    U1 = np.zeros((m, 1))
    U2 = np.zeros((m, 1))

    # Define kernel function
    def kernel(x, y):
        return np.exp(-np.linalg.norm(x - y)**2 / (2.0 * sigma**2))

    # Compute kernel matrices
    K_Z = np.zeros((n, q))
    K_Y = np.zeros((n, n))
    K_S = np.zeros((n, n))
    for i in range(n):
        for j in range(q):
            K_Z[i, j] = kernel(Z[i], Z[j])
        for j in range(n):
            K_Y[i, j] = kernel(Y[i], Y[j])
            K_S[i, j] = kernel(S[i], S[j])
    K_Z = csr_matrix(K_Z)
    K_Y = csr_matrix(K_Y)
    K_S = csr_matrix(K_S)

    # Compute constants
    A1 = K_Z.dot(H).dot(L_Y)
    A2 = K_Z.dot(H).dot(L_S)
    B = K_Y.dot(L_Y) - K_S.dot(L_S)
    C = K_S.dot(L_S)

    # Define update functions
    def update_theta():
        term1 = sparse_inv(A1.T.dot(A1) + rho * np.eye(m)
                           ).dot(A1.T).dot(B + rho * (U1 - U2))
        term2 = sparse_inv(A2.T.dot(A2) + rho * np.eye(m)
                           ).dot(A2.T).dot(C + rho * (U2 - U1))
        return term1 - term2

    def update_u1():
        return U1 + rho * (theta.dot(K_Z.T).dot(H).dot(L_Y) - B - U2)

    def update_u2():
        return U2 + rho * (theta.dot(K_Z.T).dot(H).dot(L_S) - C - U1)

    # Define function to compute objective value
    def objective_value():
        return np.linalg.norm(theta.dot(K_Z).dot(H).dot(L_Y))**2 / (n**2)

    # Run ADMM algorithm
    for iter in range(max_iter):
        # Update theta
        theta_new = update_theta()

        # Check convergence
        if la.norm(theta_new - theta) < tol:
            break

        # Update auxiliary variables
        U1 = update_u1()
        U2 = update_u2()

        # Update theta
        theta = theta_new

        # Project theta onto feasible set
        U, S, V = np.linalg.svd(theta)
        S = np.minimum(
            S, epsilon / (2 * np.linalg.norm(K_Z.dot(H).dot(L_S))**2 / (n**2)))
        theta = U.dot(np.diag(S)).dot(V)

    return theta, objective_value()
