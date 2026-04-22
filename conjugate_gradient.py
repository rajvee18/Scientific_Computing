import numpy as np

def conjugate_gradient(A, b, x0=None, tol=1e-6, max_iter=100):
    n = len(b)

    # Initial guess
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Step 1
    r = b - A @ x
    p = r.copy()

    history = [x.copy()]

    for k in range(max_iter):

        # Step 2(a): stopping condition
        if np.linalg.norm(r) < tol * np.linalg.norm(b):
            print(f"Converged in {k} iterations")
            break

        # Step 2(b): alpha
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)

        # Step 2(c): update x
        x = x + alpha * p

        # Step 2(d): update r
        r_new = r - alpha * Ap

        # Step 2(e): beta
        beta = (r_new @ r_new) / (r @ r)

        # Step 2(f): update p
        p = r_new + beta * p

        r = r_new

        history.append(x.copy())

    return x, np.array(history)


# Example symmetric positive definite matrix
A = np.array([[4, 1],
              [1, 3]])

b = np.array([1, 2])

x, history = conjugate_gradient(A, b)

print("Solution:", x)