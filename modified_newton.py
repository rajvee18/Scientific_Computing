import numpy as np
import matplotlib.pyplot as plt

# Function
def f(x):
    return x[0]**2 + x[1]**4

# Gradient
def grad_f(x):
    return np.array([2*x[0], 4*x[1]**3])

# Hessian
def hessian_f(x):
    return np.array([[2, 0],
                     [0, 12*x[1]**2]])

# Line search using Newton method
def find_alpha(x, p, tol=1e-6, max_iter=50):
    alpha = 1.0

    for _ in range(max_iter):
        x_new = x + alpha * p

        grad = grad_f(x_new)
        phi_prime = np.dot(grad, p)

        H = hessian_f(x_new)
        phi_double = p.T @ H @ p

        if abs(phi_double) < 1e-10:
            break

        alpha_new = alpha - phi_prime / phi_double

        if abs(alpha_new - alpha) < tol:
            return alpha_new

        alpha = alpha_new

    return alpha


# Modified Newton Method
def modified_newton(x0, tol=1e-6, max_iter=50, epsilon=1e-6):
    x = x0.copy()
    history = [x.copy()]

    for _ in range(max_iter):
        grad = grad_f(x)

        # stopping condition
        if np.linalg.norm(grad) < tol:
            break

        H = hessian_f(x)

        # Regularization (important!)
        H_mod = H + epsilon * np.eye(2)

        # Newton direction
        p = -np.linalg.solve(H_mod, grad)

        # Line search
        alpha = find_alpha(x, p)

        # Update
        x = x + alpha * p

        history.append(x.copy())

    return np.array(history)


# Initial point
x0 = np.array([2.0, 1.5])

# Run algorithm
path = modified_newton(x0)

# Plot iterates
plt.plot(path[:,0], path[:,1], 's-', label='Modified Newton Path')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Modified Newton Method with Line Search')
plt.grid()
plt.show()