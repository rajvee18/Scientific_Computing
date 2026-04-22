import numpy as np
import matplotlib.pyplot as plt

# Function
def f(x):
    return x[0]**2 + x[1]**4

# Gradient
def grad_f(x):
    return np.array([2*x[0], 4*x[1]**3])

# Line search using Newton method
def find_alpha(x, p, tol=1e-6, max_iter=20):
    alpha = 0.0   # initial guess

    for _ in range(max_iter):
        # phi'(alpha)
        x_new = x + alpha * p
        grad = grad_f(x_new)
        phi_prime = np.dot(grad, p)

        # phi''(alpha)
        # Hessian
        H = np.array([[2, 0],[0, 12*x_new[1]**2]])
        phi_double = p.T @ H @ p

        # Avoid division by zero
        if abs(phi_double) < 1e-10:
            break

        alpha_new = alpha - phi_prime / phi_double

        if abs(alpha_new - alpha) < tol:
            return alpha_new

        alpha = alpha_new

    return alpha


# Steepest Descent Algorithm
def steepest_descent(x0, tol=1e-6, max_iter=100):
    x = x0.copy()
    history = [x.copy()]

    for _ in range(max_iter):
        grad = grad_f(x)

        # stopping condition
        if np.linalg.norm(grad) < tol:
            break

        p = -grad

        # compute optimal step size
        alpha = find_alpha(x, p)

        # update
        x = x + alpha * p
        history.append(x.copy())

    return np.array(history)


# Initial point
x0 = np.array([2.0, 1.5])

# Run algorithm
path = steepest_descent(x0)


plt.plot(path[:,0], path[:,1], 'o-', label='Steepest Descent Path')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Steepest Descent with Line Search')
plt.show()