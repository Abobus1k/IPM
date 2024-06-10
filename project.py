import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Дефолтные значения для примера
def solve_and_plot(m=1.0, p=0.5, k=4.0, y0=1.0, v0=0.0, t0=0.0, tf=10.0, dt=0.01):
    initial_conditions = [y0, v0]
    t = np.arange(t0, tf, dt)

    def derivatives(yv, t, m, p, k):
        y, v = yv
        dydt = v
        dvdt = -(p / m) * v - (k / m) * y
        return [dydt, dvdt]

    def runge_kutta_4th_order(derivatives, initial_conditions, t, m, p, k):
        n = len(t)
        yv = np.zeros((n, 2))
        yv[0] = initial_conditions
        for i in range(1, n):
            h = t[i] - t[i - 1]
            k1 = h * np.array(derivatives(yv[i - 1], t[i - 1], m, p, k))
            k2 = h * np.array(derivatives(yv[i - 1] + 0.5 * k1, t[i - 1] + 0.5 * h, m, p, k))
            k3 = h * np.array(derivatives(yv[i - 1] + 0.5 * k2, t[i - 1] + 0.5 * h, m, p, k))
            k4 = h * np.array(derivatives(yv[i - 1] + k3, t[i - 1] + h, m, p, k))
            yv[i] = yv[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return yv

    solution_rk = runge_kutta_4th_order(derivatives, initial_conditions, t, m, p, k)

    solution_odeint = odeint(derivatives, initial_conditions, t, args=(m, p, k))

    plt.figure(figsize=(10, 6))

    plt.plot(t, solution_rk[:, 0], label='y(t) RK', linestyle='--')
    plt.plot(t, solution_rk[:, 1], label='v(t) RK', linestyle='--')
    plt.plot(t, solution_odeint[:, 0], label='y(t) odeint')
    plt.plot(t, solution_odeint[:, 1], label='v(t) odeint')

    plt.xlabel('Time t')
    plt.ylabel('y, v')
    plt.title('Solution of the Differential Equation')
    plt.legend()
    plt.grid(True)
    plt.show()

solve_and_plot()
