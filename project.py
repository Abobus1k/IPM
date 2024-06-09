import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, diff, sympify, lambdify, sin, cos, exp
from scipy.integrate import odeint


def main():
    x, u, t = symbols('x u t')

    map = {
        "1": "sin(x) * cos(u) + exp(t) - x * u",
        "2": "x**2 + u**2 + t**2",
    }

    print("Choose a function to linearize:")
    for key, func in map.items():
        print(f"{key}: {func}")
    choice = input("Enter the number of the function: ")
    f = sympify(map.get(choice, map["1"]))

    df_dx = diff(f, x)
    df_du = diff(f, u)
    df_dt = diff(f, t)

    x0 = float(input("Enter the linearization point for x: "))
    u0 = float(input("Enter the linearization point for u: "))
    t0 = float(input("Enter the linearization point for t: "))

    f_0 = f.subs({x: x0, u: u0, t: t0})
    df_dx_0 = df_dx.subs({x: x0, u: u0, t: t0})
    df_du_0 = df_du.subs({x: x0, u: u0, t: t0})
    df_dt_0 = df_dt.subs({x: x0, u: u0, t: t0})

    f_lin = f_0 + (x - x0) * df_dx_0 + (u - u0) * df_du_0 + (t - t0) * df_dt_0

    f_np = lambdify((x, u, t), f, 'numpy')
    f_lin_np = lambdify((x, u, t), f_lin, 'numpy')

    range_size = 1
    x_vals = np.linspace(x0 - range_size, x0 + range_size, 100)
    u_vals = np.linspace(u0 - range_size, u0 + range_size, 100)
    t_vals = np.linspace(0, 10, 100)
    plot_integrated_functions(f_np, f_lin_np, x0, t_vals)


def ode_system(f_np, x0, t_vals, u_func):
    def system(x, t):
        u = u_func(t)
        return f_np(x, u, t)

    x_vals = odeint(system, x0, t_vals)
    return x_vals

def plot_integrated_functions(f_np, f_lin_np, x0, t_vals):
    u0_func = lambda t: 0
    x_orig_u0 = ode_system(f_np, x0, t_vals, u0_func)
    x_lin_u0 = ode_system(f_lin_np, x0, t_vals, u0_func)

    plt.plot(t_vals, x_orig_u0, label='Original f, u=0', linestyle='-', color='blue')
    plt.plot(t_vals, x_lin_u0, label='Linearized f, u=0', linestyle='--', color='blue')

    plt.legend()
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('Comparison of Integrated Functions (u=0)')
    plt.grid()
    plt.show()

    # u=0.1*sin(t)
    u1_func = lambda t: 0.1 * np.sin(t)

    x_orig_u1 = ode_system(f_np, x0, t_vals, u1_func)
    x_lin_u1 = ode_system(f_lin_np, x0, t_vals, u1_func)

    plt.plot(t_vals, x_orig_u1, label='Original f, u=0.1*sin(t)', linestyle='-', color='red')
    plt.plot(t_vals, x_lin_u1, label='Linearized f, u=0.1*sin(t)', linestyle='--', color='red')

    plt.legend()
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('Comparison of Integrated Functions (u=0.1*sin(t))')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()