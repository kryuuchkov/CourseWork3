#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import cos, exp, fabs, sin
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


# In[2]:


def diff_eq(x: float, y: float):
    return sin(x) * cos(x) - y * cos(x)


# In[3]:


def runge_kutta_ord4(func: Callable, x0: float, y0: float, x1: float, h: float):
    n = int((x1 - x0) / h)
    vx = [0] * (n + 1)
    vy = [0] * (n + 1)
    vx[0] = x = x0
    vy[0] = y = y0
    for i in range(1, n + 1):
        k1 = h * func(x, y)
        k2 = h * func(x + 0.5 * h, y + 0.5 * k1)
        k3 = h * func(x + 0.5 * h, y + 0.5 * k2)
        k4 = h * func(x + h, y + k3)
        vx[i] = x = x0 + i * h
        vy[i] = y = y + (k1 + k2 + k2 + k3 + k3 + k4) / 6
    return vx, vy


# In[4]:


def lagrange_polynomial(x, y, x_val):
    poly = 0
    for j in range(len(y)):
        p1 = 1
        p2 = 1
        for i in range(len(x)):
            if i == j:
                p1 = p1 * 1
                p2 = p2 * 1
            else:
                p1 = p1 * (x_val - x[i])
                p2 = p2 * (x[j] - x[i])
        poly += y[j] * p1 / p2
    return poly


# In[5]:


def f_x(x: float):
    return exp(-sin(x)) + sin(x) - 1                              


# In[6]:


if __name__ == "__main__":
    np.random.seed(42)
    y = 0
    a, b, h = 0, 1, 0.5
    fig = plt.figure()
    # 1 Runge-Kutta method (fourth-order Runge–Kutta method)
    # y' = sin(x)cos(x) - ycos(x); y(0) = 0, x at [a, b]
    x_k, y_k = runge_kutta_ord4(diff_eq, a, y, b, h)
    print("Runge-Kutta method vals:")
    for x, y in zip(x_k, y_k):
        print(f"x - {x}, y - {y}")
    plt.subplot(2, 2, 1)
    plt.plot(x_k, y_k)

    # 2 Building Interpolation polynom P_2(x), using solutions of table #1
    x_n = np.linspace(np.min(x_k), np.max(x_k), 100)
    P_x_n = [lagrange_polynomial(x_k, y_k, i) for i in x_n]
    plt.subplot(2, 2, 2)
    plt.plot(x_n, P_x_n)

    # 3 Finding max value of F(x) = |P_2(x) - f(x)| at [a, b]
    # f(x) = exp(-sin(x)) + sin(x) - 1
    x_res_n = x_n = np.linspace(a, b, 100)
    F_x = [fabs(lagrange_polynomial(x_k, y_k, x) - f_x(x)) for x in x_res_n]
    print(f"\nMax value of F(x) = |P_2(x) - f(x)| is {max(F_x)}")
    plt.subplot(2, 2, 3)
    plt.plot(x_res_n, F_x)
    plt.show()


# In[ ]:




