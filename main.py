# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from scipy.optimize import minimize
import math

constraints = [
    {'type': 'ineq', 'fun': lambda x: (x[0] - 5)**2 + (x[1] - 5)**2 - 4},       # constraint1 ≥ 0
    {'type': 'ineq', 'fun': lambda x: x[0]**2 + (x[1] - 10)**2 - 1},            # constraint2 ≥ 0
    {'type': 'ineq', 'fun': lambda x: x[0]**2 + x[1]**2 - 1}                    # constraint3 ≥ 0
]

import math

def objective(flat_points, eps=1e-6):
    n = len(flat_points) // 2
    min_dist = float('inf')
    for i in range(n):
        xi, yi = flat_points[2*i], flat_points[2*i + 1]
        for j in range(i + 1, n):
            xj, yj = flat_points[2*j], flat_points[2*j + 1]
            dist = math.hypot(xi - xj, yi - yj)
            min_dist = min(min_dist, dist)

    # Return a function that increases with min_dist (to maximize it)
    return 1 / (1 + 1 / (min_dist + eps))

def optimize_efficiency(n):
    if (n == 2):
        bounds = []
        for i in range(2 * n):
            bounds.append((0, 10))
        x0 = [1, 1, 1, 1]
        print("deez nuts")
        return minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)


def testconstraints():
    print(constraint1(0, 10))
    print(constraint2(0, 10))
    print(constraint3(0, 10))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    result = optimize_efficiency(2)
    print(result)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
