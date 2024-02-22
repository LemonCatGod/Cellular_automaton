from math import erfc, pow, exp, pi, fabs, factorial, sqrt, gamma
from decimal import Decimal as Dl

P_CONST = 0.01


def rect_integral(f, xmin, xmax, n=10000):
    dx = (xmax - xmin) / n
    area = 0
    x = xmin
    for i in range(n):
        area += dx * f(x)
        x += dx
    return area


def tr_integral(f, xmin, xmax, n=10000):
    dx = (xmax - xmin) / n
    area = 0
    x = xmin
    for i in range(n):
        area += dx * (f(x) + f(x + dx)) / 2
        x += dx
    return area


def sign(x):
    return 1 if x >= 0 else -1


def erfc_sys(x):
    return erfc(x)


def erfc_mine(x):
    t = 1 / (1 + fabs(x) / 2)
    return t * exp(
        -pow(x, 2) - 1.26551223 - 1.00002365 * t - 0.37409196 * pow(t, 2) + 0.09678418 * pow(t, 3) - 0.18628806 * pow(t,
        4) + 0.27886807 * pow(t, 5) - 1.13520398 * pow(t, 6) + 1.48851587 * pow(t, 7) - 0.82215223 * pow(t, 8))


def erfc_asymp(x):
    a = sum([(-1) ** i * factorial(2 * i) / (factorial(i) * pow(2 * x, 2 * i)) for i in range(0, 50)])
    return exp(-1 * pow(x, 2)) / (x * sqrt(pi)) * a


def first_test(path):
    f = open(path, "r")
    arr_numbers = []
    for i in f.read():
        a = int(i)
        arr_numbers.append(a if a == 1 else -1)
    f.close()
    total_value = sum(arr_numbers)
    total_length = len(arr_numbers)
    S_obs = total_value / (total_length ** 0.5)
    print("First test")
    p_value = erfc_sys(S_obs / (2 ** 0.5))
    p_value_2 = erfc_mine(S_obs / (2 ** 0.5))
    p_value_4 = erfc_asymp(S_obs / (2 ** 0.5))
    print(S_obs, p_value, p_value_2, p_value_4)
    return p_value >= P_CONST


def lower_gamma(s, x):
    lower_gamma = tr_integral(lambda t: pow(t, s - 1) + exp(-t), 0, x)
    return lower_gamma

def lower_gamma_2(s, x, q=10):
    lower_gamma = 0
    for i in range(q):
        lower_gamma += pow(x, i) / (gamma(s+i+1))
    return pow(x, s)*gamma(s)*exp(-x)*lower_gamma


def upper_gamma(s, x):
    return (gamma(s) - lower_gamma_2(s, x))/gamma(s)


def hi_quadro(arr, M):
    return 4 * M * sum(map(lambda x: pow(x - 0.5, 2), arr))


def second_test(path, M):
    print("2 test")
    f = open(path, "r")
    PP = []
    counter = 0
    number = 0
    for i in f.read():
        counter += 1
        a = int(i)
        number += a
        if counter % M == 0:
            PP.append(number / M)
            number = 0
    f.close()
    hi = hi_quadro(PP, M)
    p_value = upper_gamma(len(PP) / 2, hi / 2)
    print(hi, p_value)
    return p_value

def third_test(path):
    print("3 test")
    f = open(path, "r")
    b = 2
    one_value = 0
    n = 0
    V_value = 0
    for i in f.read():
        n += 1
        a = int(i)
        V_value += 1 if b != a else 0
        one_value += a
        b = a
    one_value /= n
    f.close()
    if abs(one_value - 0.5) < 2 / sqrt(n):
        print(abs(one_value - 0.5) < 2 / sqrt(n))
        print(V_value)
        res = abs(V_value - 2*n*one_value*(1-one_value))/(2*sqrt(2*n)*one_value*(1-one_value))
        p_value = erfc_sys(res)
        p_value_2 = erfc_mine(res)
        print(p_value, p_value_2)
        return p_value > P_CONST
    else:
        return False

def fourth_test(path):
    print("4 test")
    f = open("1.txt", "r")
    prev_value = 0
    one_value = 0
    n = 0
    values = []
    for i in f.read():
        n += 1
        a = int(i)
        one_value = one_value + 1 if a == 1 else 0
        prev_value = max(one_value, prev_value)
        if n % 8 == 0:
            values.append(prev_value)
            one_value = 0
            prev_value = 0
    f.close()
    v = [0, 0, 0, 0]
    print(values)
    for i in values:
        if i <= 1:
            v[0] += 1
        elif i == 2:
            v[1] += 1
        elif i == 3:
            v[2] += 1
        else:
            v[3] += 1
    p = [0.2148, 0.3672, 0.2305, 0.1875]
    R = 16
    hi = 0
    print(v)
    for i in range(4):
        hi += pow(v[i] - R*p[i], 2)/(R*p[i])
    p_value = lower_gamma_2(3/2, hi/2)
    print(hi, p_value)
    return p_value > P_CONST

first_test("1.txt")
second_test("1.txt", 3)
third_test("1.txt")
fourth_test("1.txt")