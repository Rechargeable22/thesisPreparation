import matplotlib.pyplot as plt
import numpy as np
import random
import csv

ROWS_PER_FUNC = 1000
COLS_PER_FUNC = 10


def f(x):
    return x

def g(x):
    return np.cos(x)

def h(x):
    return np.power(x, 2)

def i(x):
    return np.exp(x)

def plot_functions():
    x = np.arange(0, 2.0, 0.02)
    fig, ax = plt.subplots()
    ax.plot(x, f(x), "b", label='x')
    ax.plot(x, g(x), "r", label='cos(x)')
    ax.plot(x, h(x), "g", label='x^2')
    ax.plot(x, i(x), "k", label='e^x')
    ax.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(False)
    # plt.savefig("test.png")
    plt.show()

''' 400 datasets should be sufficient
    100 per function
    each consisting of 10 randomly distributed points
    then saves them as a csv file with labels
    !! works only on the [0,2] range so far    
'''
def generate_data():
    all = np.empty((0, COLS_PER_FUNC), dtype=float)
    for i in range(0, 4):
        arr = gen_arr_for_fun_num(i)
        all = np.append(all, arr, axis=0)
    return all

def gen_arr_for_fun_num(i):
    if i == 0:
        return gen_arr_f()
    elif i == 1:
        return gen_arr_g()
    elif i == 2:
        return gen_arr_h()
    else:
        return gen_arr_i()

def gen_arr_f():
    arr = np.zeros((ROWS_PER_FUNC, COLS_PER_FUNC), dtype=float)
    for row in range(0, ROWS_PER_FUNC):
        for i in range(0, COLS_PER_FUNC):
            arr[row, i] = f( random.uniform(0.2*i, 0.2*i + 0.2) )
        arr[row, COLS_PER_FUNC] = 'f'
    return arr

def gen_arr_g():
    arr = np.zeros((ROWS_PER_FUNC, COLS_PER_FUNC), dtype=float)
    for row in range(0, ROWS_PER_FUNC):
        for i in range(0, COLS_PER_FUNC):
            arr[row, i] = g( random.uniform(0.2*i, 0.2*i + 0.2) )
        arr[row, COLS_PER_FUNC] = 'g'
    return arr

def gen_arr_h():
    arr = np.zeros((ROWS_PER_FUNC, COLS_PER_FUNC), dtype=float)
    for row in range(0, ROWS_PER_FUNC):
        for i in range(0, COLS_PER_FUNC):
            arr[row, i] = h( random.uniform(0.2*i, 0.2*i + 0.2) )
        arr[row, COLS_PER_FUNC] = 'h'
    return arr

def gen_arr_i():
    arr = np.zeros((ROWS_PER_FUNC, COLS_PER_FUNC), dtype=float)
    for row in range(0, ROWS_PER_FUNC):
        for k in range(0, COLS_PER_FUNC):
            arr[row, k] = i( random.uniform(0.2*k, 0.2*k + 0.2) )
        arr[row, COLS_PER_FUNC] = 'i'
    return arr

def write_csv(arr):
    with open('func_data.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, arr.shape[0]):
            writer.writerow(arr[i, :])

def read_csv():
    #with open('func_data2.csv', 'r') as csvfile:
    #    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    return np.genfromtxt('func_data.csv', delimiter=' ')

def main():
    plot_functions()
    #generate_data()
    #arr = generate_data()
    #write_csv(arr)
    #print(read_csv().shape)

main()