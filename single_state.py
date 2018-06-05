import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 2 + np.sin(2 * np.pi * x) + -((x - 1) ** 2) / 2

def randomPoint():
    x = np.random.randint(0, 400) / 100
    plt.plot(x, f(x), 'ro')

def hill_climbing():
    s = np.random.randint(0, 200) / 100
    plt.plot(s, f(s), 'rx')
    shapes = ["bx", "yx", "gx", "rx"]
    for i in range(0, 20):
        r = s + np.random.randint(-10, 10)/100
        if f(r) > f(s):
            s = r
        plt.plot(r, f(r), shapes[int(i / 5)])
    plt.plot(s, f(s), "ro")

def steepest_hill_climbing(ax):
    #randWalk = np.random.rand(1,10) - 0.5
    #s = randWalk[np.argmax(f(randWalk))]
    searchrange = 5
    s = np.random.randint(50, 75) / 100
    ax.plot(s, f(s), 'rx')
    shapes = ["bx", "yx", "gx", "ro"]
    for i in range(0, 7):
        r = s + np.random.randint(-searchrange, searchrange) / 100
        for y in range(0, 10):
            w = s + np.random.randint(-searchrange, searchrange) / 100
            if f(w) > f(r):
                r = w
        if f(r) > f(s):
            s = r
        ax.plot(r, f(r), shapes[int(i/2)])
    #plt.plot(s, f(s), "ro")












def main():
    s = np.arange(0.0, 2.0, 0.02)
    fig, (ax1, ax2) = plt.subplots(2, sharey=True)
    ax1.plot(s, f(s))
    ax2.plot(s, f(s))

    steepest_hill_climbing(ax2)
    ax1.plot(0.1, f(0.1), "ro")
    ax1.plot(0.45, f(0.45), "ro")
    ax1.plot(1.15, f(1.15), "ro")
    ax1.plot(1.4, f(1.4), "ro")


    #plt.xlabel('range')
    #plt.ylabel('fitness')
    plt.grid(False)
    plt.savefig("test.png")
    plt.show()

#randWalk = np.random.rand(1,10) - 0.5
#index = np.argmax(f(randWalk))
#s = randWalk[0,index]
main()
