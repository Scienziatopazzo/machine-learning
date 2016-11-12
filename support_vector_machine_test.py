from simple_perceptron import Perceptron
from support_vector_machine import SVM
import random

def random_point():
    '''
    Returns a random 2-dimensional vector of floats between -1 and +1
    '''
    return [random.uniform(-1., 1.), random.uniform(-1., 1.)]

def generate_line():
    '''
    Randomly generates a line from 2 random points in [-1,1]x[-1,1]
    and returns the tuple (m, q, inv) for y = mx + q  with inv a boolean which decides what side of the line maps to +1
    (ignores vertical lines)
    '''
    while (True):
        pointA = random_point()
        pointB = random_point()
        if ((pointB[0] - pointA[0]) != 0):
            break

    m = (pointB[1] - pointA[1]) / (pointB[0] - pointA[0])
    q = pointA[1] - m*pointA[0]
    inv = bool(random.getrandbits(1))
    return (m, q, inv)

def compute_f(line, point):
    '''
    Takes an (m, q, inv) tuple representing a line and takes a point, computes f(x)
    Returns 1 if the point is over the line, returns -1 if it's under it
    '''
    if (point[1] >= (line[0]*point[0] + line[1])):
        if (line[2]):
            return 1
        else:
            return -1
    else:
        if (line[2]):
            return -1
        else:
            return 1


def generate_dataset(line, n):
    '''
    Takes an (m, q, inv) tuple representing a line and n=total number of datapoints to generate
    Returns a length n list of tuples (x, y) with x a random vector and y=f(x)
    '''
    data = []
    for c in range(n):
        x = random_point()
        y = compute_f(line, x)
        data.append((x, y))

    return data

def experiment(n):
    s = SVM(2)
    p = Perceptron(2)
    tot_better = 0
    tot_sv = 0
    for run in range(1000):
        line = generate_line()
        data = []
        alleq = True
        while alleq:
            data = generate_dataset(line, n)
            prevy = data[0][1]
            for i in range(1, len(data)):
                if data[i][1]==prevy:
                    alleq = False
                    break
                prevy = data[i][1]
        s.reset(data)
        p.reset(data)
        s.solve()
        tot_sv += len(s.suppvectors)
        p.train()
        new_data = generate_dataset(line, n*5)
        if p.f_disagreement(new_data) > s.classification_error(new_data):
            tot_better += 1
    perc_better = tot_better/1000
    avg_sv = tot_sv / 1000
    return (perc_better, avg_sv)

#print(experiment(10))
print(experiment(100))
