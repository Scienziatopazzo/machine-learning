from simple_perceptron import Perceptron
from linear_regression import Regression
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


def experiment1(n):
    '''
    Runs the experiment on n data points
    Returns the in-sample error and the out-of-sample error
    '''
    r = Regression(2)
    total_Ein = 0.0
    total_Eout = 0.0
    for run in range(1000):
        line = generate_line()
        data = generate_dataset(line, n)
        r.reset(data)
        r.solve()
        total_Ein += r.classification_error(r.data)
        new_data = generate_dataset(line, n*10)
        total_Eout += r.classification_error(new_data)

    avg_Ein = total_Ein / 1000
    avg_Eout = total_Eout / 1000

    return (avg_Ein, avg_Eout)

def experiment2(n):
    '''
    Runs the experiment on n data points
    Returns the number of iterations needed for the PLA to converge after being fed weights computed by linear regression
    '''
    r = Regression(2)
    p = Perceptron(2)
    total_iterations = 0
    for run in range(1000):
        line = generate_line()
        data = generate_dataset(line, n)
        r.reset(data)
        r.solve()
        p.reset(data, r.weights[0])
        p.train()
        total_iterations += p.iterations

    avg_iterations = total_iterations / 1000

    return avg_iterations


#Experiments
results = experiment1(100)
print(results)

results = experiment2(10)
print(results)
