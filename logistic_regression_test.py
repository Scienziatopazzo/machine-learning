from logistic_regression import LogRegression
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

def compute_y(line, point):
    '''
    Takes an (m, q, inv) tuple representing a line and takes a point, computes y
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
        y = compute_y(line, x)
        data.append((x, y))

    return data

def experiment(n):
    l = LogRegression(2, 0.01)
    total_Eout = 0.0
    total_epochs = 0
    for run in range(100):
        line = generate_line()
        data = generate_dataset(line, n)
        l.reset(data)
        l.gradient_descent(0.01)
        total_epochs += l.epochs
        new_data = generate_dataset(line, n*10)
        total_Eout += l.cross_entropy_error(new_data)

    avg_Eout = total_Eout / 100
    avg_epochs = total_epochs / 100

    return (avg_Eout, avg_epochs)

print(experiment(100))
