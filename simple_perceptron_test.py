from simple_perceptron import Perceptron
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
    '''
    Runs the experiment on n data points
    Returns the average number of iterations needed to converge and the average probability of the result being wrong
    '''
    p = Perceptron(2, [])
    total_iter = 0
    total_P = 0.0
    for run in range(1000):
        line = generate_line()
        data = generate_dataset(line, n)
        p.reset(data)
        p.train()
        total_iter += p.iterations
        new_data = generate_dataset(line, n*5)
        total_P += p.f_disagreement(new_data)
    
    avg_iter = total_iter / 1000
    avg_P = total_P / 1000
    
    return (avg_iter, avg_P)


#Experiments
results = experiment(10)
print(results)
results = experiment(100)
print(results)
