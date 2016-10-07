from linear_regression import Regression
import math
import random
import numpy as np

def random_point():
    '''
    Returns a random 2-dimensional vector of floats between -1 and +1
    '''
    return [random.uniform(-1., 1.), random.uniform(-1., 1.)]

def target_f(x1, x2):
    '''
    Nonlinear target function f(x1, x2) = sign(x1**2 + x2**2 - 0.6)
    '''
    return math.copysign(1.0, (x1**2 + x2**2 - 0.6))

def generate_dataset(n):
    '''
    Takes n=total number of datapoints to generate
    Returns a length n list of tuples (x, y) with x a random vector and y=f(x)
    with a 10% random noise
    '''
    data = []
    for c in range(n):
        x = random_point()
        y = target_f(x[0], x[1])
        if random.uniform(0., 1.) <= 0.1:
            y = -y
        data.append((x, y))

    return data

def experiment1(n):
    '''
    Runs the experiment on n data points without transformations
    Returns the in-sample error
    '''
    r = Regression(2)
    total_Ein = 0.0
    for run in range(1000):
        data = generate_dataset(n)
        r.reset(data)
        r.solve()
        total_Ein += r.classification_error(r.data)

    avg_Ein = total_Ein / 1000

    return avg_Ein

def experiment2(n):
    '''
    Runs the experiment on n data points with the feature vector (1, x1, x2, x1x2, x1**2, x2**2)
    Returns the weights of the solution and the out-of-sample error
    '''
    r = Regression(5)
    total_weights = np.zeros(6)
    total_Eout = 0.0
    for run in range(1000):
        data = generate_dataset(n)
        for point in data:
            point[0].extend([point[0][0]*point[0][1], point[0][0]**2, point[0][1]**2])
        r.reset(data)
        r.solve()
        total_weights += r.weights[0]
        new_data = generate_dataset(n)
        for point in new_data:
            point[0].extend([point[0][0]*point[0][1], point[0][0]**2, point[0][1]**2])
        total_Eout += r.classification_error(new_data)

    avg_weights = total_weights / 1000
    avg_Eout = total_Eout / 1000

    return (avg_weights, avg_Eout)


#Experiments
'''
results = experiment1(1000)
print(results)
'''
results = experiment2(1000)
print(results)
