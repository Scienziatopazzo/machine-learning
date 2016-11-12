from linear_regression import Regression
from copy import deepcopy

def import_data(path):
    data = []
    f = open(path)
    for line in f:
        strpoint = line.split(" ")
        point = []
        for s in strpoint:
            if s != '':
                point.append(float(s))
        x = [point[0], point[1]]
        y = point[2]
        data.append((x, y))

    return data

def nonlinear_transform(data, k):
    newdata = []
    for point in data:
        newpoint = deepcopy(point)
        newdata.append(newpoint)
        for i in range(k):
            if i==2:
                newpoint[0].append(newpoint[0][0]**2)
            elif i==3:
                newpoint[0].append(newpoint[0][1]**2)
            elif i==4:
                newpoint[0].append(newpoint[0][0]*newpoint[0][1])
            elif i==5:
                newpoint[0].append(abs(newpoint[0][0]-newpoint[0][1]))
            elif i==6:
                newpoint[0].append(abs(newpoint[0][0]+newpoint[0][1]))
    return newdata

data_in = import_data('data/in.dta')
data_out = import_data('data/out.dta')
#data_training = data_in[0:25]
#data_validation = data_in[25:35]
data_training = data_in[25:35]
data_validation = data_in[0:25]

for k in range(3, 8):
    r = Regression(k)
    datat_t = nonlinear_transform(data_training, k)
    datat_v = nonlinear_transform(data_validation, k)
    datat_o = nonlinear_transform(data_out, k)

    r.reset(datat_t)
    r.solve()

    e_val = r.classification_error(datat_v)
    e_out = r.classification_error(datat_o)

    print(k)
    print(e_val)
    print(e_out)
