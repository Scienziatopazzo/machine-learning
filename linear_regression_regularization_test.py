from linear_regression import Regression

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

def nonlinear_transform(data):
	for point in data:
		point[0].extend([point[0][0]**2, point[0][1]**2, point[0][0]*point[0][1], abs(point[0][0]-point[0][1]), abs(point[0][0]+point[0][1])])

def lam(k):
	return 10**k

r = Regression(7)
data_in = import_data('data/in.dta')
data_out = import_data('data/out.dta')
nonlinear_transform(data_in)
nonlinear_transform(data_out)

r.reset(data_in)
r.solve()

e_in = r.classification_error(data_in)
e_out = r.classification_error(data_out)

print(e_in)
print(e_out)

for k in range(-10, 10):

	r.reset(data_in, lam(k))
	r.solve()

	e_in = r.classification_error(data_in)
	e_out = r.classification_error(data_out)

	print(k)
	print(e_in)
	print(e_out)
