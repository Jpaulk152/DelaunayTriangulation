import matplotlib.pyplot as plt
from numpy import array

def connectpoints(x,y,p1,p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x1,x2],[y1,y2],'k-')

ax = plt.subplot()

x = []
y = []
test = []

#reads the points.txt for the points
data = open('points.txt','r').read()
lines = data.split('\n')

triangles = open('triangles.txt','r').read()
l = triangles.split('\n')

for line in lines:
	if len(line) > 1:
		_x, _y = line.split(',')
		x.append(_x)
		y.append(_y)

for i in l:
	if len(i) > 1:
		test = []
		_x, _y, _z = i.split(',')
		_x = int(_x)
		_y = int(_y)
		_z = int(_z)

		test.append(_x)
		test.append(_y)
		test.append(_z)

	connectpoints(x,y,test[0],test[1])
	connectpoints(x,y,test[0],test[2])
	connectpoints(x,y,test[2],test[1])
		
#drawing the points
for i in range(0, len(x), 2):
   plt.scatter(x, y)


plt.gca().invert_yaxis()
ax.xaxis.tick_top()
plt.show()

type()