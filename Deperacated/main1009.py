# Scratch to find the equalibrium of sinc function
from numpy import *
from matplotlib.pyplot import *
import cfft

x = linspace(-3, 3, 1000)
y = sin(pi*x)/(pi*x) if x[0] != 0 else 1
y[x == 0] = 1
y1 = 1/(pi*x)

figure()
plot(x, y)
plot(x, 1/(pi/2)*ones_like(x))
plot(x, y1)
ylim([-1,1])

show()