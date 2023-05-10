import numpy as np

# The code is written by Drs.Yaning Liu and Giray Ökten
# Input: Points(x(1),y(1)),(x(2),y(3)),...spline must interpolate.
# Output: coefficents of each piece of the piecewise cubic spline
# Note: S_i(x) = ai+bi(x - x_i-1)+ci(x - x_i-1)^2+di(x - x_i-1),
#                          for x in [x_i-1, x_i]  i = 1,2,... n.
#       left end point x_i-1 in each subinterval is used.
#
# Modified by Brandon Mitchell
# Changed x.size to len(x) so normal lists are supported as input

def CubicNatural(x, y):
    m = len(x) # m is the number of data points
    n = m - 1
    a = np.zeros(m)
    b = np.zeros(n)
    c = np.zeros(m)
    d = np.zeros(n)
    for i in range(m):
        a[i] = y[i]
    h = np.zeros(n)
    for i in range(n):
        h[i] = x[i+1] - x[i]
    u = np.zeros(n)
    u[0] = 0
    for i in range(1, n):
        u[i] = 3*(a[i+1]-a[i])/h[i]-3*(a[i]-a[i-1])/h[i-1]
    s = np.zeros(m)
    z = np.zeros(m)
    t = np.zeros(n)
    s[0] = 1
    z[0] = 0
    t[0] = 0
    for i in range(1, n):
        s[i] = 2*(x[i+1]-x[i-1])-h[i-1]*t[i-1]
        t[i] = h[i]/s[i]
        z[i]=(u[i]-h[i-1]*z[i-1])/s[i]
    s[m-1] = 1
    z[m-1] = 0
    c[m-1] = 0
    for i in np.flip(np.arange(n)):
        c[i] = z[i]-t[i]*c[i+1]
        b[i] = (a[i+1]-a[i])/h[i]-h[i]*(c[i+1]+2*c[i])/3
        d[i] = (c[i+1]-c[i])/(3*h[i])
    a = a[0:m-1]
    c = c[0:m-1]
    return a, b, c, d