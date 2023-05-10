# +--------------------------------------------------------------------------+
# 
# Group Project #2 for Numerical Computing S23
#
# Team Members: William Franzen, Noah Harbor, Brandon Mitchell, Logan Reed
#
# Description:  Uses Newton's interpolation method and the cubic spline 
#               interpoloation methods to create and plot interploating 
#               functions.
#
# +--------------------------------------------------------------------------+

from CubicNatural import CubicNatural

# Must install using pip install <module name>
import numpy as np
import matplotlib.pyplot as plt

from math import cos, pi



# Netwon's Divided Difference
# Params:
#   x: list, x part of coordinates
#   y: list, y part of coordinates
# Return:
#   list, coefficients in the form [a0, a1, ... an]
# Description:
#   Uses Netwon's interpolation method to find the divided differences
def diff(x, y):

    if len(x) != len(y):
        raise ValueError("Inputs x and y must have the same length")
    
    n = len(y)
    coef = np.zeros([n, n])
    
    # The first column is y
    coef[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    
    # First row contains all coefficients we need
    return coef[0]



# Netwon's Polynomial Interpolation Function
# Params:
#   x: list, x part of coordinates
#   y: list, y part of coordinates
# Return:
#   function pointer, the interpolating function
# Description:
#   Uses Netwon's interpolation method to create an interpolating function
def newtonPolynomialFunc(xVals, yVals):

    coef = diff(xVals, yVals)
    
    func = f"lambda x: {coef[0]}"
    part = ""
    
    for i in range(1, len(coef)):
        func += f" + {coef[i]}"
        part += f" * (x - {xVals[i - 1]})"
        func += part
    
    return eval(func)



# Newton's Interpolation
# Params:
#   x: list, x part of coordinates
#   y: list, y part of coordinates
# Return:
#   float, the value estimated at pointToEval
# Description:
#   Uses Netwon's interpolation method to estimate the value at pointToEval
def newton(x, y, pointToEval):
    
    return newtonPolynomialFunc(x, y)(pointToEval)



# Read CSV Data
# Params:
#   fileName: string, the file to open and read
# Return:
#   twow lists of floats, x and y data from the file
# Description:
#   Opens the file, reads it, and converts the lines into x and y data
def readCSV(fileName):

    with open(fileName) as file:
        rawData = file.readlines()
    
    xData = []
    yData = []
    
    # Convert to floats, skip first line
    for line in rawData[1:]:
        x, y = [float(val) for val in line.split(',')]
        xData.append(x)
        yData.append(y)
    
    return xData, yData
    
    

# Write Spline Coefficients
# Params:
#   fileName: string, the name of the file to be created
# Description:
#   Writes the spline to a file filename in CSV format
def writeSplineCoef(spline, fileName):

    with open(fileName, 'w') as file:
        
        # Header of CSV
        file.write("Si,a,b,c,d\n")
        for x in range(len(spline[0])):
            file.write(f"S{x + 1},{spline[0][x]},{spline[1][x]},{spline[2][x]},{spline[3][x]}\n")



# Evaluate Using Cubic Spline
# Params:
#   pointToEval: float, the point to evaluate
#   x:, float list, the coordinate of the x points
#   coef: 2D float list, the return value of CubicNatural function
# Return:
#   float, the interpolated  value of pointToEval
# Description:
#   Locates the correct range of pointToEval and then evaluates it using the 
#   proper x value and coef terms
def evalCubicSpline(pointToEval, x, coef):
    if pointToEval < x[0] or pointToEval > x[-1]:
        raise ValueError("Point to evaluate out of range")
    
    # As long as points are in order, it is easy to find which range they are in
    for i in range(len(x) - 1):
        if pointToEval < x[i + 1]:
            break
            
    return coef[0][i] + coef[1][i] * (pointToEval - x[i]) + \
        coef[2][i] * (pointToEval - x[i]) ** 2 + coef[3][i] * (pointToEval - x[i]) ** 3



if __name__ == "__main__":
    # Question 2    
    # Our function that will be approximated
    q2Func = lambda x: 1 / (1 + 25 * x ** 2)
    
    # Xs and Ys of the various parts
    q2ax = [-1 + 0.5 * i for i in range(5)]
    q2ay = [q2Func(x) for x in q2ax]
    
    q2bx = [-1 + 0.2 * i for i in range(11)]
    q2by = [q2Func(x) for x in q2bx]
    
    q2cx = [-cos((2 * i + 1) / 22 * pi) for i in range(11)]
    q2cy = [q2Func(x) for x in q2cx]
        
    # Our interpolating functions, could simply eval each point using the 
    # newton function, but it is more efficient to get the function and then 
    # reuse it instead of needing to generate it each call
    q2aInterpFunc = newtonPolynomialFunc(q2ax, q2ay)
    q2bInterpFunc = newtonPolynomialFunc(q2bx, q2by)
    q2cInterpFunc = newtonPolynomialFunc(q2cx, q2cy)

    # Values used when ploting function and interpolating function
    q2Range = np.linspace(-1, 1, 100)
    
    plt.figure()
    plt.scatter(q2ax, q2ay)
    plt.plot(q2Range, [q2Func(x) for x in q2Range])
    plt.plot(q2Range, [q2aInterpFunc(x) for x in q2Range])
    plt.plot(q2Range, [q2Func(x) - q2aInterpFunc(x) for x in q2Range])
    plt.legend(["Points", "Original Func", "Interp Func", "Error"])
    plt.title("Q2, Part A")
    
    plt.figure()
    plt.scatter(q2bx, q2by)
    plt.plot(q2Range, [q2Func(x) for x in q2Range])
    plt.plot(q2Range, [q2bInterpFunc(x) for x in q2Range])
    plt.plot(q2Range, [q2Func(x) - q2bInterpFunc(x) for x in q2Range])
    plt.legend(["Points", "Original Func", "Interp Func", "Error"])
    plt.title("Q2, Part B")

    plt.figure()
    plt.scatter(q2cx, q2cy)
    plt.plot(q2Range, [q2Func(x) for x in q2Range])
    plt.plot(q2Range, [q2cInterpFunc(x) for x in q2Range])
    plt.plot(q2Range, [q2Func(x) - q2cInterpFunc(x) for x in q2Range])
    plt.legend(["Points", "Original Func", "Interp Func", "Error"])
    plt.title("Q2, Part C")
    
    
    
    # Question 3
    camelDataX, camelDataY = readCSV("camel data.csv")
    camelSpline = CubicNatural(camelDataX, camelDataY)
    camelRange = np.linspace(camelDataX[0], camelDataX[-1], 300)
    
    plt.figure()
    plt.scatter(camelDataX, camelDataY)
    plt.plot(camelRange, [evalCubicSpline(point, camelDataX, camelSpline) for point in camelRange])
    plt.legend(["Points", "Interp Func"])
    plt.title("Q3, Camel")
    
    
    
    # Question 4
    # Cat Stretching
    catDataX, catDataY = readCSV("cat data.csv")
    catSpline = CubicNatural(catDataX, catDataY)
    catRange = np.linspace(catDataX[0], catDataX[-1], 300)
    
    plt.figure()
    plt.scatter(catDataX, catDataY)
    plt.plot(catRange, [evalCubicSpline(point, catDataX, catSpline) for point in catRange])
    plt.legend(["Points", "Interp Func"])
    plt.title("Q4, Cat")
    
    # Cool Shadow on a motorcycle
    shadowDataX, shadowDataY = readCSV("shadow data.csv")
    shadowSpline = CubicNatural(shadowDataX, shadowDataY)
    shadowRange = np.linspace(shadowDataX[0], shadowDataX[-1], 300)
    
    plt.figure()
    plt.scatter(shadowDataX, shadowDataY)
    plt.plot(shadowRange, [evalCubicSpline(point, shadowDataX, shadowSpline) for point in shadowRange])
    plt.legend(["Points", "Interp Func"])
    plt.title("Q4, Shadow the Hedgehog")
    
    # An impressive tree
    treeDataX, treeDataY = readCSV("tree data.csv")
    treeSpline = CubicNatural(treeDataX, treeDataY)
    treeRange = np.linspace(treeDataX[0], treeDataX[-1], 300)
    
    plt.figure()
    plt.scatter(treeDataX, treeDataY)
    plt.plot(treeRange, [evalCubicSpline(point, treeDataX, treeSpline) for point in treeRange])
    plt.legend(["Points", "Interp Func"])
    plt.title("Q4, Tree")
    
    # Some imtimating felines
    jojoDataX, jojoDataY = readCSV("jojo cats data.csv")
    jojoSpline = CubicNatural(jojoDataX, jojoDataY)
    jojoRange = np.linspace(jojoDataX[0], jojoDataX[-1], 300)
    
    plt.figure()
    plt.scatter(jojoDataX, jojoDataY)
    plt.plot(jojoRange, [evalCubicSpline(point, jojoDataX, jojoSpline) for point in jojoRange])
    plt.legend(["Points", "Interp Func"])
    plt.title("Q4, JoJo Cats")
    
    # Save the splines to a file so their values can be viewed
    writeSplineCoef(camelSpline, "camel spline.csv")
    writeSplineCoef(catSpline, "cat spline.csv")
    writeSplineCoef(shadowSpline, "shadow spline.csv")
    writeSplineCoef(treeSpline, "tree spline.csv")
    writeSplineCoef(jojoSpline, "jojo cats spline.csv")
    
    # Shows all figures
    plt.show()