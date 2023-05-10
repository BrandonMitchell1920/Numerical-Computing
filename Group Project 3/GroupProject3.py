# +---------------------------------------------------------------------------+
#
# Group Project #3 Methods for First-Order ODEs
# Group Members: William Franzen, Noah Harbor, Brandon Mitchell, Logan Reed
# Description:  Uses iterative formulas to estimate answers to initial value problems.
#               Implement Euler's Method, Midpoint Method, and Runge-Kutta Method.
#
# +---------------------------------------------------------------------------+

import matplotlib.pyplot as plt
import numpy as np
import math



#
#   Euler's Method
#   Description:
#       Implements Euler's method to approximate initial value problems.
#   Parameters:
#       func: A given function f(x, y) to evaluate.
#       x0: Initial x value.
#       y0: Initial y value.
#       start: The start point of the interval.
#       end: The ending point of the interval.
#       h: The size of each sub-interval.
#       exact: The exact function f(x).
#   Output:
#       xVals: An array that contains the estimated x values.
#       yVals: An array that contains the estimated y values.
#       error: An array that contains the difference between the estimated and exact values.
#
def eulers(func, x0, y0, start, end, h, exact):
    #Initializes the arrays with the given initial parameters.
    xVals = [x0]
    yVals = [y0]
    error = [exact(x0) - y0]
    
    #Iterates from the start to the end with step size h.
    for i in np.arange(start, end, h):
        #Fetches the rightmost values in the arrays, and stores them into x and y.
        x = xVals[-1]
        y = yVals[-1]
        
        #Euler's formula.
        try:
            yn = y + h * func(x, y)
        except ZeroDivisionError:
            yn = y
        
        #Appends the new values to the arrays.
        xVals.append(x + h)
        yVals.append(yn)
        error.append(abs(exact(x + h) - yn))

    return xVals, yVals, error



#
#   Improved Euler's Method
#   Description:
#       Implements the Improved Euler's method to approximate initial value problems.
#   Parameters:
#       func: A given function f(x, y) to evaluate.
#       x0: Initial x value.
#       y0: Initial y value.
#       start: The start point of the interval.
#       end: The ending point of the interval.
#       h: The size of each sub-interval.
#       exact: The exact function f(x).
#   Output:
#       xVals: An array that contains the estimated x values.
#       yVals: An array that contains the estimated y values.
#       error: An array that contains the difference between the estimated and exact values.
#
def improvedEulers(func, x0, y0, start, end, h, exact):
    #Initializes the arrays with the given initial parameters.
    xVals = [x0]
    yVals = [y0]
    error = [exact(x0) - y0]
    
    #Iterates from the start to the end with step size h.
    for i in np.arange(start, end, h):
        #Fetches the rightmost values in the arrays, and stores them into x and y.
        x = xVals[-1]
        y = yVals[-1]
        
        #Improved Euler's formula
        yn = y + h / 2 * (func(x, y) + func(x + h, y + h * func(x, y)))
        
        #Appends the new values to the arrays.        
        xVals.append(x + h)
        yVals.append(yn)
        error.append(abs(exact(x + h) - yn))
        
    return xVals, yVals, error



#
#   Midpoint Method
#   Description:
#       Implements the Midpoint method to approximate initial value problems.
#   Parameters:
#       func: A given function f(x, y) to evaluate.
#       x0: Initial x value.
#       y0: Initial y value.
#       start: The start point of the interval.
#       end: The ending point of the interval.
#       h: The size of each sub-interval.
#       exact: The exact function f(x).
#   Output:
#       xVals: An array that contains the estimated x values.
#       yVals: An array that contains the estimated y values.
#       error: An array that contains the difference between the estimated and exact values.
#
def midpoint(func, x0, y0, start, end, h, exact):
    #Initializes the arrays with the given initial parameters.
    xVals = [x0]
    yVals = [y0]
    error = [exact(x0) - y0]
    
    #Iterates from the start to the end with step size h.
    for i in np.arange(start, end, h):
        #Fetches the rightmost values in the arrays, and stores them into x and y.
        x = xVals[-1]
        y = yVals[-1]

        #The Midpoint formula.        
        yn = y + h * func(x + h / 2, y + h / 2 * func(x, y))
        
        #Appends the new values to the arrays.
        xVals.append(x + h)
        yVals.append(yn)
        error.append(abs(exact(x + h) - yn))

    return xVals, yVals, error


#
#   Runge-Kutta Method (4th Order)
#   Description:
#       Implements the Runge-Kutta method (4th Order) to approximate initial value problems.
#   Parameters:
#       func: A given function f(x, y) to evaluate.
#       x0: Initial x value.
#       y0: Initial y value.
#       start: The start point of the interval.
#       end: The ending point of the interval.
#       h: The size of each sub-interval.
#       exact: The exact function f(x).
#   Output:
#       xVals: An array that contains the estimated x values.
#       yVals: An array that contains the estimated y values.
#       error: An array that contains the difference between the estimated and exact values.
#
def rungeKutta(func, x0, y0, start, end, h, exact):
    #Initializes the arrays with the given initial parameters.
    xVals = [x0]
    yVals = [y0]
    error = [exact(x0) - y0]
    
    #Iterates from the start to the end with step size h.
    for i in np.arange(start, end, h):
        #Fetches the rightmost values in the arrays, and stores them into x and y.
        x = xVals[-1]
        y = yVals[-1]
        
        #Handles some exceptions (asymptotes and overflows)
        try:
            #Sets the four k values, as given by the Runge-Kutta formula.
            k1 = h * func(x, y)
            k2 = h * func(x + (h / 2), y + (1 / 2) * k1)
            k3 = h * func(x + (h / 2), y + (1 / 2) * k2)
            k4 = h * func(x + h, y + k3)
            
            #The iterative formula.
            yn = y + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            #Appends the new values to the arrays.
            xVals.append(x + h)
            yVals.append(yn)
            error.append(abs(exact(x + h) - yn))
            
        # Question 3 has an asymptote at 1
        except ZeroDivisionError:
            error.append(float('inf') - yn)
            
        # Possibility of an overflow error with question 3
        except OverflowError:
            break

    return xVals, yVals, error
    
    
    
#
#   Taylor's Method (2nd Order)
#   Description:
#       Implements Taylor's method (2nd Order) to approximate initial value problems.
#   Parameters:
#       func: A given function f(x, y) to evaluate.
#       funcX: The partial derivative of func with respect to x.
#       funcY: The partial derivative of func with respect to y.
#       x0: Initial x value.
#       y0: Initial y value.
#       start: The start point of the interval.
#       end: The ending point of the interval.
#       h: The size of each sub-interval.
#       exact: The exact function f(x).
#   Output:
#       xVals: An array that contains the estimated x values.
#       yVals: An array that contains the estimated y values.
#       error: An array that contains the difference between the estimated and exact values.
#
def taylors(func, funcX, funcY, x0, y0, start, end, h, exact):
    #Initializes the arrays with the given initial parameters.
    xVals = [x0]
    yVals = [y0]
    error = [exact(x0) - y0]
    
    #Iterates from the start to the end with step size h.
    for i in np.arange(start, end, h):
        #Fetches the rightmost values in the arrays, and stores them into x and y.
        x = xVals[-1]
        y = yVals[-1]
        
        #The formula for Taylor's Method Order 2.
        yn = y + func(x, y) * h + (h ** 2 / 2) * (funcX(x, y) + funcY(x, y) * func(x, y))
        
        #Appends the new values to the arrays.
        xVals.append(x + h)
        yVals.append(yn)
        error.append(abs(exact(x + h) - yn))

    return xVals, yVals, error





if __name__ == "__main__":

    # Question 1 --------------------------------------------------------------

    q1ivp = lambda x, y: y + 2 * x - x ** 2
    q1exact = lambda x: x ** 2 + math.e ** x
    
    q1x0 = 0
    q1y0 = 1
    q1start = 0
    q1end = 2
    q1h = 0.05
    
    q1xExact = np.linspace(q1start, q1end, 100)
    q1yExact = [q1exact(x) for x in q1xExact]
    
    
    
    #Euler's
    q1xEuler, q1yEuler, q1EulerError = eulers(q1ivp, q1x0, q1y0, q1start, q1end, q1h, q1exact)
    
    plt.figure()
    plt.plot(q1xEuler, q1yEuler)
    plt.plot(q1xExact, q1yExact)
    plt.plot(q1xEuler, q1EulerError)
    plt.legend(["Approximation", "Exact", "Error"])
    plt.title("Q1 Euler's Method, h = 0.05")
    
    print("Q1 Euler's Error, h = 0.05")
    for i in range(len(q1xEuler)):
        print(f"{q1xEuler[i]}: {q1EulerError[i]}")
        
        
    q1xEuler, q1yEuler, q1EulerError = eulers(q1ivp, q1x0, q1y0, q1start, q1end, .001, q1exact)
    
    plt.figure()
    plt.plot(q1xEuler, q1yEuler)
    plt.plot(q1xExact, q1yExact)
    plt.plot(q1xEuler, q1EulerError)
    plt.legend(["Approximation", "Exact", "Error"])
    plt.title("Q1 Euler's Method, h = 0.001")
    
    print("Q1 Euler's Error, h = 0.001")
    for i in range(len(q1xEuler)):
        print(f"{q1xEuler[i]}: {q1EulerError[i]}")
    
    
    
    #Midpoint
    q1xMidpoint, q1yMidpoint, q1MidpointError = midpoint(q1ivp, q1x0, q1y0, q1start, q1end, q1h, q1exact)
    
    plt.figure()
    plt.plot(q1xMidpoint, q1yMidpoint)
    plt.plot(q1xExact, q1yExact)
    plt.plot(q1xMidpoint, q1MidpointError)
    plt.legend(["Approximation", "Exact", "Error"])
    plt.title("Q1 Midpoint Method")
    
    print("\nQ1 Midpoint's Error")
    for i in range(len(q1xMidpoint)):
        print(f"{q1xMidpoint[i]}: {q1MidpointError[i]}")
    
    
    #Runge-Kutta
    q1xRunge, q1yRunge, q1RungeError = rungeKutta(q1ivp, q1x0, q1y0, q1start, q1end, q1h, q1exact)
    
    plt.figure()
    plt.plot(q1xRunge, q1yRunge)
    plt.plot(q1xExact, q1yExact)
    plt.plot(q1xMidpoint, q1RungeError)
    plt.legend(["Approximation", "Exact", "Error"])
    plt.title("Q1 Runge-Kutta Method")
    
    print("\nQ1 Runge-Kutta's Error")
    for i in range(len(q1xRunge)):
        print(f"{q1xRunge[i]}: {q1RungeError[i]}")
    
    
    
    # Question 2 --------------------------------------------------------------

    q2ivp = lambda x, y: -y + x + 1
    q2ivpDerX = lambda x, y: 1
    q2ivpDerY = lambda x, y: -1
    q2exact = lambda x: x + 1 / math.e ** x
    
    q2x0 = -2
    q2y0 = -2 + math.e ** 2
    q2start = -2
    q2end = 3
    
    q2xExact = np.linspace(q2start, q2end, 100)
    q2yExact = [q2exact(x) for x in q2xExact]
    
    
    
    # h = 0.2
    q2xTaylor1, q2yTaylor1, q2TalyorError1 = taylors(q2ivp, q2ivpDerX, q2ivpDerY, q2x0, q2y0, q2start, q2end, 0.2, q2exact)
    
    plt.figure()
    plt.plot(q2xTaylor1, q2yTaylor1)
    plt.plot(q2xExact, q2yExact)
    plt.plot(q2xTaylor1, q2TalyorError1)
    plt.legend(["Approximation", "Exact", "Error"])
    plt.title("Q2 Taylor's Method, h = 0.2")
    
    print("\nQ2 Taylors's Error, h = 0.2")
    for i in range(len(q2xTaylor1)):
        print(f"{q2xTaylor1[i]}: {q2TalyorError1[i]}")
    
    
    
    # h = 0.1
    q2xTaylor2, q2yTaylor2, q2TalyorError2 = taylors(q2ivp, q2ivpDerX, q2ivpDerY, q2x0, q2y0, q2start, q2end, 0.1, q2exact)
    
    plt.figure()
    plt.plot(q2xTaylor2, q2yTaylor2)
    plt.plot(q2xExact, q2yExact)
    plt.plot(q2xTaylor2, q2TalyorError2)
    plt.legend(["Approximation", "Exact", "Error"])
    plt.title("Q2 Taylor's Method, h = 0.1")
    
    print("\nQ2 Taylors's Error, h = 0.1")
    for i in range(len(q2xTaylor2)):
        print(f"{q2xTaylor2[i]}: {q2TalyorError2[i]}")
    
    
    
    # h = 0.05    
    q2xTaylor3, q2yTaylor3, q2TalyorError3 = taylors(q2ivp, q2ivpDerX, q2ivpDerY, q2x0, q2y0, q2start, q2end, 0.05, q2exact)
    
    plt.figure()
    plt.plot(q2xTaylor3, q2yTaylor3)
    plt.plot(q2xExact, q2yExact)
    plt.plot(q2xTaylor3, q2TalyorError3)
    plt.legend(["Approximation", "Exact", "Error"])
    plt.title("Q2 Taylor's Method, h = 0.05")
    
    print("\nQ2 Taylors's Error, h = 0.05")
    for i in range(len(q2xTaylor3)):
        print(f"{q2xTaylor3[i]}: {q2TalyorError3[i]}")
    
    
    
    # Question 3 --------------------------------------------------------------
    
    q3ivp = lambda x, y: y ** 2
    q3exact = lambda x: 1 / (1 - x)
    
    q3x0 = 0
    q3y0 = 1
    q3start = 0
    q3end = 2
    q3h = 0.25
    
    q3xExact = np.linspace(q3start, q3end, 100)
    q3yExact = [q3exact(x) for x in q3xExact]
    
    q3xRunge, q3yRunge, q3yRungeError = rungeKutta(q3ivp, q3x0, q3y0, q3start, q3end, q3h, q3exact)
    
    plt.figure()
    plt.plot(q3xRunge, q3yRunge)
    plt.plot(q3xExact, q3yExact)
    plt.plot(q3xRunge, q3yRungeError)
    plt.legend(["Approximation", "Exact", "Error"])
    plt.title("Q3 Runge-Kutta Method")
    
    plt.figure()
    plt.plot(q3xRunge, q3yRunge)
    plt.plot(q3xExact, q3yExact)
    plt.plot(q3xRunge, q3yRungeError)
    plt.legend(["Approximation", "Exact", "Error"])
    plt.title("Q3 Runge-Kutta Method, Adjusted Scale")
    plt.ylim(-5, 50)
    
    
    
    # Extras ------------------------------------------------------------------
    
    # Extra 1:
    print ("\nExtra IVP 1:")
    ext1ivp = lambda x, y: 4 * math.sin(x) - 3 * y
    ext1exact = lambda x: (2/5) * (math.e ** (-3*x) + 3 * math.sin(x) - math.cos(x))

    ext1x0 = 0
    ext1y0 = 0
    ext1start = 0
    ext1end = 4
    ext1h = 0.25

    ext1xExact = np.linspace(ext1start, ext1end, 100)
    ext1yExact = [ext1exact(x) for x in ext1xExact]

    ext1Eulerx, ext1Eulery, ext1EulerError = eulers(ext1ivp, ext1x0, ext1y0, ext1start, ext1end, ext1h, ext1exact)

    plt.figure()
    plt.plot(ext1Eulerx, ext1Eulery)
    plt.plot(ext1xExact, ext1yExact)
    plt.plot(ext1Eulerx, ext1EulerError)
    plt.legend(["Approximation", "Exact", "Error"])
    plt.title("Extra Question 1 (Euler's With Trigonometry)")
    plt.ylim(-1, 4)



    # Extra 2:
    print ("\nExtra IVP 1:")
    ext2ivp = lambda x, y: (y - math.log(x)) / x
    ext2exact = lambda x: math.log(x) + 1

    ext2x0 = 1
    ext2y0 = 1
    ext2start = 1
    ext2end = 5
    ext2h = 0.25

    ext2xExact = np.linspace(ext2start, ext2end, 100)
    ext2yExact = [ext2exact(x) for x in ext2xExact]

    ext2Eulerx, ext2Eulery, ext2EulerError = eulers(ext2ivp, ext2x0, ext2y0, ext2start, ext2end, ext2h, ext2exact)

    plt.figure()
    plt.plot(ext2Eulerx, ext2Eulery)
    plt.plot(ext2xExact, ext2yExact)
    plt.plot(ext2Eulerx, ext2EulerError)
    plt.legend(["Approximation", "Exact", "Error"])
    plt.title("Extra Question 2 (Euler's With Logarithm Stuff)")
    plt.ylim(-1, 4)
    
    
    
    # Extra 3 HW 6 Question 2 
    
    hw6Q2ivp = lambda x, y: 1 + y / x
    hw6Q2Exact = lambda x: x * math.log(x) + 2 * x
    
    hw6Q2x0 = 1
    hw6Q2y0 = 2
    hw6Q2start = 1
    hw6Q2end = 2
    
    hw6Q2h1 = 0.25
    hw6Q2h2 = 0.5
    
    hw6Q2xExact = np.linspace(hw6Q2start, hw6Q2end, 100)
    hw6Q2yExact = [hw6Q2Exact(x) for x in hw6Q2xExact]
    
    hw6Q2xMidpoint1, hw6Q2yMidpoint1, hw6Q2xMidpointError1 = midpoint(hw6Q2ivp, hw6Q2x0, hw6Q2y0, hw6Q2start, hw6Q2end, hw6Q2h1, hw6Q2Exact)
    hw6Q2xMidpoint2, hw6Q2yMidpoint2, hw6Q2xMidpointError2 = midpoint(hw6Q2ivp, hw6Q2x0, hw6Q2y0, hw6Q2start, hw6Q2end, hw6Q2h2, hw6Q2Exact)
    
    plt.figure()
    plt.plot(hw6Q2xMidpoint1, hw6Q2yMidpoint1)
    plt.plot(hw6Q2xExact, hw6Q2yExact)
    plt.plot(hw6Q2xMidpoint1, hw6Q2xMidpointError1)
    plt.legend(["Approximation", "Exact", "Error"])
    plt.title("Extra Question 3, HW 6 Q2, Midpoint Method, h = 0.25")
    
    print("\nHW 6 Q2, Midpoint Error, h = 0.25")
    for i in range(len(hw6Q2xMidpoint1)):
        print(f"{hw6Q2xMidpoint1[i]}: {hw6Q2xMidpointError1[i]}")
    
    plt.figure()
    plt.plot(hw6Q2xMidpoint2, hw6Q2yMidpoint2)
    plt.plot(hw6Q2xExact, hw6Q2yExact)
    plt.plot(hw6Q2xMidpoint2, hw6Q2xMidpointError2)
    plt.legend(["Approximation", "Exact", "Error"])
    plt.title("Extra Question 3, HW 6 Q2, Midpoint Method, h = 0.5")
    
    print("\nHW 6 Q2, Midpoint Error, h = 0.5")
    for i in range(len(hw6Q2xMidpoint2)):
        print(f"{hw6Q2xMidpoint2[i]}: {hw6Q2xMidpointError2[i]}")
    
    
    
    # Extra 4, HW 6 Question 3
    hw6Q3ivp = lambda x, y: x * math.e ** (3 * x) - 2 * y
    hw6Q3Exact = lambda x: 1 / 5 * x * math.e ** (3 * x) - 1 / 25 * math.e ** (3 * x) + 1 / 25 * math.e ** (-2 * x)
        
    hw6Q3x0 = 0
    hw6Q3y0 = 0
    hw6Q3start = 0
    hw6Q3end = 2
    hw6Q3h = 0.25
    
    hw6Q3xExact = np.linspace(hw6Q3start, hw6Q3end, 100)
    hw6Q3yExact = [hw6Q3Exact(x) for x in hw6Q3xExact]

    hw6Q3xEuler, hw6Q3yEuler, hw6Q3EulerError = eulers(hw6Q3ivp, hw6Q3x0, hw6Q3y0, hw6Q3start, hw6Q3end, hw6Q3h, hw6Q3Exact)
    hw6Q3xImprovedEuler, hw6Q3yImprovedEuler, hw6Q3ImprovedEulerError = improvedEulers(hw6Q3ivp, hw6Q3x0, hw6Q3y0, hw6Q3start, hw6Q3end, hw6Q3h, hw6Q3Exact)

    plt.figure()
    plt.plot(hw6Q3xEuler, hw6Q3yEuler)
    plt.plot(hw6Q3xExact, hw6Q3yExact)
    plt.plot(hw6Q3xEuler, hw6Q3EulerError)
    plt.legend(["Approximation", "Exact", "Error"])
    plt.title("Extra Question 4, HW 6 Q3, Euler's Method")
    
    plt.figure()
    plt.plot(hw6Q3xImprovedEuler, hw6Q3yImprovedEuler)
    plt.plot(hw6Q3xExact, hw6Q3yExact)
    plt.plot(hw6Q3xImprovedEuler, hw6Q3ImprovedEulerError)
    plt.legend(["Approximation", "Exact", "Error"])
    plt.title("Extra Question 4, HW 6 Q3, Improved Euler's Method")
    
    print("\nHomework 6, Question 3")
    print("xi   Euler yi            Error               Improved Euler yi   Error                Exact")
    for i in range(len(hw6Q3xEuler)):
        print(f"{hw6Q3xEuler[i]:<4} {hw6Q3yEuler[i]:<19} {hw6Q3EulerError[i]:<19} " +
        f"{hw6Q3yImprovedEuler[i]:<19} {hw6Q3ImprovedEulerError[i]:<20} {hw6Q3Exact(hw6Q3xEuler[i]):<19}")
    
    
    
    # Extra 5
    ext5ivp = lambda x, y: 2 * x / (y + x ** (2 * y))
    ext5exact = lambda x: 2 * x ** 5 / (-2 * x ** 4 + 1)
    
    ext5x0 = 0
    ext5y0 = -2
    ext5start = 0
    ext5end = 3
    ext5h = 0.05

    ext5xExact = np.linspace(ext5start, ext5end, 100)
    ext5yExact = [ext5exact(x) for x in ext5xExact]

    ext5xEuler, ext5yEuler, ext5EulerError = eulers(ext5ivp, ext5x0, ext5y0, ext5start, ext5end, ext5h, ext5exact)
    
    plt.figure()
    plt.plot(ext5xEuler, ext5yEuler)
    plt.plot(ext5xExact, ext5yExact)
    plt.plot(ext5xEuler, ext5EulerError)
    plt.legend(["Approximation", "Exact", "Error"])
    plt.title("Extra Question 5, IVP with Asymptote")
    
    print("\nQ1 Euler's Error")
    for i in range(len(ext5xEuler)):
        print(f"{ext5xEuler[i]}: {ext5EulerError[i]}")



    # Extra 6, comparison
    ext6ivp = lambda x, y: -y + x + 1
    ext6ivpDerX = lambda x, y: 1
    ext6ivpDerY = lambda x, y: -1
    ext6exact = lambda x: x + 1 / math.e ** x
    
    ext6x0 = -2
    ext6y0 = -2 + math.e ** 2
    ext6start = -2
    ext6end = 3
    ext6h = 0.25
    
    ext6xExact = np.linspace(ext6start, ext6end, 100)
    ext6yExact = [ext6exact(x) for x in ext6xExact]
    
    
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle("Q2 Comparison, h = 0.25")
    fig.legend(["Approximation", "Exact", "Error"])
    fig.tight_layout()
    
    ext6x, ext6y, ext6Error = eulers(ext6ivp, ext6x0, ext6y0, ext6start, ext6end, ext6h, ext6exact)
    
    ax1.plot(ext6x, ext6y)
    ax1.plot(ext6xExact, ext6yExact)
    ax1.plot(ext6x, ext6Error)
    ax1.set_title("Euler's")
    
    print("\nQ2 Euler's Error")
    for i in range(len(ext6x)):
        print(f"{ext6x[i]}: {ext6Error[i]}")
    
    
    
    ext6x, ext6y, ext6Error = midpoint(ext6ivp, ext6x0, ext6y0, ext6start, ext6end, ext6h, ext6exact)
    
    ax2.plot(ext6x, ext6y)
    ax2.plot(ext6xExact, ext6yExact)
    ax2.plot(ext6x, ext6Error)
    ax2.set_title("Midpoint")
    
    print("\nQ2 Midpoint Error")
    for i in range(len(ext6x)):
        print(f"{ext6x[i]}: {ext6Error[i]}")
    
    
    
    ext6x, ext6y, ext6Error = rungeKutta(ext6ivp, ext6x0, ext6y0, ext6start, ext6end, ext6h, ext6exact)
    
    ax3.plot(ext6x, ext6y)
    ax3.plot(ext6xExact, ext6yExact)
    ax3.plot(ext6x, ext6Error)
    ax3.set_title("Runge-Kutta")
    
    print("\nQ2 Runge-Kutta's Error")
    for i in range(len(ext6x)):
        print(f"{ext6x[i]}: {ext6Error[i]}")
    
    
    
    ext6x, ext6y, ext6Error = taylors(ext6ivp, ext6ivpDerX, ext6ivpDerY, ext6x0, ext6y0, ext6start, ext6end, ext6h, ext6exact)
    
    ax4.plot(ext6x, ext6y)
    ax4.plot(ext6xExact, ext6yExact)
    ax4.plot(ext6x, ext6Error)
    ax4.set_title("Taylor's")
    
    print("\nQ2 Taylors's Error")
    for i in range(len(ext6x)):
        print(f"{ext6x[i]}: {ext6Error[i]}")



    plt.show()