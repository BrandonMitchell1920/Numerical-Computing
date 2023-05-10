# +--------------------------------------------------------------------------+
# 
# Group Project #1 for Numerical Computing S23
#
# Team Members: William Franzen, Noah Harbor, Brandon Mitchell, Logan Reed
#
# Description: Functions are provided for three different methods of finding
# the roots of on linear equations.  Each method prints out its itermediate
# steps to aid in understanding how they work.  In addition, a function to 
# find fixed points is also provided.
#
# +--------------------------------------------------------------------------+

import math

# Max error allowed, lower to get more accuracy, though more steps are needed
epsilon = 10e-6



# Netwon's Method
# Params:
#   fn: function pointer, the function being tested
#   der: function pointer, the derivative of fn
#   initialGuess: float, where the user thinks the root is
#   maxIterations: int, how many iterations to perform until the root is found
# Description:
#   Uses the Newton-Raphson method to locate the root
def newton(fn, der, initialGuess, maxIterations = 75):
    
    # Ensure error is always larger so while loop is entered
    error = epsilon + 1
    x = initialGuess
    iterations = 0
    
    print("Newton's Method")
    print(f"Initial Guess: {initialGuess}")
    print(f"Max Iterations: {maxIterations}")
    
    while error > epsilon and iterations < maxIterations:
        oldX = x
        x = x - fn(x) / der(x)
        
        error = abs(x - oldX)
        
        iterations += 1
        
        print(f"Iteration: {iterations}   Error: {error}   Current X: {x}")
    
    if iterations >= maxIterations:
        print("Max iterations reached, no root found\n")
    else:
        print(f"Root {x}\n")



# Secant Method
# Params:
#   fn: function pointer, the function being tested
#   xl: float, the lower bound of the range to search
#   xu: float, the upper bound of the range to search
# Description:
#   Uses the secant method to find the root located in the range
#   [xl, xu], runs until the error is less than epsilon
def secant(fn, xl, xu):
    error = epsilon + 1
    iterations = 0
    
    print("Secant Method")
    print(f"Initial Range: [{xl}, {xu}]")
    
    while error > epsilon:
        
        temp = fn(xu)
        xr = xu - temp * (xu - xl) / (temp - fn(xl))
        xl = xu
        xu = xr
        
        error =  abs(xl - xu)
        
        iterations += 1
        
        print(f"Iteration: {iterations}   Error: {error}   Range: [{xl}, {xu}]")

    print(f"Root: {xr}\n")



# False Position Method
# Params:
#   fn: function pointer, the function being tested
#   xl: float, the lower bound of the range to search
#   xu: float, the upper bound of the range to search
# Description:
#   Uses the method of false position to find the root located in the range
#   [xl, xu], runs until the error is less than epsilon
def falsePosition(fn, xl, xu):
    error = epsilon + 1
    oldxr = 0
    iterations = 0
    
    print("Method of False Position")
    print(f"Initial Range: [{xl}, {xu}]")
    
    while error > epsilon:      
    
        temp = fn(xu)
        xr = xu - (temp * (xl - xu) / (fn(xl) - temp))
        
        error = abs(xr - oldxr)
        
        oldxr = xr
        
        if fn(xr) * fn(xl) < 0:
            xu = xr
        else:
            xl = xr
            
        iterations += 1
            
        print(f"Iteration: {iterations}   Error: {error}   Range: [{xl}, {xu}]")
           
    print(f"Root: {xr}\n")



# Fixed-Point Iteration
# Params:
#   fn: function pointer, the function being tested
#   initialGuess: float, the initial guess value
#   iterationCount: int, the amount of times to iterate
# Description:
#   Uses fixed point iteration a set amount of times, given a function,
#   initial guess, and iteration count, Uses a for loop to iterate
def fixedPointIteration(fn, initialGuess, iterationCount):
    x = initialGuess
    
    print("Fixed Point Iteration")
    print(f"Initial guess: {initialGuess}")
    print(f"Iterations: {iterationCount}")

    for i in range(0, iterationCount):
        x = fn(x)
        
        print(f"Iteration: {i + 1}   Fixed Point: {x}")
    
    print(f"Final Fixed Point: {x}\n")



if __name__ == "__main__":
    
    # Question 1 --------------------------------------------------------------
    # The given functions a, b, c, and d.
    g1 = lambda x: x ** 3 - 6 * (x ** 2) + 10 * x - 4
    g2 = lambda x: x ** 3 - 2.4 * x + 2.4
    g3 = lambda x: x ** 3 - 2.9 * x + 2.9
    g4 = lambda x: x ** 3 - 3 * x + 3
    
    # Printing the outputs for each function n' stuff.
    print("Question 1.a -----------------------------------------------------")
    fixedPointIteration(g1, 1.1, 50) #Output: 1.0059526030562829
    
    print("Question 1.b -----------------------------------------------------")
    fixedPointIteration(g2, 1.1, 50) #Output: 1.000000000008191
    
    print("Question 1.c -----------------------------------------------------")
    fixedPointIteration(g3, 1.1, 50) #Output: 1.0
    
    print("Question 1.d -----------------------------------------------------")
    fixedPointIteration(g4, 1.1, 50) #Output: 1.0



    # Question 2 --------------------------------------------------------------
    q2a = lambda x: 2 * x ** 4 + 24 * x ** 3 + 61 * x ** 2 - 16 * x + 1
    q2aDer = lambda x: 8 * x ** 3 + 72 * x ** 2 + 122 * x - 16
    
    q2b = lambda x: x ** 3 + 94 * x ** 2 - 389 * x + 294
    q2bDer = lambda x: 3 * x ** 2 + 188 * x - 389
    
    q2c = lambda x: 0.5 + 0.25 * x ** 2 - x * math.sin(x) - 0.5 * math.cos(2 * x)
    q2cDer = lambda x: 0.5 * x + math.sin(2 * x) - x * math.cos(x) - math.sin(x)
    
    print("Question 2.a -----------------------------------------------------")
    newton(q2a, q2aDer, 0.0)
    newton(q2a, q2aDer, 0.2)
    
    print("Question 2.b -----------------------------------------------------")
    newton(q2b, q2bDer, 2)
    newton(q2b, q2bDer, 2.2)
    newton(q2b, q2bDer, 1.8)
    newton(q2b, q2bDer, 1)
    newton(q2b, q2bDer, 3)
    newton(q2b, q2bDer, 0)
    newton(q2b, q2bDer, 4)
    
    print("Question 2.c -----------------------------------------------------")
    newton(q2c, q2cDer, 0.5 * math.pi)
    newton(q2c, q2cDer, 5 * math.pi)
    
    
    
    # Question 3 --------------------------------------------------------------
    q3 = lambda x: 230 * x ** 4 + 18 * x ** 3 + 9 * x ** 2 - 221 * x - 6
    q3Der = lambda x: 920 * x ** 3 + 54 * x ** 2 + 18 * x - 221
    
    print("Question 3.a -----------------------------------------------------")
    newton(q3, q3Der, -0.5)
    newton(q3, q3Der, 0.5)
    
    print("Question 3.b -----------------------------------------------------")
    secant(q3, -1, 0)
    secant(q3, 0, 1)
    
    print("Question 3.c -----------------------------------------------------")
    falsePosition(q3, -1, 0)
    falsePosition(q3, 0, 1)