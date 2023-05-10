# =========================================
# M 410
# Group Project 4
# 
# Authors: William Franzen, Noah Harbor, Brandon Mitchell, Logan Reed
#
# Description:  Demonstrates the affect of rounding error on the solution
#               of a augmented matrix by rounding fractions to different 
#               precisions.  Linear and quadratic least squares fits are also
#               explored.
# =========================================

import numpy
from numpy.polynomial import polynomial
from matplotlib import pyplot as plt

from MatrixOps import invertMatrix, illConditionedValue

# ------------------
# Problem 1a
# Description:
#   Does row reduction on the augmented matrix [A I] to solve for inverse(A).
# ------------------

A1 = numpy.array([
    [1, 1/2, 1/3],
    [1/2, 1/3, 1/4],
    [1/3, 1/4, 1/5]
])

b = numpy.array([
    [1],
    [2],
    [2],
])

print("A1 =")
print(A1)

invA1 = invertMatrix(A1)

print("Inverse(A1) =")
print(invA1)

print("Inverse(A1) x b =")
print(invA1 @ b) #Python uses the @ symbol for cross products.

print("IllConditionedValue(A1) =")
print(illConditionedValue(A1))
print()



# ------------------
# Problem 1b
# Description:
#   Does row reduction on the augmented matrix [A I] to solve for inverse(A).
#   Shows how rouding errors can have a large impact on the result.
# ------------------

A2 = numpy.array([
    [1, 0.5, 0.33],
    [0.5, 0.33, 0.25],
    [0.33, 0.25, 0.2]
])

print("A2 =")
print(A2)

invA2 = invertMatrix(A2)

print("Inverse(A2) =")
print(invA2)

print("Inverse(A2) x b =")
print(invA2 @ b)

print("IllConditionedValue(A2) =")
print(illConditionedValue(A2))
print()



# ------------------
# Problem 1b
# Description:
#   Does row reduction on the augmented matrix [A I] to solve for inverse(A).
#   Rouding error is not as severe so we can compare with 1a and 1b.
# ------------------

A3 = numpy.array([
    [1, 0.5, 0.333],
    [0.5, 0.333, 0.25],
    [0.333, 0.25, 0.2]
])

print("A3 =")
print(A3)

invA3 = invertMatrix(A3)

print("Inverse(A3) =")
print(invA3)

print("Inverse(A3) x b =")
print(invA3 @ b)

print("IllConditionedValue(A3) =")
print(illConditionedValue(A3))
print()



# ------------------
# Problem 2
# Description:
#   Finding the least squares fit of test scores.
# ------------------

testScores = numpy.array([5, 8, 10, 12, 14, 18, 22, 24])
grades = numpy.array([0, 1.3, 2, 1.7, 2.3, 3, 4, 3.3])

lineCoeff = polynomial.polyfit(testScores, grades, 1)

# y = mx + b
lineFunc = lambda x: lineCoeff[0] + lineCoeff[1] * x
lineX = numpy.linspace(testScores[0], testScores[-1], 10)
lineY = [lineFunc(x) for x in lineX]

print("y = a + b * x")
print("a =", lineCoeff[0])
print("b =", lineCoeff[1])

plt.scatter(testScores, grades)
plt.plot(lineX, lineY)
plt.title("Q2 Test Scores vs. Grades, Linear Least-Squares Fit")

# x = (y - b) / m
scoreForB = (3.0 - lineCoeff[0]) / lineCoeff[1]

print("A test score of about", scoreForB, "predicts a B for this course.\n")



# ------------------
# Problem 3
# Description:
#   Quadratic least-square fit
# ------------------

dataX = numpy.array([2.3, 3.0, 5.8, 6.4, 7.2])
dataY = numpy.array([-3.1, 0.2, 1.5, 0, -2.3])

quadCoeff = polynomial.polyfit(dataX, dataY, 2)

quadFunc = lambda x: quadCoeff[0] + quadCoeff[1] * x + quadCoeff[2] * x ** 2
quadX = numpy.linspace(dataX[0], dataX[-1], 50)
quadY = [quadFunc(x) for x in quadX]

print("y = a + b * x + c * x ^ 2")
print("a =", quadCoeff[0])
print("b =", quadCoeff[1])
print("c =", quadCoeff[2])

plt.figure()
plt.scatter(dataX, dataY)
plt.plot(quadX, quadY)
plt.title("Q3 Quadratic Least-Squares Fit")

plt.show()