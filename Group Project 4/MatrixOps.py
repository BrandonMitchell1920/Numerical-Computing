import numpy

def invertMatrix(matrix):
    size = matrix.shape[0]
    #Create an augmented matrix by using an identity matrix.
    aug = numpy.concatenate((matrix, numpy.identity(size)), axis=1)

    #Iterate through each row in the augmented matrix.
    for i in range (size):
        pivot = aug[i][i] #Sets the pivot to each value on the diagonal.
        aug[i, :] /= pivot #Divides each row by the pivot value.
        for j in range(i+1, size): #Eliminates values below each pivot.
            factor = aug[j, i]
            aug[j, :] -= factor * aug[i, :]

    #Iterates through the matrix backwards to do backwards substitution.
    for i in range(size-1, 0, -1):
        for j in range(i-1, -1, -1):
            factor = aug[j, i]
            aug[j, :] -= factor * aug[i, :]

    return aug[:, size:] #Retrieves the inverted matrix from the augmented matrix.
    
def illConditionedValue(matrix):
    # Find the determinant, the numerator of the inequality
    det = numpy.linalg.det(matrix)
    
    # Square the matrix, sum it, and then take the squre root
    denom = (matrix ** 2).sum() ** 0.5
    
    return abs(det) / denom