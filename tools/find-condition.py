#RUN scream and place file into 
import numpy as np
import math

# Load matrix from file
filename = "well-1024.mtx"
a = np.loadtxt(filename)

# Calculate condition number
cond_a = np.linalg.cond(a, p=2)

print("Condition number of matrix a:")
print(math.log(cond_a))
