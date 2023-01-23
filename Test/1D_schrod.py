import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Define constants
hbar = 1.0
m = 20.0
a = 10.0
N = 1000

# Define x-vector from 0 to L with N steps
x = np.linspace(0, a, N)

# Define step size
dx = x[1] - x[0]

# Create the diagonal of the Hamiltonian
V = np.zeros(N)

# Define the potential
for i in range(N):
    V[i] = (x[i]-5)**2
    #V[i] = np.sin(x[i])**2
    # add boundary conditions
    V[0] = V[-1] = 1000

# Define the Laplacian
T = -(hbar**2)/(2*m) * np.diag(np.ones(N-1),-1) + np.diag(np.ones(N-1),1)
T[0,N-1] = -1
T[N-1,0] = -1
T = T/(dx**2)

# Define the Hamiltonian
H = T + np.diag(V)

# Diagonalize the Hamiltonian yielding the wavefunctions and their energies
E, psi = eigh(H)

# Normalize energy wrt to GS
#E = E-E[0] 

# Normalize the wavefunctions
psi = psi / np.sqrt(dx)

# Plot the first three wavefunctions
plt.plot(x, psi[:,0]**2, label='$\psi_1$')
plt.plot(x, psi[:,1]**2, label='$\psi_2$')
plt.plot(x, psi[:,2]**2, label='$\psi_3$')

plt.plot(x, V, 'k--', label='V')
plt.xlabel('$x$')
plt.ylabel('$\psi(x)$')
plt.legend()

plt.ylim(0, 5)
plt.show()