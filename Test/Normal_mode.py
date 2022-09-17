import pyscf.hessian.rhf
from pyscf import gto, scf
import numpy as np
from pyscf.data import nist

# Build molecule
mol = gto.Mole()
# mol.atom = [
#    [8, (0., 0., 0.)],
#    [1, (0., -0.757, 0.587)],
#    [1, (0., 0.757, 0.587)]]
mol.atom = [
    [1, (0., 0., 0.)],
    [1, (0., 0., 0.5)]]
mol.basis = '6-31g'
mol.build()
N = mol.natm

# HF calculation for the GS
mf = scf.UHF(mol)
mf.run()
mo_GS = mf.mo_coeff
mocc = mf.mo_occ
LUMO_idx = (mol.nelec[0])  # nbr of alpha ele
HOMO_idx = LUMO_idx - 1

# MOM calculation on HOMO=>LUMO  alpha transition: excited singlet state
mocc[0][HOMO_idx] = 0
mocc[0][LUMO_idx] = 1
mf_ES = scf.UHF(mol)
dm = mf_ES.make_rdm1(mo_GS, mocc)
mf_ES = scf.addons.mom_occ(mf_ES, mo_GS, mocc)
mf_ES.scf(dm)

# Calculate Hessian
# PySCF returns the hessian as 4 dim array (at1, at2, dim1, dim2)
h_GS = mf.Hessian().kernel()
h_ES = mf_ES.Hessian().kernel()
# change the array layout > transpose to (at1, dim1, at2,dim2)
# then reshape to a square matrix
h_GS = h_GS.transpose(0, 2, 1, 3).reshape(N*3, N*3)
h_ES = h_ES.transpose(0, 2, 1, 3).reshape(N*3, N*3)

# Mass-weighted Hessian
atom_masses = mol.atom_mass_list(isotope_avg=True)
atom_masses = np.repeat(atom_masses, 3)
Mhalf = 1/np.sqrt(np.outer(atom_masses, atom_masses))
weighted_h_GS = h_GS*Mhalf
weighted_h_ES = h_ES*Mhalf

# calculate eigenvalues and eigenvectors of the hessian
force_cst_GS, modes_GS = np.linalg.eigh(weighted_h_GS)
force_cst_ES, modes_ES = np.linalg.eigh(weighted_h_ES)

# force to freq
# 3N values -> 3 vib + 3 rot + 3 tr
freq_GS = np.sqrt(np.abs(force_cst_GS))
freq_ES = np.sqrt(np.abs(force_cst_ES))

# sort by increasing frequency
zipped_lists = zip(freq_GS, modes_GS)
sorted_zipped_lists = sorted(zipped_lists)
modes_GS = [element for _, element in sorted_zipped_lists]
freq_GS = np.sort(freq_GS)
freq_ES = np.sort(freq_ES)

# Change units for wavenumbers
to_wavenumber = (nist.HARTREE2J/(nist.ATOMIC_MASS*nist.BOHR_SI**2))**0.5
to_wavenumber *= 1/(2*np.pi)/nist.LIGHT_SPEED_SI*1e-2
# take larger freq (corresponds to vib)
freq_GS = to_wavenumber * freq_GS[-(3*N-5):]  # -6 if mol non linear
freq_ES = to_wavenumber * freq_ES[-(3*N-5):]

print()
print('Frequencies in cm-1')
print('GS: ', freq_GS)
print('ES: ', freq_ES)
print('Normal modes')
print(modes_GS)
