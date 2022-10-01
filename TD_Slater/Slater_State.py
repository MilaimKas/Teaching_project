"""
Class to calculate Slater-State for a given molecule and basis sets
The Slater States are single excitation from a closed-shell GS
"""

import numpy as np
from pyscf import scf,gto

# TODO: sort Ene, ST_tdm and det as a function of energy
# TODO: diagonal element of S_tdm are the eigenstates of H_0 (energies Ej) ?

class Slater:
    def __init__(self,mol,mf):
        # HF related properties
        self.mol = mol
        self.mf = mf
        self.dim = mol.nao_nr()
        self.mo_coeff = mf.mo_coeff  # column=MO, row=AO
        self.mo_ene = mf.mo_energy
        self.nocc = mol.nelectron // 2
        self.nvir = self.mo_coeff[:, self.nocc:].shape[1]
        self.h1e = mf.get_hcore()  # single particle hamiltonian

        # nuclear charge
        self.charges = mol.atom_charges()
        self.coords = mol.atom_coords()
        self.nucl_dip = np.einsum('i,ix->x', self.charges, self.coords)

        # dipole integrals in AOs basis set
        with mol.with_common_orig((0, 0, 0)):
            self.ao_dip  = mol.intor_symmetric('int1e_r', comp=3)
            #self.ao_quad = mol.intor_symmetric('int')

        # initialize properties of Slater States

        # Store all Slater determinant
        self.det = []
        # index of the single excitation i -> a respective to the GS for each Slater states
        self.Orb_ex_ind = []
        self.Orb_ex_ind.append([0, 0])  # GS excitation indices = 0,0
        # xyz components of the tdm between Slater state q and p
        # diagonal elements are the dipole moments
        self.S_tdm = None
        # electronic energy of the Slater states
        self.Ene = None
        # energy difference between Slater states
        self.DE = None

    def build(self):

      # Transition dipole moment for all possible orbital change <phi_i|d|phi_a>
      # O_tdm[i,a] is the TDM of the i --> a excitation
      # Note = O_tdm are symmetric <phi_i|d|phi_a> = <phi_a|d|phi_i>
      # O = orbitals

      dim = self.dim
      mo_coeff = self.mo_coeff
      ao_dip = self.ao_dip
      nocc = self.nocc
      nvir = self.nvir

      O_tdmx = np.zeros((dim, dim))
      O_tdmy = np.zeros((dim, dim))
      O_tdmz = np.zeros((dim, dim))
      for i in range(dim):
          for a in range(dim):
              O_tdmx[i, a] = np.einsum('j,k,jk', mo_coeff[:, i], mo_coeff[:, a].conj(), ao_dip[0])
              O_tdmy[i, a] = np.einsum('j,k,jk', mo_coeff[:, i], mo_coeff[:, a].conj(), ao_dip[1])
              O_tdmz[i, a] = np.einsum('j,k,jk', mo_coeff[:, i], mo_coeff[:, a].conj(), ao_dip[2])

      # Create GS determinant
      GS_det = np.concatenate((np.repeat(2, nocc), np.repeat(0, nvir)))
      self.det.append(GS_det)
      # Create all possible singly excited Slater determinant from the closed-shell GS
      # len(det) = p = i*a
      for i in range(nocc):
        for a in range(nvir):
          tmp_vec = GS_det.copy()
          tmp_vec[i] = 1
          tmp_vec[a + nocc] = 1
          self.det.append(tmp_vec)

      # xyz components of the tdm between Slater state q and p
      # diagonal elements are the dipole moments
      self.S_tdm = np.zeros((len(self.det), len(self.det), 3))
      # electronic energy of the Slater states
      self.Ene = np.zeros(len(self.det))
      # energy difference between Slater states
      self.DE = np.zeros((len(self.det), len(self.det)))

      for p in range(len(self.det)):
        # density matrix for the given determinant
        dm = np.einsum('pi,ij,qj->pq', mo_coeff, np.diag(self.det[p]), mo_coeff.conj())
        # electronic energy from dm
        self.Ene[p] = scf.hf.energy_elec(self.mf, dm=dm)[0]
        for q in range(len(self.det)):
          if p != q:
            # look for difference in MOs
            tmp_vec = self.det[q] - self.det[p]
            # Store indices of MOs
            i = np.asarray(np.where(tmp_vec == -1)).flatten()[0]
            a = np.asarray(np.where(tmp_vec == 1)).flatten()[0]
            # Store excitation indices for GS-->ES
            if p == 0:
                self.Orb_ex_ind.append([i, a])
            # Store tdm for p-->q transition
            self.S_tdm[p, q, 0] = O_tdmx[i, a]
            self.S_tdm[p, q, 1] = O_tdmy[i, a]
            self.S_tdm[p, q, 2] = O_tdmz[i, a]
            # Calculate energy for p and q
            # DE = Delta(mo_ene) + Delta(h1e)
            # --> does not work
            # TODO: check energy of Slater determinant
            # ha = np.einsum('j,jj', mo_coeff[:, a], h1e)
            # hi = np.einsum('j,jj', mo_coeff[:, i], h1e)
            # DE[p,q] = (mo_ene[a] - mo_ene[i]) + (ha - hi)
            dm = np.einsum('pi,ij,qj->pq', mo_coeff, np.diag(self.det[q]), mo_coeff.conj())
            E_tmp = scf.hf.energy_elec(self.mf, dm=dm)[0]
            self.DE[p, q] = self.Ene[p] - E_tmp
          elif p == q:
            # electric dipole moment of a Slater state as diagonal element of S_tdm
            self.S_tdm[p, q, 0] = self.nucl_dip[0] - np.einsum('jj,j', O_tdmx, self.det[p])
            self.S_tdm[p, q, 1] = self.nucl_dip[1] - np.einsum('jj,j', O_tdmy, self.det[p])
            self.S_tdm[p, q, 2] = self.nucl_dip[2] - np.einsum('jj,j', O_tdmz, self.det[p])
            # diagonal element of DE are 0
            self.DE[p, q] = 0.0

      # Test dipole moment
      # note: in order to get the total dipole moment, use: (nucl_dip-el_dip) (*2.541746 in Debye)
      # print(2.5417*(nucl_dip[2]-S_tdm[0,0,2]),mf.dip_moment())
      # checked !

      # Test energy
      # DeltaE between 2 Slater det should be Delta MO_ene ?
      # TODO: verify

if __name__ == '__main__':
    mol = gto.Mole()

    # mol.atom ='''
    # H 0 0 0
    # F 0 0 1.1
    # '''

    # water
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -.957, .587)],
        [1, (0.2, .757, .487)]]

    mol.verbose = 0
    mol.basis = 'sto3g'
    mol.symmetry = True
    mol.build()

    # HF calculation
    mf = scf.RHF(mol)
    mf.kernel()

    S_State = Slater(mol,mf)
    S_State.build()
    det = S_State.det
    Ene = S_State.Ene
    DE = S_State.DE
    S_tdm = S_State.S_tdm
    Orb_ex_ind = np.asarray(S_State.Orb_ex_ind)

    print('Transition dipole moments in z')
    print(S_tdm[:,:,2])
    print()

    print('Dipole moment of the Slater states')
    print(np.diag(S_tdm[:,:,2]))
    print()

    print('Energies of the Slater states')
    print(Ene)
    print()

    print('Orbital excitation')
    print(Orb_ex_ind)
    print()

    print(" Info for 0 -> 1 transition")
    print(det[0],'->',det[1])
    print('DE= ',DE[0,1])
    print('TDM= ',S_tdm[0,1])

