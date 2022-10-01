# TODO: absorption spectrum, RIXS moments, quadrupole moments
# TODO: sort E, and DE, S_tdm, Orb_ind accordingly
# TODO: how to deal with degeneracy ? Remove degenerate states or MO ?


import numpy as np
from Get_IE_TDM import dip_mo, get_total_cs, prefactor
from TDSE_solve import TDSE
import matplotlib.pyplot as plt
from Slater_State import Slater
from random import randint, seed
from Molecule import get_molecule

seed(6) # for nice colors

# conversion factor
ev_to_H = 0.0367493


# what to calculate: TDSE, IE or plot_orb
todo = ('TDSE',)

# get molecule
mol,mf = get_molecule('CH4',1)

print(mf.mo_energy)

# build Slater states
S_State = Slater(mol, mf)  # slater state object
S_State.build()
det = S_State.det  # Slater determinants
Ene = S_State.Ene  # energie of the Slater states
DE = S_State.DE  # list of Delta(E)
S_tdm = S_State.S_tdm  # transition dipole elements
Orb_ex_ind = np.asarray(S_State.Orb_ex_ind)
#print(Ene)
#print(S_tdm[0,1,:])
#print(Orb_ex_ind[1],Orb_ex_ind[2])

###########################################################################################
# Plot orbitals with jmol
# ----------------------------------------------------------------------------------------
###########################################################################################

if 'Plot_orb' in todo:
    from Jmol import plot

    # output file name
    file = 'Mol_HF'
    # plot list of orbital with jmol
    mo = mf.mo_coeff
    plot(file,mo,mol,make_png=True)


###########################################################################################
# TD Schrödinger equation with field
# ----------------------------------------------------------------------------------------
###########################################################################################

if 'TDSE' in todo:
    # INPUT
    #-------

    # Laser pulse parameters in a.u
    E0 = [5] # intensities
    tau = [0.8] #width
    freq = [DE[0,1]]  # frequencies/energies of the field
    t0 = [1.0] # time at max(E0)
    # integration parameters
    tf = 3 # final time
    laser = {'E0':E0,'tau':tau,'freq':freq,'t0':t0}

    # exponentialy switched potential
    tini = 0.0
    k = 2.0
    H = 1.0
    epot = {'tini':tini,'k':k,'H':H}

    # Expansion coefficients, initial condition
    # todo: initial value of c has complex component ?
    c0 = np.zeros((len(det)),dtype=complex)
    c0[0] = complex(1.0,1.0)

    # Solve TDSE
    #-------------------
    my_tdse = TDSE(S_tdm,DE,laser=laser)
    Result = my_tdse.Solve(c0, tf)
    ct_list, t_list, V_list = Result

    # Plot Results
    # ------------------
    # colors
    colors = []
    for i in range(len(det)):
      colors.append('#%06X' % randint(0, 0xFFFFFF))
    color_field = ('black','red','blue')

    # initialize figure
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.set_xlabel("t")
    ax.set_ylabel("$|c_i(t)|^2$")

    # plot each probability amplitude (|c^2|) as a function of time
    thres=0.1
    for i in range(len(det)):
      if np.max(ct_list[:,i]) > thres:
         ax.plot(t_list,ct_list[:,i],label=f"{Orb_ex_ind[i,0]}->{Orb_ex_ind[i,1]}",color=colors[i])

    inset = fig.add_axes([0.76, 0.25, 0.17, 0.25]) # left, bottom, width, height
    inset.set_title("Field")
    inset.set_xlabel("t")

    for i in range(V_list[0,:].shape[0]):
       inset.plot(t_list,V_list[:,i],color=color_field[i])

    ax.legend(bbox_to_anchor=(1.05, 1), ncol=2, shadow=True, title="Legend", fancybox=True,loc=2, borderaxespad=0.)
    plt.show()

###########################################################################################
# Ionization cross section
# ----------------------------------------------------------------------------------------
###########################################################################################

if 'IE' in todo:

    # INPUT

    lmax = 2 # The maximum l of the free electron
    mine = 0.0 # The minimum K.E. of the free electron in eV
    maxe = 20 # The maximum K.E. of the free electron in eV
    de = 0.5 # The intervals of of K.E of the free electron
    #orb = int(mol.nelectron/2) # MO you want the overlap with - here the HOMO
    orb = 1
    IE = abs(mf.mo_energy[orb-1]) # ionization energy in au
    print(IE)

    nk = np.int((maxe - mine) / de + 0.1) + 1
    lmax += 1
    e_k = np.linspace(mine,maxe,nk)
    e_k *= ev_to_H
    ks = np.sqrt(e_k*2) # kinetic energy to momentum

    cklm = dip_mo(mf,ks,lmax,orb)
    cs = get_total_cs(cklm)
    cs = prefactor(cs,IE,e_k,ks)
    plt.plot(e_k,cs)
    plt.ylabel('$\sigma (au)$')
    plt.xlabel('$E_{k,ele} (au)$')
    plt.show()

###########################################################################################
# Absorption spectrum
# ----------------------------------------------------------------------------------------
###########################################################################################

#if 'abs' in todo:
