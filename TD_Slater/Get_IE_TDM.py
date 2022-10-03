import numpy as np


def dip_mo(mf, ks, lmax, orb, Z=1.0, tol=1e-4, dx=0.5, ao_tol=1e-3):
    # mf = hartree Fock object
    # ks = momentum of the outgoing electron (ks = sqrt(2*KE) in au)
    # ao_tol = tolerance at which a MO is considered as a AO
    # orb = identification number of the MO from which the calculation is performed

    '''
    calculate the dipole matrix elements between a MO and a free electron wavefunction

    The free electron wavefunction is defined as a sum over R_kl Y_lm, where k is
    defined by the kinetic energy, and the sum over l is truncated at l_max
    '''

    # MO in MO basis ( [...,0,0,1,0,0,...] vector)
    MO = np.zeros(len(mf.mo_coeff))
    MO[orb] = 1.0

    # convert MO to atomic basis
    MO = np.einsum('ap,p->a', mf.mo_coeff, MO)

    # number of kinetic energy to calculate
    nk = len(ks)
    # dimension
    nq = len(mf.mo_energy)
    # PySCF molecule object
    mol = mf.mol
    # if calculation on a single atom
    atom = False
    if len(mol.atom) == 1:
        atom = True

    # If MO is basically the same as an AO, can be done more efficiently
    check_MO = list(np.abs(MO))
    check_MO.remove(max(check_MO))
    if max(check_MO) < ao_tol:
        print('Molecular orbital is basically an atomic orbital and will be integrated accordingly')
        atom = True

    if atom:

        # AO dipole matrix elements

        # Calculate <xi|r|psi_el> = int(xi.r.R_l.Y_lm dr) where xi are the AOs, for an array of k and l values
        cqklm = np.zeros((3, nq, nk, lmax, 2 * lmax + 1), dtype=np.complex)
        for q in range(nq):
            # only account for AOs with lcao coeff > ao_tol
            if np.abs(MO[q]) < ao_tol : continue
            cqklm[:, q, :, :, :] = dip_ao(mol, q, ks, lmax, Z=Z)

        # cklm for the given MO
        cklm = np.einsum('xqklm,q->xklm', cqklm, MO)

    else:

        # get <phi|r|phi_el> where phi is the given MO
        grid = get_grid(mol, tol=tol, dx=dx)
        cklm = get_c_on_grid(mol, ks, lmax, grid, MO, Z=Z)

    return cklm


def get_grid(mol, tol=1e-5, dx=0.2):
    # Determines a safe integration box
    # given a tolerance for the r value
    # and an integration step dx

    xyzmax = np.zeros((3, 2))
    # list of atomic coords.
    coords = mol.atom_coords()

    # loop over the number of basis function mol.nao
    # to find the optimal xyzmax value depending on the basis set
    for q in range(mol.nao):
        # get information for the given basis function q
        q_atm, q_sh, ql, q_m, qi = get_ao_info(mol, q)[:5]
        # get exponential (es) and contraction coeff (cs) for the shell q_sh
        es, cs = get_es_cs(mol, q, sh=q_sh)

        # smaller exponent
        mine = np.min(es)
        # max r value for exp(-mine/r**2), shouldn't it be mine/log ?
        rmax = np.sqrt(-np.log(tol) / mine)
        # loop over coordinates of atom on which basis is centered
        # from rmax to xyzmax
        for i in range(3):
            # store max value for xyz
            checkmax = coords[q_atm, i] + rmax
            if checkmax > xyzmax[i, 0]:
                xyzmax[i, 0] = checkmax
            # store min value for xyz
            checkmax = coords[q_atm, i] - rmax
            if checkmax < xyzmax[i, 1]:
                xyzmax[i, 1] = checkmax

    xyzmax = np.rint(xyzmax / dx)
    # number of points
    nx = int(xyzmax[0, 0] - xyzmax[0, 1] + 1)
    ny = int(xyzmax[1, 0] - xyzmax[1, 1] + 1)
    nz = int(xyzmax[2, 0] - xyzmax[2, 1] + 1)
    xyzmax = xyzmax * dx

    # x,y,z grid
    x_grid = np.linspace(xyzmax[0, 1], xyzmax[0, 0], nx)
    y_grid = np.linspace(xyzmax[1, 1], xyzmax[1, 0], ny)
    z_grid = np.linspace(xyzmax[2, 1], xyzmax[2, 0], nz)
    print('There are a total of', int(nx * ny * nz), 'points on the integration grid')

    xs, ys, zs = np.meshgrid(x_grid, y_grid, z_grid)
    grid = np.stack([xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)]).transpose()

    return grid


def get_c_on_grid(mol, ks, lmax, grid, MO, Z=1.0):
    # This function evaluates the matrix element <Psi_free | r | MO>

    from scipy.special import spherical_jn
    from scipy.special import sph_harm

    ngrid = len(grid[:, 0])
    MO_grid = np.zeros(ngrid)
    # This section evaluates the orbital on the grid
    nbas = mol.nbas  # orbital basis, nbr of shells
    nqmin = 0
    for bas in range(nbas):
        # evaluate gto for a given shell
        # shls_slice[2-element list] (shl_start, shl_end) -> shell slice
        #  - If given, only part of AOs (shl_start <= shell_id <shl_end) are evaluated.
        #    By default, all shells defined in mol will be evaluated
        gto_vals = mol.eval_gto('GTOval_sph', grid, shls_slice=(bas, bas + 1))
        # store
        ng, n_slc = gto_vals.shape
        nqmax = nqmin + n_slc
        # MO = lcao coeff for the given MO
        MO_grid += np.einsum('gq,q->g', gto_vals, MO[nqmin:nqmax])
        nqmin += n_slc

    # ???
    dV = (grid[1, 2] - grid[0, 2]) ** 3
    lnorm = np.sum(np.square(MO_grid)) * dV

    # This ensures the centre of the orbital is the centre of the free electron spherical harmonics
    xbar = np.einsum('n,nx->x', np.square(MO_grid), grid) * dV / (lnorm)
    grid -= xbar

    # cartesian to spherical coordinates
    nk = len(ks)
    polar_grid = np.zeros((ngrid, 3)) # r, theta, phi

    polar_grid[:, 0] = np.sqrt(np.sum(np.square(grid), axis=1)) # r
    polar_grid[:, 1] = np.arccos(grid[:, 2] / polar_grid[:, 0]) # thet
    polar_grid[:, 2] = np.arctan2(grid[:, 1], grid[:, 0]) # phi

    # This multiplies the value of the wavefunction by (almost) the operator r -> r.Y.MO
    # Almost, because complex spherical harmonics are used, not real.
    # ??????????????
    temp1 = np.zeros((ngrid, 3, 2), dtype=complex)
    # first term is r, second term is Y(phi,theta) for l=1 and last term is MO in cartesian grid
    temp1[:, 0, 0] = polar_grid[:, 0] * sph_harm(-1, 1, polar_grid[:, 2], polar_grid[:, 1]) * MO_grid
    temp1[:, 1, 0] = polar_grid[:, 0] * sph_harm(0, 1, polar_grid[:, 2], polar_grid[:, 1]) * MO_grid
    temp1[:, 2, 0] = polar_grid[:, 0] * sph_harm(1, 1, polar_grid[:, 2], polar_grid[:, 1]) * MO_grid
    temp1[:, 0, 1] = polar_grid[:, 0] * sph_harm(-1, 1, polar_grid[:, 2],polar_grid[:, 1]) * MO_grid
    temp1[:, 1, 1] = polar_grid[:, 0] * sph_harm(0, 1, polar_grid[:, 2],polar_grid[:, 1]) * MO_grid
    temp1[:, 2, 1] = polar_grid[:, 0] * sph_harm(1, 1, polar_grid[:, 2],polar_grid[:, 1]) * MO_grid

    # prefactor i**l
    il = [(1j) ** l for l in range(lmax)]

    # initialize dipole transition element cklm
    cklm = np.zeros((3, nk, lmax, 2 * lmax + 1), dtype=np.complex)

    # Here the coefficients are evaluated, with temp above multiplied by the free electron wavefunction
    # and then integrated on the grid
    for k in range(nk):
        for l in range(lmax):
            # radial part of phi_el, does not depend on ml
            r_kl = spherical_jn(l, ks[k] * polar_grid[:, 0]) * il[l]

            temp2 = temp1 * r_kl[:, None, None]

            for m in range(0, 2 * l + 1):
                Y_lm = sph_harm(m - l, l, polar_grid[:, 2], polar_grid[:, 1])

                cklm[0, k, l, m] = np.sum(temp2[:, 0, 1] * Y_lm)
                cklm[1, k, l, m] = np.sum(temp2[:, 1, 1] * Y_lm)
                cklm[2, k, l, m] = np.sum(temp2[:, 2, 1] * Y_lm)

    # Normalisation (prefactor 4pi/3)
    sf = np.sqrt(4 * np.pi / 3.0) * dV
    cklm *= sf

    return cklm


def dip_ao(mol, q, ks, lmax, Coulomb=False, Z=1.0):
    '''
    calculate the dipole matrix element between a GTO and
    a free electron wavefunction, with a radial bessel function part
    and real spherical harmonic angular part
    1) angular momentum selection rules
    2) angular part with clebsch-gordon coefficients
    3) radial part numerically integrated
    '''

    nk = len(ks)
    cklm = np.zeros((3, nk, lmax, 2 * lmax + 1), dtype=np.complex)

    for l in range(lmax):
        ang_part = cgc_ang_int(mol, q, l)[0]
        rad_part = radial_int(mol, q, ks, l, Coulomb, Z=Z)
        cklm[:, :, l, :(2 * l + 1)] = np.einsum('xm,k->xkm', np.conjugate(ang_part), rad_part)

    return cklm


def get_total_cs(cklm):

    # Return the total cross-section z polarized light and averaged over x,y,z

    nk = len(cklm[0,:,0,0])
    cs = np.zeros((nk,2),dtype = complex)

    # sum over l and ml
    for k in range(nk):
        cs[k,0] = np.sum(np.conjugate(cklm[1,k,:,:])*cklm[1,k,:,:])
        cs[k,1] = np.sum(np.conjugate(cklm[:,k,:,:])*cklm[:,k,:,:])

    cs[:,1] /= 3.0

    if np.any(np.abs(cs.imag) > 1e-10):
       print('CS has imaginary part!',cs)
    return np.real(cs)

def prefactor(obj,e_i,e_k,ks):

    # Convert an object whose first dimension is in k
    # to a cross-section with the suitable prefactor
    # e_i is the ionisation energy of the system
    # e_k and k are the kinetic energies and momenta of the free electron
    # all in a.u.

    e_tot = e_k+e_i
    n = len(ks)
    for i in range(n):
        obj[i] *= e_tot[i]*ks[i]

    c = 137.0
    sf = 8*np.pi/c

    return obj*sf


def radial_int(mol, q, ks, l, Coulomb=False, Z=1.0):
    from scipy.integrate import quad, trapz
    from scipy.special import spherical_jn

    q_atm, q_sh, ql, q_m, qi = get_ao_info(mol, q)[:5]
    es, cs = get_es_cs(mol, q, sh=q_sh) # get exponent and contraction coeff.

    nprim = len(es)
    ncntr = len(cs)
    nk = len(ks)
    integral = np.zeros(nk)

    # Determines a safe integration length and grid
    mine = np.min(es)
    tol = 1e-12
    r = np.sqrt(-np.log(tol) / mine)
    nr = np.int(r * 10)
    rmax = nr / 10.0
    nr += 1
    rs = np.linspace(0, rmax, nr) # integration step

    for ik, k in enumerate(ks):

        if Coulomb:
            hypfunc = eval_cw(rs, k, l, Z=1.0)
        else:
            hypfunc = spherical_jn(l, k * rs)

        for j in range(nprim):
            ys = r3G(rs, es[j], ql)   # r^l*r*r^2*G*norm_G
            ys *= hypfunc             # radial part of electronic WF
            di = trapz(ys, x=rs)      # perform integral with step rs

            if np.abs(di.imag) > 1e-10:
                print(di)
                print('The Coulomb Wave has a non-negligible imaginary part')
                exit()
            else:
                di = di.real

            integral[ik] += cs[j, qi] * di

    return integral


def eval_cw(r, k, l, Z=1.0):
    from scipy.special import gamma
    from mpmath import hyp1f1

    eta = -Z / k
    cw = np.exp(-(np.pi * eta / 2 + k * r * 1j))
    cw *= np.abs(gamma(l + 1 + eta * 1j))
    cw *= (2 * k * r) ** l
    norm = 1.0 / gamma(2 * l + 2)
    cw *= norm

    for i in range(len(r)):
        cw[i] *= hyp1f1(l + 1 - eta * 1j, 2 * l + 2, 2 * k * r[i] * 1j)

    return cw


def r3G(r, a, ql):
    from pyscf.gto import gto_norm
    '''The product of r, a gaussian function, and r^2'''

    ans = r ** (ql + 3)
    ans *= np.exp(-a * r ** 2)
    ans *= gto_norm(ql, a)

    return ans.astype(complex)


def get_es_cs(mol, q, sh=None):
    if sh is None: sh = get_ao_info(mol, q)[1]
    cs = mol.bas_ctr_coeff(sh)
    es = mol.bas_exp(sh)

    return es, cs


def cgc_ang_int(mol, q, l):
    # q is the atomic orbital, l is the angular momentum of the free electron

    Ym = [-1, 0, 1]
    ql, qm = get_lm(mol, q)
    q2 = get_q_mm(q, ql, qm)
    scale = np.sqrt((2 * l + 1) / (2 * ql + 1.0))
    scale *= cgc(ql, l, 0, 0, 0)

    ang_int1 = np.zeros((3, 2 * l + 1), dtype=np.complex)
    if qm != 0:
        ang_int2 = np.zeros((3, 2 * l + 1), dtype=np.complex)

    temp1 = np.zeros((3, 2 * l + 1), dtype=np.complex)
    if qm != 0: temp2 = np.zeros((3, 2 * l + 1), dtype=np.complex)

    onrt2 = np.sqrt(0.5)
    for m in range(-l, l + 1):
        for i in range(3):
            temp1[i, m + l] = cgc(ql, l, Ym[i], qm, m)
            if qm != 0:
                temp2[i, m + l] = cgc(ql, l, Ym[i], -qm, m)

    temp1 *= scale

    if qm == 0:
        return temp1, q, temp1

    if qm != 0:

        temp2 *= scale

        eqm = qm % 2
        if qm < 0:
            ang_int1 = -1j * (temp1 - temp2 * (-1) ** eqm) * onrt2
            ang_int2 = (temp1 + temp2 * (-1) ** eqm) * onrt2
        else:
            ang_int2 = -1j * (temp2 - temp1 * (-1) ** eqm) * onrt2
            ang_int1 = (temp2 + temp1 * (-1) ** eqm) * onrt2

        return ang_int1, q2, ang_int2


def get_lm(mol, q):
    return get_ao_info(mol, q)[2:4]


def get_q_mm(q, l, m):
    if l > 1:
        q_new = q - m * 2
    elif l == 1:
        q_new = q + m
    else:
        q_new = q

    return q_new


def cgc(ll, lr, Ym, ml, mr):
    '''
    a function for evaluating the clebsch-gordon coefficients for the special
    case of selection rules \Delta l=\pm 1 and \Delta m = 0, \pm 1
    (l)eft, (r)ight
    '''
    c = 0.0
    dl = abs(ll - lr)
    dm = ml - mr
    if dl == 1 and dm == Ym:
        if ll > lr:
            if Ym == 1:
                c = np.sqrt((lr + ml) * (lr + ml + 1) / ((2 * lr + 1) * (2 * lr + 2.0)))
            elif Ym == 0:
                c = np.sqrt((lr - ml + 1) * (lr + ml + 1) / ((2 * lr + 1) * (lr + 1.0)))
            elif Ym == -1:
                c = np.sqrt((lr - ml) * (lr - ml + 1) / ((2 * lr + 1) * (2 * lr + 2.0)))
        else:
            if Ym == 1:
                c = np.sqrt((lr - ml) * (lr - ml + 1) / (2 * lr * (2 * lr + 1.0)))
            elif Ym == 0:
                c = -np.sqrt((lr - ml) * (lr + ml) / (lr * (2 * lr + 1.0)))
            elif Ym == -1:
                c = np.sqrt((lr + ml + 1) * (lr + ml) / (2 * lr * (2 * lr + 1.0)))

    return c



def get_ao_info(mol, q):
    # arg:
    #    q = basis function index (from mol.nao)
    # returns:
    #    index for: atom, shell, l value, m value, ao contraction index

    # nbr of shells
    nsh = mol.nbas
    ao_sh = 0
    # index location for ao 
    loc = mol.ao_loc_nr()
    while q >= loc[ao_sh + 1]:
        ao_sh += 1

    # store l and atom index
    ao_l = mol.bas_angular(ao_sh)
    ao_atm = mol.bas_atom(ao_sh)

    # mol.bas_nctr() = nbr of contracted GTO for the given shell
    if mol.bas_nctr(ao_sh) > 1:
        ao_cntr_index = (q - loc[ao_sh]) / (2 * ao_l + 1)
    else:
        ao_cntr_index = 0

    degen = 2 * ao_l + 1
    m = (q - loc[ao_sh]) % degen - ao_l
    if ao_l == 1:
        m -= 1
        if m == -2: m = 1
    ao_m = m

    return ao_atm, ao_sh, ao_l, ao_m, ao_cntr_index



def center_geom(mol):
    natom = len(mol.atom)
    coords = np.zeros((natom, 3))
    for i in range(natom):
        coords[i] = mol.atom[i][1]
    xbar = np.sum(coords, axis=0) / natom
    for i in range(natom):
        mol.atom[i][1] -= xbar
