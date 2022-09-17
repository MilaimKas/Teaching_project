import numpy as np
from scipy.integrate import ode

# TODO: S_tdm are 0 ? or the dipole moment ?

class TDSE():
    def __init__(self,S_tdm,DE,epot={'tini':0,'k':1.0,'H':1.0},laser=None):
       # laser is a dict containing the following items
       # E0, Tau, freq, t0,
       # epot is a dict containing the following items
       # tini, k, H

       # system info
       self.S_tdm = S_tdm
       self.DE = DE

       # Field = perturbation function (Laser or Epot)
       self.Field = self.Epot

       if laser is not None:
         self.cst = 1.0
         self.freq = laser.get('freq')
         self.t0 = laser.get('t0')
         self.E0 = laser.get('E0')
         self.Tau = laser.get('tau')
         self.Field = self.Laser
       else:
         self.H = epot.get('H')
         self.k = epot.get('k')
         self.tini = epot.get('tini')

         # assign arbitrary values for initialzing
         self.cst = 1.0
         self.freq = [1.0]
         self.t0 = [0.0]
         self.E0 = [1.0]
         self.Tau = [1.0]

    # define Laser pulse shape
    def Laser(self, t):

       cst =self.cst
       Vt = 0
       #It = 0
       i = complex(0, 1)
       V_single = []

       for w, center, I, tau in zip(self.freq, self.t0, self.E0, self.Tau):
         #It += cst * I * np.exp(-((t - center) / tau) ** 2)
         Vi = cst * I * np.exp(-((t - center) / tau) ** 2) * np.exp(i * (w * t))
         V_single.append(Vi.real)
         Vt += Vi

       return Vt, V_single

    def Epot(self,t):

       if t >= self.tini:
         Vt = self.H*(1-np.exp(-self.k*(t-self.tini)))
       else:
         Vt = 0.0

       return Vt, [Vt]

    # right hand side of the ode
    def f(self, t, c_exp):

       # Vfield in the z axes --> S_tdm only z component
       # todo = generalize for all components

       i = complex(0, 1)
       #f = i*np.einsum('q,pq,pq->p',c_exp,S_tdm[:,:,2],np.exp(i*DE*t))*Laser(t,E0,tau,freq,t0)

       f = (1/i) * np.einsum('q,pq,pq->p', c_exp, self.S_tdm[:, :, 2], np.exp(i * self.DE * t)) \
           * self.Field(t)[0]

       return f


    def Solve(self, c0, tf, dt=0.005):

       print('There are {} integration points'.format(np.rint(tf/dt)))

       # ode solver
       r_G = ode(self.f).set_integrator('zvode', method='bdf') # also try 'Adams' method and/or 'zvode' for complex numb
       r_G.set_initial_value(c0, 0) # set initial condition

       ct_list_R = [] # store complex part of expansion coefficients
       ct_list_C = [] # store real part
       ct_list = []   # store modulus square
       t_list = []    # store time intervals
       V_list = []    # store value of field

       while r_G.successful() and r_G.t < tf:

          res = r_G.integrate(r_G.t+dt)
          # store reald and imaginary part of the expansion coefficients
          ct_list_R.append(res.real)
          ct_list_C.append(res.imag)
           # store normalized probablity amplitudes
          res = np.abs(res)**2/(np.sum(np.abs(res)**2))
          ct_list.append(res)

          # store time steps and value of the fields
          t_list.append(r_G.t+dt)
          V_list.append(self.Field(r_G.t+dt)[1])

       ct_list_R = np.asarray(ct_list_R)
       ct_list_C = np.asarray(ct_list_C)
       ct_list   = np.asarray(ct_list)
       V_list    = np.asarray(V_list)

       return ct_list, t_list, V_list