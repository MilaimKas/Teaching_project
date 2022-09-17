from pyscf import gto,scf

def get_molecule(name,basis):

   mol = gto.Mole()

   coord = mol_library(name)

   mol.atom = coord
   mol.verbose = 0
   if basis == 1:
     mol.basis = '6-31+g*'
   else:
     mol.basis = '6-31g'
   mol.symmetry = True
   mol.build()

   # HF calculation
   mf = scf.RHF(mol)
   mf.kernel()

   return mol,mf

def mol_library(name):

   if name == 'h2o':
     coord = [
         [8 , (0. , 0.     , 0.)],
         [1 , (0. , -.957 , .587)],
         [1 , (0.2,  .757 , .487)]
     ]

   elif name == 'co2':
       coord = [
           [6, (0., 0., 0.)],
           [8,(0., 0., -1.16)],
           [8,(0., 0., 1.16)]
       ]

   elif name == 'h2':
       coord = '''
       H 0.0 0.0 0.0
       H 0.0 0.0 1.1
       '''

   elif name == 'ch4':
       coord = '''
       C	0.0000000	0.0000000	0.0000000
       H	0.6265510	0.6265510	0.6265510
       H	-0.6265510	-0.6265510	0.6265510
       H	-0.6265510	0.6265510	-0.6265510
       H	0.6265510	-0.6265510	-0.6265510
       '''

   elif name == 'CO':
       coord = '''
     C 0. 0. 0.
     O 0. 0. 1.147 
     '''
   elif name == 'HNO':
       coord = '''
     H  -0.938530   0.9102970  0.0000000
     N  0.0625690   0.5847110  0.0000000
     O  0.0625690  -0.6254090  0.0000000 
     '''
   elif name == 'CO2':
       coord = '''
     C  0.  0.  0.
     O  0.  0.  -1.16
     O  0.  0.  1.16
     '''
   elif name == 'CH2O':
       coord = '''
     O  0.0000000  0.0000000    0.6840550
     C  0.0000000  0.0000000   -0.5377120
     H  0.0000000  0.9482250   -1.1230830
     H  0.0000000  -0.9482250  -1.1230830
     '''
   elif name == 'CH4':
       coord = '''
     C  0.0000000   0.0000000   0.0000000
     H  0.6265510   0.6265510   0.6265510
     H  -0.6265510  -0.6265510  0.6265510
     H  -0.6265510  0.6265510  -0.6265510
     H  0.6265510   -0.6265510 -0.6265510
     '''
   elif name == 'HF':
       coord = '''
     H 0. 0. 0.
     F 0. 0. 0.91
     '''

   else:
     return 'Molecule not found in library'

   return coord
