from pyscf import tools
import numpy as np

def plot(file_name,mo,mol,orb_id=None,make_png=False):
  # mo is a matrix for which each column is a one patricle function
  #  - HF MOs
  #  - natural orbitals
  #  - deformed orbitals
  # orb_id is the index of the orbital to be plotted interactively
  # make_png = True : calling jmol file_name.spt will produce one png file for each single particle function

  # show HOMO by default
  if orb_id is None:
      orb_id = int(mol.nelectron/2)

  molden_file = file_name+".molden"
  jmol_file = file_name+".spt"
  tools.molden.from_mo(mol, molden_file, np.array(mo))

  fspt = open(jmol_file,'w')
  fspt.write('''
  initialize;
  set background [xffffff];
  set frank off
  set autoBond true;
  set bondRadiusMilliAngstroms 66;
  set bondTolerance 0.5;
  set forceAutoBond false;
  load %s
  ''' % molden_file)
  fspt.write('''
  zoom 100;
  rotate 45 y
  rotate 20 x
  axes
  MO COLOR [xff0020] [x0060ff];
  MO COLOR translucent 0.25;
  MO fill noDots noMesh;
  MO titleformat "";
  ''')
  if make_png:
     for i in range(mo.shape[1]):
         fspt.write('MO %d cutoff 0.025;\n' % (i+1))
         fspt.write('write IMAGE 400 400 PNG 90 "MO_{}.png";\n'.format(i+1))
  else:
      fspt.write('MO %d cutoff 0.025;\n' % (orb_id))
  fspt.close()
