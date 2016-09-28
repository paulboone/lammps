/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "dihedral.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "atom_masks.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   set dihedral contribution to Vdwl and Coulombic energy to 0.0
   DihedralCharmm will override this
------------------------------------------------------------------------- */

Dihedral::Dihedral(LAMMPS *lmp) : Pointers(lmp)
{
  energy = 0.0;
  writedata = 0;

  allocated = 0;

  maxeatom = maxvatom = maxhatom = 0;
  eatom = NULL;
  vatom = NULL;
  hatom = NULL;
  setflag = NULL;

  execution_space = Host;
  datamask_read = ALL_MASK;
  datamask_modify = ALL_MASK;

  copymode = 0;
}

/* ---------------------------------------------------------------------- */

Dihedral::~Dihedral()
{
  if (copymode) return;

  memory->destroy(eatom);
  memory->destroy(vatom);
  memory->destroy(hatom);
}

/* ----------------------------------------------------------------------
   check if all coeffs are set
------------------------------------------------------------------------- */

void Dihedral::init()
{
  if (!allocated && atom->ndihedraltypes)
    error->all(FLERR,"Dihedral coeffs are not set");
  for (int i = 1; i <= atom->ndihedraltypes; i++)
    if (setflag[i] == 0) error->all(FLERR,"All dihedral coeffs are not set");
  init_style();
}

/* ----------------------------------------------------------------------
   setup for energy, virial computation
   see integrate::ev_set() for values of eflag (0-3) and vflag (0-6)
------------------------------------------------------------------------- */

void Dihedral::ev_setup(int eflag, int vflag, int alloc)
{
  int i,n,j;

  hflag_atom = 1;
  evflag = 1;

  eflag_either = eflag;
  eflag_global = eflag % 2;
  eflag_atom = eflag / 2;

  vflag_either = vflag;
  vflag_global = vflag % 4;
  vflag_atom = vflag / 4;

  // reallocate per-atom arrays if necessary

  if (eflag_atom && atom->nmax > maxeatom) {
    maxeatom = atom->nmax;
    if (alloc) {
      memory->destroy(eatom);
      memory->create(eatom,comm->nthreads*maxeatom,"dihedral:eatom");
    }
  }
  if (vflag_atom && atom->nmax > maxvatom) {
    maxvatom = atom->nmax;
    if (alloc) {
      memory->destroy(vatom);
      memory->create(vatom,comm->nthreads*maxvatom,6,"dihedral:vatom");
    }
  }
  if (hflag_atom && atom->nmax > maxhatom) {
    maxhatom = atom->nmax;
    memory->destroy(hatom);
    memory->create(hatom,comm->nthreads*maxhatom,9,"dihedral:hatom");
  }
  // zero accumulators

  if (eflag_global) energy = 0.0;
  if (vflag_global) for (i = 0; i < 6; i++) virial[i] = 0.0;
  if (eflag_atom && alloc) {
    n = atom->nlocal;
    if (force->newton_bond) n += atom->nghost;
    for (i = 0; i < n; i++) eatom[i] = 0.0;
  }
  if (vflag_atom && alloc) {
    n = atom->nlocal;
    if (force->newton_bond) n += atom->nghost;
    for (i = 0; i < n; i++) {
      vatom[i][0] = 0.0;
      vatom[i][1] = 0.0;
      vatom[i][2] = 0.0;
      vatom[i][3] = 0.0;
      vatom[i][4] = 0.0;
      vatom[i][5] = 0.0;
    }
  }
  if (hflag_atom) {
    n = atom->nlocal;
    if (force->newton_bond) n += atom->nghost;
    for (i = 0; i < n; i++) {
      for (j = 0; j < 9; j++) {
        hatom[i][j] = 0.0;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   tally energy and virial into global and per-atom accumulators
   virial = r1F1 + r2F2 + r3F3 + r4F4 = (r1-r2) F1 + (r3-r2) F3 + (r4-r2) F4
          = (r1-r2) F1 + (r3-r2) F3 + (r4-r3 + r3-r2) F4
          = vb1*f1 + vb2*f3 + (vb3+vb2)*f4
------------------------------------------------------------------------- */

void Dihedral::ev_tally(int i1, int i2, int i3, int i4,
                        int nlocal, int newton_bond,
                        double edihedral, double *f1, double *f3, double *f4,
                        double vb1x, double vb1y, double vb1z,
                        double vb2x, double vb2y, double vb2z,
                        double vb3x, double vb3y, double vb3z)
{
  double edihedralquarter,v[6];

  if (eflag_either) {
    if (eflag_global) {
      if (newton_bond) energy += edihedral;
      else {
        edihedralquarter = 0.25*edihedral;
        if (i1 < nlocal) energy += edihedralquarter;
        if (i2 < nlocal) energy += edihedralquarter;
        if (i3 < nlocal) energy += edihedralquarter;
        if (i4 < nlocal) energy += edihedralquarter;
      }
    }
    if (eflag_atom) {
      edihedralquarter = 0.25*edihedral;
      if (newton_bond || i1 < nlocal) eatom[i1] += edihedralquarter;
      if (newton_bond || i2 < nlocal) eatom[i2] += edihedralquarter;
      if (newton_bond || i3 < nlocal) eatom[i3] += edihedralquarter;
      if (newton_bond || i4 < nlocal) eatom[i4] += edihedralquarter;
    }
  }

  if (vflag_either) {
    v[0] = vb1x*f1[0] + vb2x*f3[0] + (vb3x+vb2x)*f4[0];
    v[1] = vb1y*f1[1] + vb2y*f3[1] + (vb3y+vb2y)*f4[1];
    v[2] = vb1z*f1[2] + vb2z*f3[2] + (vb3z+vb2z)*f4[2];
    v[3] = vb1x*f1[1] + vb2x*f3[1] + (vb3x+vb2x)*f4[1];
    v[4] = vb1x*f1[2] + vb2x*f3[2] + (vb3x+vb2x)*f4[2];
    v[5] = vb1y*f1[2] + vb2y*f3[2] + (vb3y+vb2y)*f4[2];

    if (vflag_global) {
      if (newton_bond) {
        virial[0] += v[0];
        virial[1] += v[1];
        virial[2] += v[2];
        virial[3] += v[3];
        virial[4] += v[4];
        virial[5] += v[5];
      } else {
        if (i1 < nlocal) {
          virial[0] += 0.25*v[0];
          virial[1] += 0.25*v[1];
          virial[2] += 0.25*v[2];
          virial[3] += 0.25*v[3];
          virial[4] += 0.25*v[4];
          virial[5] += 0.25*v[5];
        }
        if (i2 < nlocal) {
          virial[0] += 0.25*v[0];
          virial[1] += 0.25*v[1];
          virial[2] += 0.25*v[2];
          virial[3] += 0.25*v[3];
          virial[4] += 0.25*v[4];
          virial[5] += 0.25*v[5];
        }
        if (i3 < nlocal) {
          virial[0] += 0.25*v[0];
          virial[1] += 0.25*v[1];
          virial[2] += 0.25*v[2];
          virial[3] += 0.25*v[3];
          virial[4] += 0.25*v[4];
          virial[5] += 0.25*v[5];
        }
        if (i4 < nlocal) {
          virial[0] += 0.25*v[0];
          virial[1] += 0.25*v[1];
          virial[2] += 0.25*v[2];
          virial[3] += 0.25*v[3];
          virial[4] += 0.25*v[4];
          virial[5] += 0.25*v[5];
        }
      }
    }

    if (vflag_atom) {
      if (newton_bond || i1 < nlocal) {
        vatom[i1][0] += 0.25*v[0];
        vatom[i1][1] += 0.25*v[1];
        vatom[i1][2] += 0.25*v[2];
        vatom[i1][3] += 0.25*v[3];
        vatom[i1][4] += 0.25*v[4];
        vatom[i1][5] += 0.25*v[5];
      }
      if (newton_bond || i2 < nlocal) {
        vatom[i2][0] += 0.25*v[0];
        vatom[i2][1] += 0.25*v[1];
        vatom[i2][2] += 0.25*v[2];
        vatom[i2][3] += 0.25*v[3];
        vatom[i2][4] += 0.25*v[4];
        vatom[i2][5] += 0.25*v[5];
      }
      if (newton_bond || i3 < nlocal) {
        vatom[i3][0] += 0.25*v[0];
        vatom[i3][1] += 0.25*v[1];
        vatom[i3][2] += 0.25*v[2];
        vatom[i3][3] += 0.25*v[3];
        vatom[i3][4] += 0.25*v[4];
        vatom[i3][5] += 0.25*v[5];
      }
      if (newton_bond || i4 < nlocal) {
        vatom[i4][0] += 0.25*v[0];
        vatom[i4][1] += 0.25*v[1];
        vatom[i4][2] += 0.25*v[2];
        vatom[i4][3] += 0.25*v[3];
        vatom[i4][4] += 0.25*v[4];
        vatom[i4][5] += 0.25*v[5];
      }
    }
  }

  if (hflag_atom) {
    if (newton_bond || i1 < nlocal) {
      hatom[i1][0] += 0.25*((3*vb1x - 2*vb2x - vb3x) * f1[0]);
      hatom[i1][1] += 0.25*((3*vb1x - 2*vb2x - vb3x) * f1[1]);
      hatom[i1][2] += 0.25*((3*vb1x - 2*vb2x - vb3x) * f1[2]);
      hatom[i1][3] += 0.25*((3*vb1y - 2*vb2y - vb3y) * f1[0]);
      hatom[i1][4] += 0.25*((3*vb1y - 2*vb2y - vb3y) * f1[1]);
      hatom[i1][5] += 0.25*((3*vb1y - 2*vb2y - vb3y) * f1[2]);
      hatom[i1][6] += 0.25*((3*vb1z - 2*vb2z - vb3z) * f1[0]);
      hatom[i1][7] += 0.25*((3*vb1z - 2*vb2z - vb3z) * f1[1]);
      hatom[i1][8] += 0.25*((3*vb1z - 2*vb2z - vb3z) * f1[2]);
    }
    if (newton_bond || i2 < nlocal) {
      hatom[i2][0] += 0.25*((vb1x + 2*vb2x + vb3x) * (f1[0] + f3[0] + f4[0]));
      hatom[i2][1] += 0.25*((vb1x + 2*vb2x + vb3x) * (f1[1] + f3[1] + f4[1]));
      hatom[i2][2] += 0.25*((vb1x + 2*vb2x + vb3x) * (f1[2] + f3[2] + f4[2]));
      hatom[i2][3] += 0.25*((vb1y + 2*vb2y + vb3y) * (f1[0] + f3[0] + f4[0]));
      hatom[i2][4] += 0.25*((vb1y + 2*vb2y + vb3y) * (f1[1] + f3[1] + f4[1]));
      hatom[i2][5] += 0.25*((vb1y + 2*vb2y + vb3y) * (f1[2] + f3[2] + f4[2]));
      hatom[i2][6] += 0.25*((vb1z + 2*vb2z + vb3z) * (f1[0] + f3[0] + f4[0]));
      hatom[i2][7] += 0.25*((vb1z + 2*vb2z + vb3z) * (f1[1] + f3[1] + f4[1]));
      hatom[i2][8] += 0.25*((vb1z + 2*vb2z + vb3z) * (f1[2] + f3[2] + f4[2]));
    }
    if (newton_bond || i3 < nlocal) {
      hatom[i3][0] += 0.25*((-vb1x + 2*vb2x - vb3x) * f3[0]);
      hatom[i3][1] += 0.25*((-vb1x + 2*vb2x - vb3x) * f3[1]);
      hatom[i3][2] += 0.25*((-vb1x + 2*vb2x - vb3x) * f3[2]);
      hatom[i3][3] += 0.25*((-vb1y + 2*vb2y - vb3y) * f3[0]);
      hatom[i3][4] += 0.25*((-vb1y + 2*vb2y - vb3y) * f3[1]);
      hatom[i3][5] += 0.25*((-vb1y + 2*vb2y - vb3y) * f3[2]);
      hatom[i3][6] += 0.25*((-vb1z + 2*vb2z - vb3z) * f3[0]);
      hatom[i3][7] += 0.25*((-vb1z + 2*vb2z - vb3z) * f3[1]);
      hatom[i3][8] += 0.25*((-vb1z + 2*vb2z - vb3z) * f3[2]);
    }
    if (newton_bond || i4 < nlocal) {
      hatom[i4][0] += 0.25*((-vb1x + 2*vb2x + 3*vb3x) * f4[0]);
      hatom[i4][1] += 0.25*((-vb1x + 2*vb2x + 3*vb3x) * f4[1]);
      hatom[i4][2] += 0.25*((-vb1x + 2*vb2x + 3*vb3x) * f4[2]);
      hatom[i4][3] += 0.25*((-vb1y + 2*vb2y + 3*vb3y) * f4[0]);
      hatom[i4][4] += 0.25*((-vb1y + 2*vb2y + 3*vb3y) * f4[1]);
      hatom[i4][5] += 0.25*((-vb1y + 2*vb2y + 3*vb3y) * f4[2]);
      hatom[i4][6] += 0.25*((-vb1z + 2*vb2z + 3*vb3z) * f4[0]);
      hatom[i4][7] += 0.25*((-vb1z + 2*vb2z + 3*vb3z) * f4[1]);
      hatom[i4][8] += 0.25*((-vb1z + 2*vb2z + 3*vb3z) * f4[2]);
    }
  }
}

/* ---------------------------------------------------------------------- */

double Dihedral::memory_usage()
{
  double bytes = comm->nthreads*maxeatom * sizeof(double);
  bytes += comm->nthreads*maxvatom*6 * sizeof(double);
  bytes += comm->nthreads*maxhatom*9 * sizeof(double);
  return bytes;
}
