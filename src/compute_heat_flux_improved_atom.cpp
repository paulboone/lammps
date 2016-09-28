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


#include <string.h>
#include "compute_heat_flux_improved_atom.h"
#include "atom.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "bond.h"
#include "comm.h"
#include "force.h"
#include "error.h"
#include "memory.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeHeatFluxImprovedAtom::ComputeHeatFluxImprovedAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR,"Illegal heat/flux_improved_atom command");

  peratom_flag = 1;
  size_peratom_cols = 9;
  comm_reverse = 9;
  vector_flag = 1;
  extvector = 1;

  nmax = 0;
  hf_atom = NULL;

  size_vector = 3;
  vector = new double[size_vector];

  // process optional arguments
  if (narg == 3) {
    angleflag = dihedralflag = improperflag = 1;
  } else {
    angleflag = dihedralflag = improperflag = 0;
    int iarg = 3;
    while (iarg < narg) {
      if (strcmp(arg[iarg],"angle") == 0) angleflag = 1;
      else if (strcmp(arg[iarg],"dihedral") == 0) dihedralflag = 1;
      else if (strcmp(arg[iarg],"improper") == 0) improperflag = 1;
      else error->all(FLERR,"Illegal heat/flux_improved_atom command");
      iarg++;
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeHeatFluxImprovedAtom::init(){}
ComputeHeatFluxImprovedAtom::~ComputeHeatFluxImprovedAtom(){
  memory->destroy(hf_atom);
}

/* ---------------------------------------------------------------------- */

void ComputeHeatFluxImprovedAtom::compute_vector()
{
  double **hatom;
  int i, j;

  if (atom->nmax > nmax) {
    memory->destroy(hf_atom);
    nmax = atom->nmax;
    memory->create(hf_atom,nmax,9,"heat/flux_improved_atom:hf_atom");
    array_atom = hf_atom;
  }

  int nlocal = atom->nlocal;
  int ntotal = nlocal;
  if (force->newton) ntotal += atom->nghost;

  // zero out hf_atom array
  for (i = 0; i < ntotal; i++)
    for (j = 0; j < 9; j++)
      hf_atom[i][j] = 0.0;

  // sum up per-term atomic heat fluxes
  if (angleflag && force->angle) {
    hatom = force->angle->hatom;
    for (i = 0; i < ntotal; i++)
      for (j = 0; j < 9; j++)
        hf_atom[i][j] += hatom[i][j];
  }

  if (dihedralflag && force->dihedral) {
    hatom = force->dihedral->hatom;
    for (i = 0; i < ntotal; i++)
      for (j = 0; j < 9; j++)
        hf_atom[i][j] += hatom[i][j];
  }

  if (improperflag && force->improper) {
    hatom = force->improper->hatom;
    for (i = 0; i < ntotal; i++)
      for (j = 0; j < 9; j++)
        hf_atom[i][j] += hatom[i][j];
  }


  // communicate ghost fluxes between neighbor procs
  if (force->newton)
    comm->reverse_comm_compute(this);

  double **v = atom->v;
  double heatflux[3] = {0.0,0.0,0.0};
  int *mask = atom->mask;

  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      heatflux[0] += hf_atom[i][0] * v[i][0] + hf_atom[i][1] * v[i][1] + hf_atom[i][2] * v[i][2];
      heatflux[1] += hf_atom[i][3] * v[i][0] + hf_atom[i][4] * v[i][1] + hf_atom[i][5] * v[i][2];
      heatflux[2] += hf_atom[i][6] * v[i][0] + hf_atom[i][7] * v[i][1] + hf_atom[i][8] * v[i][2];
     }
  }

  MPI_Allreduce(heatflux,vector,size_vector,MPI_DOUBLE,MPI_SUM,world);
}

/* ---------------------------------------------------------------------- */

int ComputeHeatFluxImprovedAtom::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = hf_atom[i][0];
    buf[m++] = hf_atom[i][1];
    buf[m++] = hf_atom[i][2];
    buf[m++] = hf_atom[i][3];
    buf[m++] = hf_atom[i][4];
    buf[m++] = hf_atom[i][5];
    buf[m++] = hf_atom[i][6];
    buf[m++] = hf_atom[i][7];
    buf[m++] = hf_atom[i][8];
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeHeatFluxImprovedAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;
  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    hf_atom[j][0] += buf[m++];
    hf_atom[j][1] += buf[m++];
    hf_atom[j][2] += buf[m++];
    hf_atom[j][3] += buf[m++];
    hf_atom[j][4] += buf[m++];
    hf_atom[j][5] += buf[m++];
    hf_atom[j][6] += buf[m++];
    hf_atom[j][7] += buf[m++];
    hf_atom[j][8] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeHeatFluxImprovedAtom::memory_usage()
{
  double bytes = nmax*9 * sizeof(double);
  return bytes;
}
