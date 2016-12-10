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

#include "compute_heat_flux_improved_atom.h"
#include "atom.h"
#include "angle.h"
#include "bond.h"
#include "comm.h"
#include "force.h"
#include "error.h"
#include "memory.h"

#include <iostream>
#include <stdio.h>


using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeHeatFluxImprovedAtom::ComputeHeatFluxImprovedAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute ke command");

  vector_flag = 1;
  extvector = 0;

  nmax = 0;
  hf_atom = NULL;

  size_vector = 3;
  vector = new double[size_vector];
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
  double heatflux[3];
  int i, j;

  if (atom->nmax > nmax) {
    memory->destroy(hf_atom);
    nmax = atom->nmax;
    memory->create(hf_atom,nmax,3,"heat/flux_improved_atom:hf_atom");
    array_atom = hf_atom;
  }

  int nlocal = atom->nlocal;
  int ntotal = nlocal;
  if (force->newton) ntotal += atom->nghost;

  // sum up angle forces
  hatom = force->angle->hatom;
  for (i = 0; i < ntotal; i++)
    for (j = 0; j < 3; j++)
      hf_atom[i][j] = hatom[i][j];

  heatflux[0] = 0.0;
  heatflux[1] = 0.0;
  heatflux[2] = 0.0;
  for (i = 0; i < nlocal; i++)
    for (j = 0; j < 3; j++)
      heatflux[j] += hf_atom[i][j];

  // std::cout << force->angle->ntmp0 << ", " << force->angle->ntmp1 << ", "  << force->angle->ntmp2 << ", "  << force->angle->ntmp3 << "\n";
  std::cout << heatflux[0] << ", " << heatflux[1] << ", " << heatflux[2] << "\n";


  // communicate ghost fluxes between neighbor procs
  if (force->newton)
    comm->reverse_comm_compute(this);


  heatflux[0] = 0.0;
  heatflux[1] = 0.0;
  heatflux[2] = 0.0;
  for (i = 0; i < nlocal; i++)
    for (j = 0; j < 3; j++)
      heatflux[j] += hf_atom[i][j];
  std::cout << heatflux[0] << ", " << heatflux[1] << ", " << heatflux[2] << "\n";


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
  }

  std::cout << "PACK: " << first << " to " << last << "\n";
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeHeatFluxImprovedAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;
  int tcnt = 0;
  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    if (j < 3584) tcnt++;
    hf_atom[j][0] += buf[m++];
    hf_atom[j][1] += buf[m++];
    hf_atom[j][2] += buf[m++];
  }
  std::cout << "UNPACK INTO NONGHOST: " << tcnt << "\n";
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeHeatFluxImprovedAtom::memory_usage()
{
  double bytes = nmax*3 * sizeof(double);
  return bytes;
}
