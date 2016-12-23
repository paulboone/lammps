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
    memory->create(hf_atom,nmax,9,"heat/flux_improved_atom:hf_atom");
    array_atom = hf_atom;
  }

  int nlocal = atom->nlocal;
  int ntotal = nlocal;
  if (force->newton) ntotal += atom->nghost;

  for (i = 0; i < ntotal; i++)
    for (j = 0; j < 9; j++)
      hf_atom[i][j] = 0.0;

  // sum up angle forces
  hatom = force->angle->hatom;
  for (i = 0; i < ntotal; i++)
    for (j = 0; j < 9; j++)
      hf_atom[i][j] += hatom[i][j];

  // for (i=0; i<ntotal; i++)
  //   std::cout << "hf_atom[" << i << "]: " << hf_atom[i][0] << "/" << hf_atom[i][1] << "/" << hf_atom[i][2] << "\n";

  // heatflux[0] = 0.0;
  // heatflux[1] = 0.0;
  // heatflux[2] = 0.0;
  // for (i = 0; i < nlocal; i++)
  //   for (j = 0; j < 3; j++)
  //     heatflux[j] += hf_atom[i][j];
  //
  // // std::cout << force->angle->ntmp0 << ", " << force->angle->ntmp1 << ", "  << force->angle->ntmp2 << ", "  << force->angle->ntmp3 << "\n";
  // std::cout << heatflux[0] << ", " << heatflux[1] << ", " << heatflux[2] << "\n";


  // communicate ghost fluxes between neighbor procs
  if (force->newton)
    comm->reverse_comm_compute(this);

  // for (i=0; i<ntotal; i++)
  //   std::cout << "hf_atom[" << i << "]: " << hf_atom[i][0] << "/" << hf_atom[i][1] << "/" << hf_atom[i][2] << "\n";
  // for (i=0; i<ntotal; i++)
  //   std::cout << "hatom[" << i << "]: " << hatom[i][0] << "/" << hatom[i][1] << "/" << hatom[i][2] << "\n";


  // double heatflu2[3];
  heatflux[0] = 0.0;
  heatflux[1] = 0.0;
  heatflux[2] = 0.0;
  // heatflu2[0] = 0.0;
  // heatflu2[1] = 0.0;
  // heatflu2[2] = 0.0;

  double **v = atom->v;


  for (i = 0; i < nlocal; i++) {
    // std::cout << "CHECKS: " << hatom[i][3] - hf_atom[i][3] << " _ "  << hatom[i][4] - hf_atom[i][4] << " _ "  << hatom[i][5] - hf_atom[i][5] << "\n";
    heatflux[0] += hf_atom[i][0] * v[i][0] + hf_atom[i][1] * v[i][1] + hf_atom[i][2] * v[i][2];
    heatflux[1] += hf_atom[i][3] * v[i][0] + hf_atom[i][4] * v[i][1] + hf_atom[i][5] * v[i][2];
    heatflux[2] += hf_atom[i][6] * v[i][0] + hf_atom[i][7] * v[i][1] + hf_atom[i][8] * v[i][2];
    // heatflu2[0] += hatom[i][0] * v[i][0] + hatom[i][1] * v[i][1] + hatom[i][2] * v[i][2];
    // heatflu2[1] += hatom[i][3] * v[i][0] + hatom[i][4] * v[i][1] + hatom[i][5] * v[i][2];
    // heatflu2[2] += hatom[i][6] * v[i][0] + hatom[i][7] * v[i][1] + hatom[i][8] * v[i][2];
  }

  // std::cout << "heatflux: " << heatflux[0] << "/" << heatflux[1] << "/" << heatflux[2] << "\n";
  // std::cout << "heatflu2: " << heatflu2[0] << "/" << heatflu2[1] << "/" << heatflu2[2] << "\n";
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
    std::cout << "UNPACK BEF: " << j << " " << hf_atom[j][0] << "/" << hf_atom[j][1] << "/" << hf_atom[j][2] << "\n";
    hf_atom[j][0] += buf[m++];
    hf_atom[j][1] += buf[m++];
    hf_atom[j][2] += buf[m++];
    hf_atom[j][3] += buf[m++];
    hf_atom[j][4] += buf[m++];
    hf_atom[j][5] += buf[m++];
    hf_atom[j][6] += buf[m++];
    hf_atom[j][7] += buf[m++];
    hf_atom[j][8] += buf[m++];
    std::cout << "UNPACK AFT: " << j << " " << hf_atom[j][0] << "/" << hf_atom[j][1] << "/" << hf_atom[j][2] << "\n";
  }
  std::cout << "UNPACK INTO NONGHOST: " << n << "\n";
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeHeatFluxImprovedAtom::memory_usage()
{
  double bytes = nmax*3 * sizeof(double);
  return bytes;
}
