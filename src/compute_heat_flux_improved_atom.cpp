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
#include "force.h"
#include "error.h"

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

  size_vector = 3;
  vector = new double[size_vector];
}

/* ---------------------------------------------------------------------- */

void ComputeHeatFluxImprovedAtom::init(){}
ComputeHeatFluxImprovedAtom::~ComputeHeatFluxImprovedAtom(){}

/* ---------------------------------------------------------------------- */

void ComputeHeatFluxImprovedAtom::compute_vector()
{
  double heatflux[2];

  int nlocal = atom->nlocal;
  int ntotal = nlocal;
  if (force->newton) ntotal += atom->nghost;

  // sum up angle forces
  double **hatom = force->angle->hatom;
  for (int i = 0; i < ntotal; i++)
    for (int j = 0; j < 3; j++)
      heatflux[j] += hatom[i][j];

  MPI_Allreduce(heatflux,vector,size_vector,MPI_DOUBLE,MPI_SUM,world);
}
