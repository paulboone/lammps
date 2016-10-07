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

#include "compute_heat_flux_torii.h"
#include "atom.h"
#include "angle.h"
#include "update.h"
#include "force.h"
#include "domain.h"
#include "group.h"
#include "error.h"

#include <iostream>
#include <stdio.h>


using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeHeatFluxTorii::ComputeHeatFluxTorii(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute ke command");

  vector_flag = 1;
  extvector = 1;

  size_vector = 3;
  vector = new double[size_vector];
}

/* ---------------------------------------------------------------------- */

void ComputeHeatFluxTorii::init()
{

}

ComputeHeatFluxTorii::~ComputeHeatFluxTorii()
{
  // delete [] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeHeatFluxTorii::compute_vector()
{
  double total[3];
  double **hatom = force->angle->hatom;
  int i,n;

  total[0] = 0.0;
  total[1] = 0.0;
  total[2] = 0.0;

  n = atom->nlocal;
  for (i=0; i<n; i++) {
    total[0] += hatom[i][0];
    total[1] += hatom[i][1];
    total[2] += hatom[i][2];
  }
  // std::cout << "HF_TORII total: " << total[0] << ", " << total[1] << ", " << total[2] << "\n";
  MPI_Allreduce(&total,vector,size_vector,MPI_DOUBLE,MPI_SUM,world);
  // std::cout << "HF_TORII scalar: " << vector[0] << ", " << vector[1] << ", " << vector[2] << ", " << "\n";
}
