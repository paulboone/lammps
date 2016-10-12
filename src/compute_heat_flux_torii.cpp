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
#include "angle.h"
#include "bond.h"
#include "force.h"
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
  extvector = 0;

  size_vector = 3;
  vector = new double[size_vector];
}

/* ---------------------------------------------------------------------- */

void ComputeHeatFluxTorii::init(){}
ComputeHeatFluxTorii::~ComputeHeatFluxTorii(){}

/* ---------------------------------------------------------------------- */

void ComputeHeatFluxTorii::compute_vector()
{
//  double * heatflux = force->angle->heatflux_angle;
  double * heatflux = force->bond->heatflux_bond;
  // std::cout << "HF_TORII total: " << total[0] << ", " << total[1] << ", " << total[2] << "\n";
  MPI_Allreduce(heatflux,vector,size_vector,MPI_DOUBLE,MPI_SUM,world);
  // std::cout << "HF_TORII scalar: " << vector[0] << ", " << vector[1] << ", " << vector[2] << ", " << "\n";
}
