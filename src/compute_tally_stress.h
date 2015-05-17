/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(tally/stress,ComputeTallyStress)

#else

#ifndef LMP_COMPUTE_TALLYSTRESS_H
#define LMP_COMPUTE_TALLYSTRESS_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeTallyStress : public Compute {
 public:
  ComputeTallyStress(class LAMMPS *, int, char **);
  virtual ~ComputeTallyStress();

  void init();
  void setup() { did_compute = -1;}
  
  double compute_scalar() { return 0.0; }
  void compute_vector() {}
  void compute_peratom() {}
  
  void pair_tally_callback(int, int, int, int,
                           double, double, double,
                           double, double, double);

 private:
  bigint did_compute;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
