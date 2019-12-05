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

/* ----------------------------------------------------------------------
   Contributing author: Francesco Cappelluti
     (francesco.cappelluti@graduate.univaq.it)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(fresp/ewald,FixFRespEwald)

#else

#ifndef LMP_FIX_FRESP_EWALD_H
#define LMP_FIX_FRESP_EWALD_H

#include "fix_fresp.h"

namespace LAMMPS_NS {

class FixFRespEwald : public FixFResp {
 public:
  FixFRespEwald(class LAMMPS *, int, char **);
  ~FixFRespEwald();
  void pre_force(int);
  void setup_pre_force(int);
  void pre_reverse(int, int);
  void post_neighbor();
  double memory_usage();

 protected:
  void q_update_Efield_bond();

 private:
  void ewald_allocate();
  void ewald_deallocate();
  void ewald_init(); //Initialize Ewald sums for long range Efield calculation
  void ewald_setup(); //Setup of Ewald sums for long range Efield calculation
  void ewald_coeffs();
  void ewald_qsum_qsq();
  void ewald_coeffs_triclinic();
  void ewald_eik_dot_r_qgen();
  void ewald_eik_dot_r_triclinic_qgen();
  void ewald_structure_factor();
  double ewald_rms(int, double, bigint, double);
  void ewald_lamda2xT(double *, double *);
  void ewald_x2lamdaT(double *, double *);
  double Efield_damping(int, double, double, double);
};

}

#endif
#endif
