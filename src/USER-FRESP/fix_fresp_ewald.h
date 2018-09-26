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
   Contributing author: Francesco Cappelluti (francesco.cappelluti@graduate.univaq.it)
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
  void pre_force(int); //charges are updated in the new geometry and forces (in real space) due to charge variation are added
  void setup_pre_force(int); //charges are set according to starting geometry
  void pre_reverse(int, int); //forces due to charge variation are added
  void post_neighbor(); //after neighbor list are reconstructed, bond Verlet lists are scaled (if necessary) and cleared
  double memory_usage();
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);

 protected:
  void q_update_Efield_bond();

 private:
  void ewald_allocate();
  void ewald_deallocate();
  void ewald_init(); //initialize Ewald sums for long range Efield calculation
  void ewald_setup(); //setup of Ewald sums for long range Efield calculation
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
