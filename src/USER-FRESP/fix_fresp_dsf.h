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

FixStyle(fresp/dsf,FixFRespDsf)

#else

#ifndef LMP_FIX_FRESP_DSF_H
#define LMP_FIX_FRESP_DSF_H

#include "fix_fresp.h"

namespace LAMMPS_NS {

class FixFRespDsf : public FixFResp {
 public:
  FixFRespDsf(class LAMMPS *, int, char **);
  ~FixFRespDsf();
  void pre_force(int);
  void setup_pre_force(int);
  void pre_reverse(int, int);
  void post_neighbor();
  double memory_usage();

 protected:
  void q_update_Efield_bond();
  double Efield_damping(double, double *, int);
  void force_update_Efield_bond();
  void force_update_angle();
  void force_update_improper();
};

}

#endif
#endif
