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
  void pre_force(int); //charges are updated in the new geometry and forces (in real space) due to charge variation are added
  void setup_pre_force(int); //charges are set according to starting geometry
  void pre_reverse(int, int); //forces due to charge variation are added
  void post_neighbor(); //after neighbor list are reconstructed, bond Verlet lists are scaled (if necessary) and cleared
  double memory_usage();
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);

 protected:
  void q_update_Efield_bond();
  double Efield_damping(double, double *);
};

}

#endif
#endif
