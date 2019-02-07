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

FixStyle(fresp/mdsf,FixFRespMDsf)

#else

#ifndef LMP_FIX_FRESP_MDSF_H
#define LMP_FIX_FRESP_MDSF_H

#include "fix_fresp_dsf.h"

namespace LAMMPS_NS {

class FixFRespMDsf : public FixFRespDsf {
 public:
  FixFRespMDsf(class LAMMPS *, int, char **);
  ~FixFRespMDsf();

 protected:
  void q_update_Efield_bond();
};

}

#endif
#endif
