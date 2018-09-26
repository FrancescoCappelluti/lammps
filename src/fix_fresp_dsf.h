/* -*- c++ -*- ----------------------------------------------------------
  Fix for a fluctuating charge model written by D. Ottaviani and
  F. Cappelluti from University of L'Aquila (Italy) - 2018
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
