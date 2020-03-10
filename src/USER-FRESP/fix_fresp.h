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

#ifndef LMP_FIX_FRESP_H
#define LMP_FIX_FRESP_H

#include "fix.h"

namespace LAMMPS_NS {

class FixFResp : public Fix {
 public:
  FixFResp(class LAMMPS *, int, char **);
  virtual ~FixFResp() = 0;
  int setmask();
  //Charges are updated in the new geometry and Efield projections derivatives 
  //are calculated
  virtual void pre_force(int) = 0;
  void min_pre_force(int);
  //Charges are set according to starting geometry
  virtual void setup_pre_force(int) = 0;
  //Forces due to charge variation are added 
  virtual void pre_reverse(int, int) = 0;
  void min_pre_reverse(int, int);
  void setup_pre_reverse(int, int);
  //After neighbor list are reconstructed, bond Verlet lists are scaled
  //(if necessary) and cleared
  virtual void post_neighbor() = 0;
  void min_post_neighbor();
  void init_list(int, class NeighList *);
  void init();
  virtual double memory_usage() = 0;
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);

 protected:
  //Flags that tell if FRESP is active
  bool bondflag, angleflag, dihedralflag, improperflag, phi0improperflag;
  bool  Efieldflag;
  int natypes; //# of FRESP atom types
  void read_file(char*);
  void read_file_types(char*);
  int nevery; //Charges will be updated each nevery steps
  //Cutoffs in real space for electric field calculation. If distance is less
  //than cutoff1, Efield is 0. If distance is between cutoff1 and cutoff2,
  //Efield is damped. If distance is between cutoff2 and cutoff3, Efield is
  //not damped.
  double cutoff1, cutoff2, cutoff3;                  
  double *q0, *qgen;
  //Index 1 is type of first atom of the bond, index 2 is type of second atom
  //and index 3 is type of center
  double ***k_bond; 
  //Index 1 is type of first atom of the angle, index 2 is type of second atom,
  //index 3 is type of third atom and index 4 is type of center
  double ****k_angle; 
  //Index 1 is type of first atom of the dihedral, index 2 is type of second
  //atom, index 3 is type of third atom, index 4 is type of fourth atom and
  //index 5 is type of center
  double *****k_dihedral;
  //Index 1 is type of first atom of the improper, index 2 is type of second
  //atom, index 3 is type of third atom, index 4 is type of fourth atom and
  //index 5 is type of center
  double *****k_improper;
  //Index 1 is type of first atom of the improper, index 2 is type of second
  //atom, index 3 is type of third atom and index 4 is type of fourth atom
  double ****phi0_improper;
  //Index 1 is type of first atom of the bond, index 2 is type of second atom
  //and index 3 is type of center
  double ***k_Efield; 
  //Arrays containing the values of the partial scaling coefficients needed
  //by Thole damping function
  double *thoascal;
  double **thobscal;
  bigint **mol_map;
  bigint nmolecules; //Number of molecules
  int *types;
  double *deltaq, **deltaf;
  void q_update_angle();
  void q_update_dihedral();
  void q_update_improper();
  virtual void q_update_Efield_bond() = 0;
  class NeighList *list;
  int nbond_old, nangle_old, nimproper_old;
  //Index 1 is bond index according to nbond_old, index 2 is atom index in
  //Verlet list union for middle bond point and index 3 is the component of
  //the vector derivative
  double ***dEr_vals;
  //Index 1 is bond index in bondlist, index 2 is atom index in Verlet list
  //union for middle bond point
  tagint ***dEr_indexes;
  //Index 1 is bond index in neighbor->bondlist, index 2 is the component
  //of the vector derivative
  double **db_vals;
  //Index 1 is angle index in neighbor->anglelist, index 2 is is the
  //index of the atom in the angle definition (going therefore from 0 to 2)
  //and index 3 is the component of the vector derivative
  double ***da_vals;
  //Index 1 is improper index in neighbor->improperlist, index 2 is is the
  //index of the atom in the improper definition (going therefore from 0 to 3)
  //and index 3 is the component of the vector derivative
  double ***dimp_vals;
  int **bond_extremes_pos;
  double q2, qsum, qsqsum, scale, triclinic, accuracy, g_ewald;
  double unitk[3]; 
  int kxmax,kymax,kzmax;
  int kcount,kmax,kmax3d,kmax_created;
  double gsqmx,volume;
  int nmax;
  int *kxvecs,*kyvecs,*kzvecs;
  int kxmax_orig,kymax_orig,kzmax_orig;
  bigint natoms_original;
  int warn_nonneutral;           // 0 = error if non-neutral system
                                 // 1 = warn once if non-neutral system
                                 // 2 = warn, but already warned
  int warn_nocharge;             // 0 = already warned
                                 // 1 = warn if zero charge
  int gewaldflag, kewaldflag;
  int dampflag;
  int printEfieldflag;
  double *ug;
  double **eg,**vg;
  double **ek;
  double *sfacrl_qgen,*sfacim_qgen,*sfacrl_all_qgen,*sfacim_all_qgen;
  double ***cs,***sn;
  double ***cs_qgen,***sn_qgen;
  double average_mol_size; //Needed for memory_usage function
  //Virtual double Efield_damping(int, double, double, double) = 0;
  void build_bond_Verlet_list(int, tagint, tagint);
  double ***appo2, **appo3; //Arrays used by kspace force correction calculation
  char *id_pe;
  int pack_flag;
  int count_total_bonds();
  class Compute *pe; // PE compute pointer
  double **kvecs; //Vectors in k-space
  double *bondvskprod_vec, *xmkprod_vec, *Im_xm_vec, *Re_xm_vec, *tmp1, *tmp2;
  double *appo2Re_pref_vec, *appo2Im_pref_vec, *Im_prod_vec, *Re_prod_vec;
  void deltaq_update_Efield(bigint molecule, int atom1_t, int atom2_t, \
    double Eparallel);
  void deltaq_update_bond(bigint molecule, int atom1_t, int atom2_t, \
    double dr);
  void deltaq_update_angle(bigint molecule, int atom1_t, int atom2_t, \
  int atom3_t, double da);
  void deltaq_update_improper(bigint molecule, int atom1_t, int atom2_t, \
  int atom3_t, int atom4_t, double absim);
  double beta; //Needed by fix_fresp_mdsf
  enum damp {NONE, EXP, SIN, THO} dampflg;
  inline int sbmask(int j) const {
    return j >> SBBITS & 3;
  }
  double SIXTH = 1. / 6.;
};

}

#endif
