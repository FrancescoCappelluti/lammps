/* -*- c++ -*- ----------------------------------------------------------
  Fix for a fluctuating charge model written by D. Ottaviani and
  F. Cappelluti from University of L'Aquila (Italy) - 2018
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
  virtual void pre_force(int) = 0; //charges are updated in the new geometry and forces (in real space) due to charge variation are added
  virtual void setup_pre_force(int) = 0; //charges are set according to starting geometry
  virtual void pre_reverse(int, int) = 0; //forces due to charge variation are added
  void setup_pre_reverse(int, int);
  virtual void post_neighbor() = 0; //after neighbor list are reconstructed, bond Verlet lists are scaled (if necessary) and cleared
  void init_list(int, class NeighList *); //added as in fix_qeq in order to use Verlet lists
  void init(); //added in order to initialize list
  virtual double memory_usage() = 0;
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  virtual int pack_reverse_comm(int, int, double *) = 0;
  virtual void unpack_reverse_comm(int, int *, double *) = 0;

 protected:
  //flags that tell if FRESP is active
  bool bondflag, angleflag, dihedralflag, improperflag, Efieldflag;
  int natypes; //# of FRESP atom types
  void read_file(char*);
  void read_file_types(char*);
  int nevery; //charges will be updated each nevery steps
  double cutoff1, cutoff2, cutoff3; //cutoffs in real space for electric field calculation
                                    //if distance is less than cutoff1, Efield is 0
                                    //if distance is between cutoff1 and cutoff2, Efield is damped
                                    //if distance is between cutoff2 and cutoff3, Efield is not damped
                                    //else, no contribution in real space
  double *q0, *qgen;
  //index 1 is type of first atom of the bond, index 2 is type of second atom and index 3 is type of center
  double ***k_bond; 
  //index 1 is type of first atom of the angle, index 2 is type of second atom, index 3 is type of third atom and index 4 is type of center
  double ****k_angle; 
  //index 1 is type of first atom of the dihedral, index 2 is type of second atom, index 3 is type of third atom, index 4 is type of fourth atom and index 5 is type of center
  double *****k_dihedral;
  //index 1 is type of first atom of the improper, index 2 is type of second atom, index 3 is type of third atom, index 4 is type of fourth atom and index 5 is type of center
  double *****k_improper;
  //index 1 is type of first atom of the bond, index 2 is type of second atom and index 3 is type of center
  double ***k_Efield; 
  bigint **mol_map;
  bigint nmolecules; //Number of molecules
  int *types;
  double *deltaq, **deltaf;
  void q_update_angle();
  void q_update_dihedral();
  void q_update_improper();
  virtual void q_update_Efield_bond() = 0;
  class NeighList *list; //added as in fix_qeq in order to use Verlet lists
  int nbond_old;
  //index 1 is bond index in bondlist, index 2 is atom index in Verlet list union for middle bond point and index 3 is the component of the vector derivative
  double ***dEr_vals;
  //index 1 is bond index in bondlist, index 2 is atom index in Verlet list union for middle bond point and index 3 is 0 for distance between center and atom 1 and 1 for distance between center and atom2
  double ***distances;
  //index 1 is bond index in bondlist, index 2 is atom index in Verlet list union for middle bond point
  tagint ***dEr_indexes;
  int **bond_extremes_pos;
  double *erfc_erf_arr; //array where sum of erfcs and erfs between atom and each other in its Verlet list is stored
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
  double average_mol_size; //needed for memory_usage function
  //virtual double Efield_damping(int, double, double, double) = 0;
  void build_erfc_erf_arr();
  void build_bond_Verlet_list(int, tagint, tagint);
  double ***appo2, **appo3; //arrays used by kspace force corrections calculation
  char *id_pe;
  int pack_flag;
  int count_total_bonds();
  short *already_cycled;
  class Compute *pe; // PE compute pointer
  double **kvecs; //vectors in k-space
  double *bondvskprod_vec, *xmkprod_vec, *Im_xm_vec, *Re_xm_vec, *tmp1;
  double *appo2Re_pref_vec, *appo2Im_pref_vec, *Im_prod_vec, *Re_prod_vec, *tmp2;
};

}

#endif
