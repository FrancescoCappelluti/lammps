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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mkl.h> //Not mandatory
#include "fix_fresp_dsf.h"
#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "improper.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "force.h"
#include "math_special.h"
#include "math_const.h"
#include "memory.h"
#include "pair.h"
#include "modify.h"
#include "error.h"
#include "math_extra.h"
#include "kspace.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define MAXLINE 1024
#define SMALL     0.001

enum damp {NONE, EXP, SIN} dampflg;

/* ---------------------------------------------------------------------- */

FixFRespDsf::FixFRespDsf(LAMMPS *lmp, int narg, char **arg) :
  FixFResp(lmp, narg, arg)
{
  thermo_virial = 1; //Enables virial contribution
  qsqsum = 1.0; //Declared here because not accessible from force->kspace
  dampflag = NONE;

  if (narg < 8 || narg > 13) error->all(FLERR,"Illegal fix fresp command");

  g_ewald = force->numeric(FLERR, arg[5]);
  
  int iarg = 8;
  
  //else if are needed in order not to have segfault when trying to access
  //elements outside arg
  while (iarg < narg) {
    if ((arg[iarg] - strchr(arg[iarg], '#')) == 0) break;
    else if (strcmp(arg[iarg], "damp") == 0) {
      if (strcmp(arg[++iarg], "exp") == 0) dampflag = EXP;
      else if (strcmp(arg[iarg], "sin") == 0) dampflag = SIN;
      cutoff1 = force->numeric(FLERR, arg[++iarg]);
      cutoff2 = force->numeric(FLERR, arg[++iarg]);
      iarg++;
    }
    else if (strcmp(arg[iarg++], "printEfield") == 0) {
      printEfieldflag = 1;
    }
  }

  //check for sane arguments
  if ((nevery <= 0) || (cutoff1 < 0.0 || cutoff2 < 0.0 || cutoff3 <= 0.0))
    error->all(FLERR,"Illegal fix fresp command");

  nmax = 0;

  //read FRESP types file
  read_file_types(arg[6]);

  //create an array where q0 is associated with atom global indexes
  memory->create(q0, natypes, "fresp:q0");
  
  //create an array where qgen is associated with atom global indexes
  memory->create(qgen, natypes, "fresp:qgen");

  //read FRESP parameters file
  read_file(arg[7]);
}

/* ---------------------------------------------------------------------- */

FixFRespDsf::~FixFRespDsf()
{
}

/* ---------------------------------------------------------------------- 
   charges fluctuation due to electric field on bonds
------------------------------------------------------------------------ */

void FixFRespDsf::q_update_Efield_bond()
{
  double xm[3], **x = atom->x, k, rvml, rvminv, rvminvsq, rvm[3], r0, dr;
  double  bondvl, bondvinv, bondvinvsq, pref, q_gen; 
  bigint atom1, atom2, center, global_center, global_atom1, global_atom2;
  bigint molecule;
  int atom1_t, atom2_t, center_t, i, iplusone, bond, atom1_pos, atom2_pos;
  int molflag;
  double bondv[3], E[3], Efield[3], Eparallel, erfc, rvmlsq, partialerfc;
  double grij, expm2, first_scalar_part, second_scalar_part, third_scalar_part;
  double minus_grijsq, ddamping[3];
  double first_vector_part[3], second_vector_part[3], bondrvmprod, damping;
  static double tgeospi = 2.0 * g_ewald / MathConst::MY_PIS;
  static double cutoff3sq = cutoff3 * cutoff3, cutoff1sq = cutoff1 * cutoff1;
  static double tgecuospi = tgeospi * g_ewald * g_ewald;
  static double f_shift = MathSpecial::expmsq(g_ewald * cutoff3) *
    ((MathSpecial::my_erfcx(g_ewald * cutoff3) / cutoff3) + tgeospi) / cutoff3;

  //This cycle over all the atoms is absolutely needed
  if (Efieldflag && qsqsum > 0.0) for (bond = 0; bond < nbond_old; bond++) {
    for (i = 0; i < dEr_indexes[bond][0][0]; i++) dEr_vals[bond][i][0] =
      dEr_vals[bond][i][1] = dEr_vals[bond][i][2] = 0.0;
  }

  for (bond = 0; bond < nbond_old; bond++) {
    //E is initialized as {0., 0., 0.}
    MathExtra::zero3(E);
    Eparallel = 0.0;

    atom1 = dEr_indexes[bond][0][1];
    atom2 = dEr_indexes[bond][0][2];
    global_atom1 = atom->tag[atom1];
    global_atom2 = atom->tag[atom2];
    molecule = atom->molecule[atom1];
    atom1_t = types[global_atom1 - 1];
    atom2_t = types[global_atom2 - 1];
   
    //Declaring bondv as xb1 - xb2 makes it as H->O for water
    bondv[0] = x[atom1][0] - x[atom2][0];
    bondv[1] = x[atom1][1] - x[atom2][1];
    bondv[2] = x[atom1][2] - x[atom2][2];
    bondvl = MathExtra::len3(bondv);

    //This check is here because, if false, bondv has already been calculated
    //and can be used for charge variation due to bond stretching
    //qsqsum is that declared in constructor, need to correct
    if (Efieldflag && qsqsum > 0.0) {
      bondvinv = 1.0 / bondvl;
      bondvinvsq = bondvinv * bondvinv;
      xm[0] = (x[atom1][0] + x[atom2][0]) * 0.5;
      xm[1] = (x[atom1][1] + x[atom2][1]) * 0.5;
      xm[2] = (x[atom1][2] + x[atom2][2]) * 0.5;
      domain->minimum_image(xm[0], xm[1], xm[2]);

      atom1_pos = bond_extremes_pos[bond][0];
      atom2_pos = bond_extremes_pos[bond][1];

      //The cycle is done over all the atoms contained 
      //in the Verlet list of bond
      for (i = 0; i < dEr_indexes[bond][0][0]; i++) {
        iplusone = i + 1;
        center = dEr_indexes[bond][iplusone][0];
        global_center = atom->tag[center];
        //molflag = 1 if center is in the same molecule, 0 otherwise        
        molflag = atom->molecule[center] == molecule;
        rvm[0] = xm[0] - x[center][0];
        rvm[1] = xm[1] - x[center][1];
        rvm[2] = xm[2] - x[center][2];
        domain->minimum_image(rvm[0], rvm[1], rvm[2]);
        rvmlsq = MathExtra::lensq3(rvm);

        dEr_indexes[bond][iplusone][1] = (tagint)1;

        if (rvmlsq > cutoff3sq || molflag) {
          //In order not to cycle over this atom in pre_reverse function
          if (i != atom1_pos && i != atom2_pos)
            dEr_indexes[bond][iplusone][1] = (tagint)-1;
          continue;
        }

        if (rvmlsq < cutoff1sq && dampflag == SIN) continue;
        rvml = sqrt(rvmlsq);
        rvminv = 1.0 / rvml;
        rvminvsq = rvminv * rvminv;
        MathExtra::copy3(rvm, Efield);
        grij = g_ewald * rvml;
        q_gen = qgen[types[global_center - 1]];
        bondrvmprod = MathExtra::dot3(bondv, rvm);
        expm2 = MathSpecial::expmsq(grij);
        partialerfc = MathSpecial::my_erfcx(grij);
        pref = expm2 * rvminv * (partialerfc * rvminv + tgeospi);
        pref -= f_shift;
        pref *= q_gen * rvminv * bondvinv;
        //Now pref is A * q_gen / (|rvm||rb|)
        first_scalar_part = 3.0 * bondrvmprod * rvminvsq;
        third_scalar_part = q_gen * bondvinv * rvminvsq * bondrvmprod *
          (f_shift * rvminv + expm2 * tgecuospi);

        if (rvml < cutoff2 && dampflag > 0) {
          MathExtra::copy3(rvm, ddamping);
          damping = Efield_damping(rvml, ddamping);
          //third_scalar_part is multiplied times damping. Because pref too will
          //be multiplied times damping, the whole Efield derivative is damped.
          third_scalar_part *= damping;
          MathExtra::scale3(bondrvmprod * rvminv * pref, ddamping);
          pref *= damping;
          MathExtra::add3(ddamping, dEr_vals[bond][i], dEr_vals[bond][i]);
          //derivative of damping function for atom1 and atom2 is calculated
          //(simply half the opposite of the previous one) and multiplied times
          //undamped Efield
          MathExtra::scale3(-0.5, ddamping);
          MathExtra::add3(ddamping, dEr_vals[bond][atom1_pos],
            dEr_vals[bond][atom1_pos]);
          MathExtra::add3(ddamping, dEr_vals[bond][atom2_pos],
            dEr_vals[bond][atom2_pos]);
        }

        MathExtra::scale3(pref, Efield);
        MathExtra::add3(E, Efield, E);

        first_vector_part[0] = first_scalar_part * rvm[0] - bondv[0];
        first_vector_part[1] = first_scalar_part * rvm[1] - bondv[1];
        first_vector_part[2] = first_scalar_part * rvm[2] - bondv[2];
        second_vector_part[0] = third_scalar_part * rvm[0];
        second_vector_part[1] = third_scalar_part * rvm[1];
        second_vector_part[2] = third_scalar_part * rvm[2];
      
        dEr_vals[bond][i][0] += pref * first_vector_part[0] + 2.0 *
          second_vector_part[0];
        dEr_vals[bond][i][1] += pref * first_vector_part[1] + 2.0 *
          second_vector_part[1];
        dEr_vals[bond][i][2] += pref * first_vector_part[2] + 2.0 *
          second_vector_part[2];

        first_scalar_part = 0.5 - bondrvmprod * bondvinvsq;
        second_scalar_part = 1.0 - 1.5 * bondrvmprod * rvminvsq;
        first_vector_part[0] = first_scalar_part * bondv[0] +
          second_scalar_part * rvm[0];
        first_vector_part[1] = first_scalar_part * bondv[1] +
          second_scalar_part * rvm[1];
        first_vector_part[2] = first_scalar_part * bondv[2] +
          second_scalar_part * rvm[2];

        dEr_vals[bond][atom1_pos][0] += pref * first_vector_part[0] -
          second_vector_part[0];
        dEr_vals[bond][atom1_pos][1] += pref * first_vector_part[1] -
          second_vector_part[1];
        dEr_vals[bond][atom1_pos][2] += pref * first_vector_part[2] -
          second_vector_part[2];

        first_scalar_part = 1.0 - first_scalar_part;
        second_scalar_part = 2.0 - second_scalar_part;
        first_vector_part[0] = first_scalar_part * bondv[0] -
          second_scalar_part * rvm[0];
        first_vector_part[1] = first_scalar_part * bondv[1] -
          second_scalar_part * rvm[1];
        first_vector_part[2] = first_scalar_part * bondv[2] -
          second_scalar_part * rvm[2];

        dEr_vals[bond][atom2_pos][0] += pref * first_vector_part[0] -
          second_vector_part[0];
        dEr_vals[bond][atom2_pos][1] += pref * first_vector_part[1] -
          second_vector_part[1];
        dEr_vals[bond][atom2_pos][2] += pref * first_vector_part[2] -
          second_vector_part[2];
      }
      Eparallel = MathExtra::dot3(E, bondv);
      if (printEfieldflag) fprintf(stderr, "%i %i %.14lf\n", global_atom1,
        global_atom2, Eparallel);
    }

    if (bondflag) {
      r0 = force->bond->equilibrium_distance(neighbor->bondlist[bond][2]);
      dr = bondvl - r0;
    }

    //The cycle is done over all the atoms contained in the same molecule
    //of the bond
    for (i = 1; i <= mol_map[molecule - 1][0]; i++) {
      global_center = mol_map[molecule - 1][i];
      center = atom->map((int)global_center);
      center_t = types[global_center - 1];
      k = k_Efield[atom1_t][atom2_t][center_t];

      //Charge variation are put in deltaq instead of atom->q in order
      //to permit their communication to other processes
      deltaq[center] += k * Eparallel;

      if (bondflag) {
        k = k_bond[atom1_t][atom2_t][center_t];

        //Charge variation are put in deltaq instead of atom->q in order
        //to permit their communication to other processes
        deltaq[center] += k * dr;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixFRespDsf::setup_pre_force(int vflag)
{
  int i, j, bond = 0;
  bigint atom1;

  //using count_total_bonds(), bonds "freezed" by SHAKE are considered too,
  //otherwise than using nbondlist
  nbond_old = count_total_bonds();

  nmax = atom->nmax;
  memory->create(deltaq, nmax, "fresp:deltaq");

  //an array of nbond double** is allocated in order to store the values of
  //derivatives of E_R * bond unit vector
  //an array of nbond tagint* is allocated in order to store the indexes of
  //atoms wrt the derivatives of dEr_vals are done
  dEr_vals = (double***) calloc(nbond_old, sizeof(double**));
  if (dEr_vals == NULL) {
    char str[128];
    sprintf(str,"Failed to allocate " BIGINT_FORMAT " bytes for array \
      fresp:dEr_vals", nbond_old * sizeof(double**));
    error->one(FLERR,str);
  }
  dEr_indexes = (tagint***) calloc(nbond_old, sizeof(tagint**));
  if (dEr_indexes == NULL) {
    char str[128];
    sprintf(str,"Failed to allocate " BIGINT_FORMAT " bytes for array \
      fresp:dEr_indexes", nbond_old * sizeof(double**));
    error->one(FLERR,str);
  }
  memory->create(bond_extremes_pos, nbond_old, 2, "fresp:bond_extremes_pos");

  for (i = 0; i < nbond_old; i++) {
    dEr_vals[i] = NULL;
    dEr_indexes[i] = NULL;
  }

  //Build new neighbor lists needed by F-RESP
  neighbor->build_one(list);

  for (i = 0; i < atom->nlocal; i++) {
    for (j = 0; j < atom->num_bond[i]; j++) {
      atom1 = atom->map(atom->bond_atom[i][j]);
      atom1 = domain->closest_image(i, (int)atom1);
      if (force->newton_bond || i < atom1)
        build_bond_Verlet_list(bond++, i, atom1);
    }
  }

  pre_force(vflag); 
}
 
/* ---------------------------------------------------------------------- */

void FixFRespDsf::post_neighbor()
{
  int i, j, end, bond = 0;
  bigint atom1;

  //Content of dEr_vals, dEr_indexes and distances arrays is freed.
  //Could find a more efficient way than freeing all these arrays so many times
  //per simulation.
  for (i = 0; i < nbond_old; i++) {
    memory->destroy(dEr_vals[i]);
    dEr_vals[i] = NULL;
    end = dEr_indexes[i][0][0];
    for (j = 0; j <= end; j++) free(dEr_indexes[i][j]);
    memory->sfree(dEr_indexes[i]);
    dEr_indexes[i] = NULL;
  }

  j = count_total_bonds();

  //dEr_vals, dEr_indexes and distances arrays are deallocated, reallocated and
  //initialized as pointing to NULL.
  if (nbond_old != j) {
    nbond_old = j;
    memory->destroy(bond_extremes_pos);
    free(dEr_vals);
    free(dEr_indexes);
    dEr_vals = (double***) calloc(nbond_old, sizeof(double**));
    dEr_indexes = (tagint***) calloc(nbond_old, sizeof(tagint**));
    memory->create(bond_extremes_pos, nbond_old, 2, "fresp:bond_extremes_pos");
    for (i = 0; i < nbond_old; i++) {
      dEr_vals[i] = NULL;
      dEr_indexes[i] = NULL;
    }
  }

  //Build new neighbor lists needed by F-RESP
  neighbor->build_one(list);

  for (i = 0; i < atom->nlocal; i++) {
    for (j = 0; j < atom->num_bond[i]; j++) {
      atom1 = atom->map(atom->bond_atom[i][j]);
      atom1 = domain->closest_image(i, (int)atom1);
      if (force->newton_bond || i < atom1)
        build_bond_Verlet_list(bond++, i, atom1);
    }
  }
}

/* ----------------------------------------------------------------------
   pre_force fluctuating charges update
------------------------------------------------------------------------- */

void FixFRespDsf::pre_force(int vflag)
{
  if (update->ntimestep % nevery) return;

  int i;

  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    memory->grow(deltaq, nmax, "fresp:deltaq");
  }

  //Activates calculation of kspace->eatom and pair->eatomcoul at each step
  update->eflag_atom = update->eflag_global = update->ntimestep;
  pe->addstep(update->ntimestep + 1);

  //deltaq, erfc_erf_arr and already_cycled arrays are cleared
  for (i = 0; i < nmax; i++) deltaq[i] = 0.0;

  if (Efieldflag || bondflag) q_update_Efield_bond();
  if (angleflag) q_update_angle();
  if (dihedralflag) q_update_dihedral();
  if (improperflag) q_update_improper();

  //Communicate deltaq for neighboring atoms
  pack_flag = 1;
  comm->reverse_comm_fix(this, 1);

  for (i = 0; i < atom->nlocal; i++)
    atom->q[i] = q0[types[atom->tag[i] - 1]] + deltaq[i];
  //Communicate atom->q for neighboring atoms
  pack_flag = 2;
  comm->forward_comm_fix(this);

  if (force->kspace) force->kspace->qsum_qsq();
}

/* ----------------------------------------------------------------------
   pre_reverse forces update
------------------------------------------------------------------------- */

void FixFRespDsf::pre_reverse(int eflag, int vflag)
{
  int bond, i, j, atom1_t, atom2_t, center_t;
  tagint der_atom, global_atom1, global_atom2, global_center, center;
  bigint molecule;
  double alpha, tot_pot, alpha_tot_pot, v[6], deltaf[3];
  double kb, kb_tot_pot, bondv[3], bondvinv;
  int atom1, atom2, atom1_pos, atom2_pos;

  if (update->ntimestep % nevery) return;

  // energy and virial setup
  if (vflag) v_setup(vflag);
  else evflag = 0;

  //Communicate force->kspace->eatom for neighboring atoms.
  pack_flag = 1;
  comm->forward_comm_fix(this);
  //Communicate reverse and forward force->pair->eatomcoul
  pack_flag = 3;
  comm->reverse_comm_fix(this);
  comm->forward_comm_fix(this);

  for (bond = 0; bond < nbond_old; bond++) {
    molecule = atom->molecule[dEr_indexes[bond][0][1]];
    global_atom1 = atom->tag[dEr_indexes[bond][0][1]];
    global_atom2 = atom->tag[dEr_indexes[bond][0][2]];
    atom1_t = types[global_atom1 - 1];
    atom2_t = types[global_atom2 - 1];
    if (bondflag) {
      atom1 = dEr_indexes[bond][0][1];
      atom2 = dEr_indexes[bond][0][2];
      atom1_pos = bond_extremes_pos[bond][0];
      atom2_pos = bond_extremes_pos[bond][1];
      bondv[0] = atom->x[atom1][0] - atom->x[atom2][0];
      bondv[1] = atom->x[atom1][1] - atom->x[atom2][1];
      bondv[2] = atom->x[atom1][2] - atom->x[atom2][2];
      bondvinv = 1.0 / MathExtra::len3(bondv);
    }
    for (j = 1; j <= mol_map[(int)molecule - 1][0]; j++) {
      global_center = (tagint)mol_map[molecule - 1][j];
      center = atom->map(global_center);
      center_t = types[global_center - 1];
      alpha = k_Efield[atom1_t][atom2_t][center_t];
      tot_pot = 2.0 * (force->pair->eatomcoul[center] +
        force->kspace->eatom[center]) / atom->q[center];
      alpha_tot_pot = alpha * tot_pot;
      if (bondflag) {
        kb = k_bond[atom1_t][atom2_t][center_t];
        kb_tot_pot = kb * tot_pot;
      }
      for (i = 0; i < dEr_indexes[bond][0][0]; i++) {
        if (dEr_indexes[bond][i + 1][1] == (tagint)-1) continue;
        der_atom = dEr_indexes[bond][i + 1][0];
        //Minus sign is needed because F = -dU/dr and dEr_vals * alpha_tot_pot
        //is dU/dr
        deltaf[0] = -dEr_vals[bond][i][0] * alpha_tot_pot;
        deltaf[1] = -dEr_vals[bond][i][1] * alpha_tot_pot;
        deltaf[2] = -dEr_vals[bond][i][2] * alpha_tot_pot;
        if (bondflag && (i == atom1_pos || i == atom2_pos)) {
          //if atom2 is considered
          if (i == atom2_pos) kb_tot_pot *= -1.0;
          deltaf[0] -= bondv[0] * kb_tot_pot * bondvinv;
          deltaf[1] -= bondv[1] * kb_tot_pot * bondvinv;
          deltaf[2] -= bondv[2] * kb_tot_pot * bondvinv;
        }
        MathExtra::add3(atom->f[der_atom], deltaf, atom->f[der_atom]);

        if (evflag) {
          v[0] = deltaf[0] * atom->x[der_atom][0];
          v[1] = deltaf[1] * atom->x[der_atom][1];
          v[2] = deltaf[2] * atom->x[der_atom][2];
          v[3] = deltaf[0] * atom->x[der_atom][1];
          v[4] = deltaf[0] * atom->x[der_atom][2];
          v[5] = deltaf[1] * atom->x[der_atom][2];
          v_tally(der_atom, v);
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixFRespDsf::memory_usage()
{
  int bond;
  double bytes = 0.0;
  bytes += atom->natoms * sizeof(int); //types
  bytes += nmolecules * (average_mol_size + 1) * sizeof(bigint); //mol_map
  bytes += natypes * natypes * natypes * sizeof(double); //k_bond
  bytes += natypes * natypes * natypes * natypes * sizeof(double); //k_angle
  bytes += natypes * natypes * natypes * natypes * natypes *
    sizeof(double); //k_dihedral
  bytes += natypes * natypes * natypes * natypes * natypes *
    sizeof(double); //k_improper
  bytes += natypes * natypes * natypes * sizeof(double); //k_Efield
  bytes += 2 * natypes * sizeof(double); //q0 and qgen
  bytes += 1 * nmax * sizeof(double); //deltaq
  //dEr_vals, dEr_indexes and distances
  for (bond = 0; bond < nbond_old; bond ++)
    bytes += dEr_indexes[bond][0][0] * (2 * (sizeof(tagint) + sizeof(bigint) + \
    3 * sizeof(double))) + 3 * sizeof(bigint);
  return bytes;
}

/* ---------------------------------------------------------------------- 
  Damping factor is returned and derivative of damping function is partially
  filled
   ---------------------------------------------------------------------- */

double FixFRespDsf::Efield_damping(double r, double *dampvec)
{
  static double c1invsq = 1.0 / (cutoff1 * cutoff1), cdiff = cutoff2 - cutoff1;
  if (dampflag == EXP) {
    double exp_part = MathSpecial::fm_exp(-0.5 * c1invsq * (r - cutoff2) *
      (r - cutoff2));
    MathExtra::scale3(exp_part * c1invsq * (r - cutoff2), dampvec);
    return exp_part;
  }
  else if (dampflag == SIN) {
    static double piocdiff = MathConst::MY_PI / cdiff;
    double sin_part, cos_part, arg = 0.5 * piocdiff * (r - cutoff1);
    #ifdef __INTEL_MKL__
    vdSinCos(1, &arg, &sin_part, &cos_part);
    #else
    sin_part = sin(arg);
    cos_part = cos(arg);
    #endif
    MathExtra::scale3(-sin_part * cos_part * piocdiff, dampvec);
    return sin_part * sin_part;
  }
  MathExtra::scale3(0.0, dampvec);
  return 1.0;
}
