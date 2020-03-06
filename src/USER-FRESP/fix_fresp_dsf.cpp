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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mkl.h> //Not mandatory
#include "fix_fresp_dsf.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "domain.h"
#include "neighbor.h"
#include "update.h"
#include "compute.h"
#include "force.h"
#include "math_special.h"
#include "math_const.h"
#include "memory.h"
#include "pair.h"
#include "error.h"
#include "math_extra.h"
#include "kspace.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define SMALL     0.001

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
  
  //else ifs are needed in order not to have segfault when trying to access
  //elements outside arg
  while (iarg < narg) {
    if ((arg[iarg] - strchr(arg[iarg], '#')) == 0) break;
    else if (strcmp(arg[iarg], "damp") == 0) {
      if (strcmp(arg[++iarg], "exp") == 0) dampflag = EXP;
      else if (strcmp(arg[iarg], "sin") == 0) dampflag = SIN;
      else if (strcmp(arg[iarg], "tho") == 0) {
        dampflag = THO;
	iarg++;
	continue;
      }
      cutoff1 = force->numeric(FLERR, arg[++iarg]);
      cutoff2 = force->numeric(FLERR, arg[++iarg]);
      iarg++;
    }
    else if (strcmp(arg[iarg++], "printEfield") == 0) {
      printEfieldflag = 1;
    }
  }

  //Check for sane arguments
  if ((nevery <= 0) || (cutoff1 < 0.0 || cutoff2 < 0.0 || cutoff3 <= 0.0))
    error->all(FLERR,"Illegal fix fresp command");

  nmax = 0;

  //Read FRESP types file
  read_file_types(arg[6]);

  //Create an array where q0 is associated with atom global indexes
  memory->create(q0, natypes, "fresp:q0");
  
  //Create an array where qgen is associated with atom global indexes
  memory->create(qgen, natypes, "fresp:qgen");

  //Read FRESP parameters file
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
  double xm[3], **x = atom->x, rvml, rvminv, rvminvsq, rvm[3], r0;
  double  bondvl, bondvinv, bondvinvsq, pref, q_gen; 
  bigint atom1, atom2, center, global_center, global_atom1, global_atom2;
  bigint molecule;
  int atom1_t, atom2_t, i, iplusone, bond, atom1_pos, atom2_pos, ftyp;
  double bondv[3], wstalin, E[3], Efield[3], Eparallel, rvmlsq, partialerfc;
  double grij, expm2, fsp, ssp, tsp;
  double ddamping[3];
  double fvp[3], svp[3], bondrvmprod, damping, factor_coul;
  static double tgeospi = 2.0 * g_ewald / MathConst::MY_PIS;
  static double cutoff3sq = cutoff3 * cutoff3, cutoff1sq = cutoff1 * cutoff1;
  static double tgecuospi = tgeospi * g_ewald * g_ewald;
  static double f_shift = MathSpecial::expmsq(g_ewald * cutoff3) *
    ((MathSpecial::my_erfcx(g_ewald * cutoff3) / cutoff3) + tgeospi) / cutoff3;
  double *special_coul = force->special_coul;

  //This cycle over all the atoms is absolutely needed
  if (Efieldflag && qsqsum > 0.0) {
    #pragma vector
    for (bond = 0; bond < nbond_old; bond++) {
      #pragma vector
      for (i = 0; i < dEr_indexes[bond][0][0]; i++) dEr_vals[bond][i][0] =
        dEr_vals[bond][i][1] = dEr_vals[bond][i][2] = 0.0;
    }
  }

  //Using bonds stored in dEr_indexes instead of nbondlist, cycle is
  //performed over all bonds, even those that are constrained by SHAKE. Even
  //bond stretching charge contribution is therefore accounted for.
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
    bondvinv = 1.0 / bondvl;

    //This check is here because, if false, bondv has already been calculated
    //and can be used for charge variation due to bond stretching
    //qsqsum is that declared in constructor, need to correct
    if (Efieldflag && qsqsum > 0.0) {
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
        //Last element of dEr_indexes[bond][][0] is the index of atom1 and
        //doesen't need a transformation through "& NEIGHMASK"
        if (iplusone == dEr_indexes[bond][0][0]) {
          factor_coul = 0.0;
          center = dEr_indexes[bond][iplusone][0];
        }
        else {
          factor_coul = special_coul[sbmask(dEr_indexes[bond][iplusone][0])];
          center = dEr_indexes[bond][iplusone][0] & NEIGHMASK;
        }
        global_center = atom->tag[center];
        rvm[0] = xm[0] - x[center][0];
        rvm[1] = xm[1] - x[center][1];
        rvm[2] = xm[2] - x[center][2];
        domain->minimum_image(rvm[0], rvm[1], rvm[2]);
        rvmlsq = MathExtra::lensq3(rvm);

        dEr_indexes[bond][iplusone][1] = (tagint)1;

        if (rvmlsq > cutoff3sq) {
          //In order not to cycle over this atom in pre_reverse function
          dEr_indexes[bond][iplusone][1] = (tagint)-1;
          continue;
        }

        if (rvmlsq < cutoff1sq && dampflag == SIN) continue;
        rvml = sqrt(rvmlsq);
        rvminv = 1.0 / rvml;
        rvminvsq = rvminv * rvminv;
        MathExtra::copy3(rvm, Efield);
        grij = g_ewald * rvml;
	ftyp = types[global_center - 1];
        q_gen = qgen[ftyp];
        bondrvmprod = MathExtra::dot3(bondv, rvm);
        expm2 = MathSpecial::expmsq(grij);
        partialerfc = MathSpecial::my_erfcx(grij);
        pref = expm2 * rvminv * (partialerfc * rvminv + tgeospi);
        pref -= f_shift;
        pref *= factor_coul * q_gen * rvminv * bondvinv;
        //Now pref is A * q_gen / (|rvm||rb|)
        fsp = 3.0 * bondrvmprod * rvminvsq;
        tsp = factor_coul * q_gen * bondvinv * rvminvsq * bondrvmprod *
          (f_shift * rvminv + expm2 * tgecuospi);

        if (rvml < cutoff2 && dampflag > 0) {
          MathExtra::copy3(rvm, ddamping);
          damping = Efield_damping(rvml, ddamping, ftyp);
          //tsp is multiplied times damping. Because pref too will
          //be multiplied times damping, the whole Efield derivative is damped.
          tsp *= damping;
          MathExtra::scale3(bondrvmprod * rvminv * pref, ddamping);
          pref *= damping;
          MathExtra::add3(ddamping, dEr_vals[bond][i], dEr_vals[bond][i]);
          //Derivative of damping function for atom1 and atom2 is calculated
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

        fvp[0] = fsp * rvm[0] - bondv[0];
        fvp[1] = fsp * rvm[1] - bondv[1];
        fvp[2] = fsp * rvm[2] - bondv[2];
        svp[0] = tsp * rvm[0];
        svp[1] = tsp * rvm[1];
        svp[2] = tsp * rvm[2];
      
        dEr_vals[bond][i][0] += pref * fvp[0] + 2.0 * svp[0];
        dEr_vals[bond][i][1] += pref * fvp[1] + 2.0 * svp[1];
        dEr_vals[bond][i][2] += pref * fvp[2] + 2.0 * svp[2];

        fsp = 0.5 - bondrvmprod * bondvinvsq;
        ssp = 1.0 - 1.5 * bondrvmprod * rvminvsq;
        fvp[0] = fsp * bondv[0] + ssp * rvm[0];
        fvp[1] = fsp * bondv[1] + ssp * rvm[1];
        fvp[2] = fsp * bondv[2] + ssp * rvm[2];

        dEr_vals[bond][atom1_pos][0] += pref * fvp[0] - svp[0];
        dEr_vals[bond][atom1_pos][1] += pref * fvp[1] - svp[1];
        dEr_vals[bond][atom1_pos][2] += pref * fvp[2] - svp[2];

        fsp = 1.0 - fsp;
        ssp = 2.0 - ssp;
        fvp[0] = fsp * bondv[0] - ssp * rvm[0];
        fvp[1] = fsp * bondv[1] - ssp * rvm[1];
        fvp[2] = fsp * bondv[2] - ssp * rvm[2];

        dEr_vals[bond][atom2_pos][0] += pref * fvp[0] - svp[0];
        dEr_vals[bond][atom2_pos][1] += pref * fvp[1] - svp[1];
        dEr_vals[bond][atom2_pos][2] += pref * fvp[2] - svp[2];
      }
      Eparallel = MathExtra::dot3(E, bondv);

      if (printEfieldflag) fprintf(stderr, BIGINT_FORMAT " " BIGINT_FORMAT
      " %.14lf\n", global_atom1, global_atom2, Eparallel);

      deltaq_update_Efield(molecule, atom1_t, atom2_t, Eparallel);
    }

    if (bondflag) {
      r0 = force->bond->equilibrium_distance(neighbor->bondlist[bond][2]);
      wstalin = bondvl - r0;

      db_vals[bond][0] = bondv[0] * bondvinv;
      db_vals[bond][1] = bondv[1] * bondvinv;
      db_vals[bond][2] = bondv[2] * bondvinv;

      deltaq_update_bond(molecule, atom1_t, atom2_t, wstalin);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixFRespDsf::setup_pre_force(int vflag)
{
  int i, j, bond = 0;
  bigint atom1;

  nmax = atom->nmax;
  memory->create(deltaq, nmax, "fresp:deltaq");

  if (Efieldflag || bondflag) {
    //Using count_total_bonds(), bonds "freezed" by SHAKE are considered too,
    //otherwise than using nbondlist
    nbond_old = count_total_bonds();

    //An array of nbond double** is allocated in order to store the values of
    //derivatives of E_R * bond unit vector
    //an array of nbond tagint* is allocated in order to store the indexes of
    //atoms wrt the derivatives of dEr_vals are done
    dEr_vals = (double***) calloc(nbond_old, sizeof(double**));
    if (dEr_vals == NULL) {
      char wengels[128];
      sprintf(wengels,"Failed to allocate " BIGINT_FORMAT " bytes for array \
        fresp:dEr_vals", nbond_old * sizeof(double**));
      error->one(FLERR,wengels);
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

    if (bondflag) memory->create(db_vals, nbond_old, 3, "fresp:db_vals");
  }

  if (angleflag) {
    nangle_old = neighbor->nanglelist;
    memory->create(da_vals, nangle_old, 4, 3, "fresp:da_vals");
  }

  if (improperflag) {
    nimproper_old = neighbor->nimproperlist;
    memory->create(dimp_vals, nimproper_old, 4, 3, "fresp:dimp_vals");
  }

  //Building of new neighbor lists needed by F-RESP
  neighbor->build_one(list);

  if (Efieldflag || bondflag) {
    for (i = 0; i < atom->nlocal; i++) {
      for (j = 0; j < atom->num_bond[i]; j++) {
        atom1 = atom->map(atom->bond_atom[i][j]);
        atom1 = domain->closest_image(i, (int)atom1);
        if (force->newton_bond || i < atom1)
          build_bond_Verlet_list(bond++, i, atom1);
      }
    }
  }

  pre_force(vflag); 
}
 
/* ---------------------------------------------------------------------- */

void FixFRespDsf::post_neighbor()
{
  int i, j, end, bond = 0;
  bigint atom1;

  if (Efieldflag || bondflag) {
    //Content of dEr_vals and dEr_indexes arrays is freed.
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

    //dEr_vals and dEr_indexes arrays are deallocated, reallocated and
    //initialized as pointing to NULL.
    if (nbond_old != j) {
      nbond_old = j;
      memory->grow(bond_extremes_pos, nbond_old, 2, "fresp:bond_extremes_pos");
      free(dEr_vals);
      free(dEr_indexes);
      dEr_vals = (double***) calloc(nbond_old, sizeof(double**));
      dEr_indexes = (tagint***) calloc(nbond_old, sizeof(tagint**));
      for (i = 0; i < nbond_old; i++) {
        dEr_vals[i] = NULL;
        dEr_indexes[i] = NULL;
      }
    }

    if (bondflag) memory->create(db_vals, nbond_old, 3, "fresp:db_vals");
  }

  if (angleflag) {
    nangle_old = neighbor->nanglelist;
    memory->grow(da_vals, nangle_old, 4, 3, "fresp:da_vals");
  }

  if (improperflag) {
    nimproper_old = neighbor->nimproperlist;
    memory->grow(dimp_vals, nimproper_old, 4, 3, "fresp:dimp_vals");
  }

  //Building of new neighbor lists needed by F-RESP
  neighbor->build_one(list);

  if (Efieldflag || bondflag) {
    for (i = 0; i < atom->nlocal; i++) {
      for (j = 0; j < atom->num_bond[i]; j++) {
        atom1 = atom->map(atom->bond_atom[i][j]);
        atom1 = domain->closest_image(i, (int)atom1);
        if (force->newton_bond || i < atom1)
          build_bond_Verlet_list(bond++, i, atom1);
      }
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

  //deltaq array is cleared
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
  if (update->ntimestep % nevery) return;

  //Energy and virial setup
  if (vflag) v_setup(vflag);
  else evflag = 0;

  //Communicate force->kspace->eatom for neighboring atoms.
  pack_flag = 1;
  comm->forward_comm_fix(this);
  //Communicate reverse and forward force->pair->eatomcoul
  pack_flag = 3;
  comm->reverse_comm_fix(this);
  comm->forward_comm_fix(this);

  if (Efieldflag || bondflag) force_update_Efield_bond();
  if (angleflag) force_update_angle();
  if (improperflag) force_update_improper();
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixFRespDsf::memory_usage()
{
  int i;
  double bytes = 0.0;
  bytes += atom->natoms * sizeof(int); //types
  bytes += nmolecules * (average_mol_size + 1) * sizeof(bigint); //mol_map
  bytes += 2 * natypes * sizeof(double); //q0 and qgen
  bytes += 1 * nmax * sizeof(double); //deltaq
  if (angleflag) {
    bytes += natypes * natypes * natypes * natypes * sizeof(double); //k_angle
    //da_vals
    bytes += nangle_old * 9 * sizeof(double);
  }
  if (dihedralflag) bytes += natypes * natypes * natypes * natypes * natypes *
    sizeof(double); //k_dihedral
  if (improperflag) {
    //k_improper
    bytes += natypes * natypes * natypes * natypes * natypes * sizeof(double);
    //dimp_vals
    bytes += nimproper_old * 12 * sizeof(double);
  }
  if (Efieldflag || bondflag) {
    //dEr_vals and dEr_indexes
    for (i = 0; i < nbond_old; i ++)
      bytes += dEr_indexes[i][0][0] * (2 * (sizeof(tagint) + sizeof(bigint) + 
      3 * sizeof(double))) + 3 * sizeof(bigint);
    //k_Efield
    if (Efieldflag) bytes += natypes * natypes * natypes * sizeof(double);
    if (bondflag) {
      bytes += nbond_old * sizeof(double); //db_vals
      bytes += natypes * natypes * natypes * sizeof(double); //k_bond
    }
  }
  return bytes;
}

/* ---------------------------------------------------------------------- 
  Damping factor is returned and derivative of damping function is partially
  filled
   ---------------------------------------------------------------------- */

double FixFRespDsf::Efield_damping(double r, double *dampvec, int jtype = 0)
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
  else if (dampflag == THO) {
    double s = ascreen[jtype];
    double shalf = 0.5 * s;
    double exp_part = MathSpecial::fm_exp(-s * r);
    MathExtra::scale3(-shalf * exp_part * (1. + 1. / r), dampvec);
    return 1. - (1. + shalf * r) * exp_part;
  }
  MathExtra::scale3(0.0, dampvec);
  return 1.0;
}

/* ---------------------------------------------------------------------- 
  Forces are updated considering electric field and bond stretching
    polarization contribution
   ---------------------------------------------------------------------- */

void FixFRespDsf::force_update_Efield_bond()
{
  int bond, i, j, atom1_t, atom2_t, center_t, iplusone;
  tagint der_atom, global_atom1, global_atom2, global_center, center;
  tagint atom1;
  bigint molecule;
  double alpha, tot_pot, alpha_tot_pot, v[6], deltaf[3];
  double kb, kb_tot_pot;
  int atom1_pos, atom2_pos;

  //Contributions from electric field and bond stretching could be separed,
  //therefore permitting to calculate bond stretching contribution only
  //for non-constrained bonds, differently from the current implementation
  for (bond = 0; bond < nbond_old; bond++) {
    atom1 = dEr_indexes[bond][0][1];
    global_atom1 = atom->tag[atom1];
    global_atom2 = atom->tag[dEr_indexes[bond][0][2]];
    atom1_t = types[global_atom1 - 1];
    atom2_t = types[global_atom2 - 1];
    molecule = atom->molecule[atom1];
    if (bondflag) {
      atom1_pos = bond_extremes_pos[bond][0];
      atom2_pos = bond_extremes_pos[bond][1];
    }
    for (j = 1; j <= mol_map[(int)molecule - 1][0]; j++) {
      global_center = (tagint)mol_map[molecule - 1][j];
      center = atom->map(global_center);
      center_t = types[global_center - 1];
      tot_pot = 2.0 * (force->pair->eatomcoul[center] +
        force->kspace->eatom[center]) / atom->q[center];
      if (Efieldflag) {
        alpha = k_Efield[atom1_t][atom2_t][center_t];
        alpha_tot_pot = alpha * tot_pot;
      }
      if (bondflag) {
        kb = k_bond[atom1_t][atom2_t][center_t];
        kb_tot_pot = kb * tot_pot;
      }
      for (i = 0; i < dEr_indexes[bond][0][0]; i++) {
        iplusone = i + 1;
        if (dEr_indexes[bond][iplusone][1] == (tagint)-1) continue;
        //Last element of dEr_indexes[bond][][0] is the index of atom1 and
        //doesen't need a transformatin through "& NEIGHMASK"
        if (iplusone == dEr_indexes[bond][0][0])
          der_atom = dEr_indexes[bond][iplusone][0];
        else
          der_atom = dEr_indexes[bond][iplusone][0] & NEIGHMASK;
        MathExtra::zero3(deltaf);
        if (Efieldflag) {
          //Minus sign is needed because F = -dU/dr and dEr_vals * alpha_tot_pot
          //is dU/dr
          deltaf[0] -= dEr_vals[bond][i][0] * alpha_tot_pot;
          deltaf[1] -= dEr_vals[bond][i][1] * alpha_tot_pot;
          deltaf[2] -= dEr_vals[bond][i][2] * alpha_tot_pot;
        }
        if (bondflag && (i == atom1_pos || i == atom2_pos)) {
          if (i == atom1_pos) {
            deltaf[0] -= db_vals[bond][0] * kb_tot_pot;
            deltaf[1] -= db_vals[bond][1] * kb_tot_pot;
            deltaf[2] -= db_vals[bond][2] * kb_tot_pot;
          }
          else if (i == atom2_pos) {
            //If atom2 is considered, the sign of bond length derivative has
            //to be reversed
            deltaf[0] += db_vals[bond][0] * kb_tot_pot;
            deltaf[1] += db_vals[bond][1] * kb_tot_pot;
            deltaf[2] += db_vals[bond][2] * kb_tot_pot;
          }
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
  Forces are updated considering angle bending polarization contribution
   ---------------------------------------------------------------------- */

void FixFRespDsf::force_update_angle()
{
  int i, j, atom1_t, atom2_t, center_t;
  tagint der_atom, global_atom1, global_atom2, global_center, center;
  tagint global_atom3;
  bigint molecule;
  double tot_pot, v[6], deltaf[3];
  int atom1, atom2, atom3, atom3_t, angle;
  int **anglelist = neighbor->anglelist;
  double ka, ka_tot_pot;

  for (angle = 0; angle < nangle_old; angle++) {
    atom1 = anglelist[angle][0];
    atom2 = anglelist[angle][1];
    atom3 = anglelist[angle][2];
    global_atom1 = atom->tag[atom1];
    global_atom2 = atom->tag[atom2];
    global_atom3 = atom->tag[atom3];
    atom1_t = types[global_atom1 - 1];
    atom2_t = types[global_atom2 - 1];
    atom3_t = types[global_atom3 - 1];
    molecule = atom->molecule[atom1];
    for (j = 1; j <= mol_map[(int)molecule - 1][0]; j++) {
      global_center = (tagint)mol_map[molecule - 1][j];
      center = atom->map(global_center);
      center_t = types[global_center - 1];
      ka = k_angle[atom1_t][atom2_t][atom3_t][center_t];
      tot_pot = 2.0 * (force->pair->eatomcoul[center] +
        force->kspace->eatom[center]) / atom->q[center];
      ka_tot_pot = ka * tot_pot;
      for (i = 0; i < 3; i ++) {
        der_atom = anglelist[angle][i];
        deltaf[0] = -da_vals[angle][i][0] * ka_tot_pot;
        deltaf[1] = -da_vals[angle][i][1] * ka_tot_pot;
        deltaf[2] = -da_vals[angle][i][2] * ka_tot_pot;
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
  Forces are updated considering improper torsion polarization contribution
   ---------------------------------------------------------------------- */

void FixFRespDsf::force_update_improper()
{
  int i, j, atom1_t, atom2_t, center_t;
  tagint der_atom, global_atom1, global_atom2, global_center, center;
  tagint global_atom3, global_atom4;
  bigint molecule;
  double tot_pot, v[6], deltaf[3];
  int atom1, atom2, atom3, atom4, atom3_t, atom4_t, improper;
  int **improperlist = neighbor->improperlist;
  double ki, ki_tot_pot;

  for (improper = 0; improper < nimproper_old; improper++) {
    atom1 = improperlist[improper][0];
    atom2 = improperlist[improper][1];
    atom3 = improperlist[improper][2];
    atom4 = improperlist[improper][3];
    global_atom1 = atom->tag[atom1];
    global_atom2 = atom->tag[atom2];
    global_atom3 = atom->tag[atom3];
    global_atom4 = atom->tag[atom4];
    atom1_t = types[global_atom1 - 1];
    atom2_t = types[global_atom2 - 1];
    atom3_t = types[global_atom3 - 1];
    atom4_t = types[global_atom4 - 1];
    molecule = atom->molecule[atom1];
    for (j = 1; j <= mol_map[(int)molecule - 1][0]; j++) {
      global_center = (tagint)mol_map[molecule - 1][j];
      center = atom->map(global_center);
      center_t = types[global_center - 1];
      ki = k_improper[atom1_t][atom2_t][atom3_t][atom4_t][center_t];
      tot_pot = 2.0 * (force->pair->eatomcoul[center] +
        force->kspace->eatom[center]) / atom->q[center];
      ki_tot_pot = ki * tot_pot;
      for (i = 0; i < 4; i ++) {
        der_atom = improperlist[improper][i];
        deltaf[0] = -dimp_vals[improper][i][0] * ki_tot_pot;
        deltaf[1] = -dimp_vals[improper][i][1] * ki_tot_pot;
        deltaf[2] = -dimp_vals[improper][i][2] * ki_tot_pot;
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
