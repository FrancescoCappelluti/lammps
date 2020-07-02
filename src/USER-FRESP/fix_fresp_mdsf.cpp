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
#include "fix_fresp_mdsf.h"
#include "atom.h"
#include "bond.h"
#include "domain.h"
#include "neighbor.h"
#include "force.h"
#include "math_special.h"
#include "math_const.h"
#include "math_extra.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixFRespMDsf::FixFRespMDsf(LAMMPS *lmp, int narg, char **arg) :
  FixFRespDsf(lmp, narg, arg)
{
  beta = 0.1; //Best value found
}

/* ---------------------------------------------------------------------- */

FixFRespMDsf::~FixFRespMDsf()
{
}

/* ---------------------------------------------------------------------- 
   charges fluctuation due to electric field on bonds
------------------------------------------------------------------------ */

void FixFRespMDsf::q_update_Efield_bond()
{
  double xm[3], **x = atom->x, rvml, rvminv, rvminvsq, rvm[3], r0;
  double  bondvl, bondvinv, bondvinvsq, pref, q_gen; 
  bigint atom1, atom2, center, global_center, global_atom1, global_atom2;
  bigint molecule;
  int atom1_t, atom2_t, i, iplusone, bond, atom1_pos, atom2_pos, center_t;
  double bondv[3], wlenin, E[3], Efield[3], Eparallel, rvmlsq, partialerfc;
  double grij, expm2, fsp, ssp, tsp, betaexp;
  double ddamping[3];
  double fvp[3], svp[3], bondrvmprod, damping, factor_coul;
  static double tgeospi = 2.0 * g_ewald / MathConst::MY_PIS;
  static double cutoff3sq = cutoff3 * cutoff3, cutoff1sq = cutoff1 * cutoff1;
  static double tgecuospi = tgeospi * g_ewald * g_ewald;
  static double f_shift = MathSpecial::expmsq(g_ewald * cutoff3) *
    ((MathSpecial::my_erfcx(g_ewald * cutoff3) / cutoff3) + tgeospi) / cutoff3;
  static double halfbeta = beta * 0.5;
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
          continue;
        }

        if (rvmlsq < cutoff1sq && dampflag == SIN) continue;
        rvml = sqrt(rvmlsq);
        rvminv = 1.0 / rvml;
        rvminvsq = rvminv * rvminv;
        MathExtra::copy3(rvm, Efield);
        grij = g_ewald * rvml;
	center_t = types[global_center - 1];
        q_gen = qgen[center_t];
        bondrvmprod = MathExtra::dot3(bondv, rvm);
        expm2 = MathSpecial::expmsq(grij);
        partialerfc = MathSpecial::my_erfcx(grij);
        pref = expm2 * rvminv * (partialerfc * rvminv + tgeospi);
        betaexp = MathSpecial::fm_exp(beta * (rvml - cutoff3));
        pref -= f_shift * betaexp;
        pref *= factor_coul * q_gen * rvminv * bondvinv;
        //Now pref is A * q_gen / (|rvm||rb|)
        fsp = 3.0 * bondrvmprod * rvminvsq;
        tsp = factor_coul * q_gen * bondvinv * rvminvsq * bondrvmprod *
          (f_shift * (rvminv + halfbeta) + expm2 * tgecuospi);

        if (rvml < cutoff2 && dampflag > 0) {
          MathExtra::copy3(rvm, ddamping);
          damping = Efield_damping(rvml, ddamping, center_t, atom1_t, atom2_t);
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
      wlenin = bondvl - r0;

      db_vals[bond][0] = bondv[0] * bondvinv;
      db_vals[bond][1] = bondv[1] * bondvinv;
      db_vals[bond][2] = bondv[2] * bondvinv;

      deltaq_update_bond(molecule, atom1_t, atom2_t, wlenin);
    }
  }
}
