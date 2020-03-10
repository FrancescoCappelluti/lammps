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
#include "fix_fresp_ewald.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "domain.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
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

#define EWALD_SMALL 0.00001
#define TWO_OVER_SQPI 1.128379167

/* ---------------------------------------------------------------------- */

FixFRespEwald::FixFRespEwald(LAMMPS *lmp, int narg, char **arg) :
  FixFResp(lmp, narg, arg) {
  if (narg < 7 || narg > 17) error->all(FLERR,"Illegal fix fresp command");

  int iarg = 7;

  //Following else ifs are needed in order not to have segfault when
  //trying to access elements outside arg
  while (iarg < narg) {
    if (strcmp(arg[iarg], "gewald") == 0) {
      gewaldflag = 0;
      g_ewald = force->numeric(FLERR, arg[++iarg]);
      iarg++;
    }
    else if (strcmp(arg[iarg], "kmax") == 0) {
      kewaldflag = 0;
      kxmax = force->inumeric(FLERR, arg[++iarg]);
      kymax = force->inumeric(FLERR, arg[++iarg]);
      kzmax = force->inumeric(FLERR, arg[++iarg]);
      iarg++;
    }
    else if (strcmp(arg[iarg], "damp") == 0) {
      if (strcmp(arg[++iarg], "sin") == 0) dampflag = SIN;
      else if (strcmp(arg[iarg], "exp") == 0) dampflag = EXP;
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

  //Read FRESP types file
  read_file_types(arg[5]);

  //Create an array where q0 is associated with atom global indexes
  memory->create(q0, natypes, "fresp:q0");
  
  //Create an array where qgen is associated with atom global indexes
  memory->create(qgen, natypes, "fresp:qgen");

  //Read FRESP parameters file
  read_file(arg[6]);

}

/* ---------------------------------------------------------------------- */

FixFRespEwald::~FixFRespEwald()
{
  memory->destroy(appo2);
  ewald_deallocate();
  memory->destroy(ek);
  memory->destroy3d_offset(cs,-kmax_created);
  memory->destroy3d_offset(sn,-kmax_created);
  memory->destroy(appo3);
}

/* ----------------------------------------------------------------------
   compute qsum,qsqsum,q2 and give error/warning if not charge neutral
   called initially, when particle count changes, when charges are changed
------------------------------------------------------------------------- */

void FixFRespEwald::ewald_qsum_qsq()
{
  double qsum_local = 0.0, qsqsum_local = 0.0;

  #if defined(_OPENMP)
    #pragma omp parallel for default(none) reduction(+:qsum_local,qsqsum_local)
  #endif

  for (int i = 0; i < atom->nlocal; i++) {
    qsum_local += atom->q[i];
    qsqsum_local += atom->q[i] * atom->q[i];
  }

  MPI_Allreduce(&qsum_local, &qsum, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&qsqsum_local, &qsqsum, 1, MPI_DOUBLE, MPI_SUM, world);

  if ((qsqsum == 0.0) && (comm->me == 0) && warn_nocharge) {
    error->warning(FLERR,"Using F-RESP kspace solver on system with no charge");
    warn_nocharge = 0;
  }

  q2 = qsqsum * force->qqrd2e;

  //Not yet sure of the correction needed for non-neutral systems
  //so issue warning or error

  if (fabs(qsum) > EWALD_SMALL) {
    char str[128];
    sprintf(str,"System is not charge neutral, net charge = %g",qsum);
    if (!warn_nonneutral) error->all(FLERR,str);
    if (warn_nonneutral == 1 && comm->me == 0) error->warning(FLERR,str);
    warn_nonneutral = 2;
  }
}

/* ----------------------------------------------------------------------
   pre-compute coefficients for each Ewald K-vector
------------------------------------------------------------------------- */

void FixFRespEwald::ewald_coeffs()
{
  int k,l,m;
  double sqk,vterm;

  double g_ewald_sq_inv = 1.0 / (g_ewald*g_ewald);
  double preu = 4.0*MathConst::MY_PI/volume;

  kcount = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (m = 1; m <= kmax; m++) {
    sqk = (m*unitk[0]) * (m*unitk[0]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = m;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = 0;
      kvecs[kcount][0] = m * unitk[0];
      kvecs[kcount][1] = 0.0;
      kvecs[kcount][2] = 0.0;
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg[kcount][0] = 2.0*unitk[0]*m*ug[kcount];
      eg[kcount][1] = 0.0;
      eg[kcount][2] = 0.0;
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg[kcount][0] = 1.0 + vterm*(unitk[0]*m)*(unitk[0]*m);
      vg[kcount][1] = 1.0;
      vg[kcount][2] = 1.0;
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;
      kcount++;
    }
    sqk = (m*unitk[1]) * (m*unitk[1]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = m;
      kzvecs[kcount] = 0;
      kvecs[kcount][0] = 0.0;
      kvecs[kcount][1] = m * unitk[1];
      kvecs[kcount][2] = 0.0;
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg[kcount][0] = 0.0;
      eg[kcount][1] = 2.0*unitk[1]*m*ug[kcount];
      eg[kcount][2] = 0.0;
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg[kcount][0] = 1.0;
      vg[kcount][1] = 1.0 + vterm*(unitk[1]*m)*(unitk[1]*m);
      vg[kcount][2] = 1.0;
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;
      kcount++;
    }
    sqk = (m*unitk[2]) * (m*unitk[2]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = m;
      kvecs[kcount][0] = 0.0;
      kvecs[kcount][1] = 0.0;
      kvecs[kcount][2] = m * unitk[2];
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg[kcount][0] = 0.0;
      eg[kcount][1] = 0.0;
      eg[kcount][2] = 2.0*unitk[2]*m*ug[kcount];
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg[kcount][0] = 1.0;
      vg[kcount][1] = 1.0;
      vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;
      kcount++;
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = l;
        kzvecs[kcount] = 0;
        kvecs[kcount][0] = k * unitk[0];
        kvecs[kcount][1] = l * unitk[1];
        kvecs[kcount][2] = 0.0;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
        eg[kcount][1] = 2.0*unitk[1]*l*ug[kcount];
        eg[kcount][2] = 0.0;
        vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
        vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
        vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
        vg[kcount][2] = 1.0;
        vg[kcount][3] = vterm*unitk[0]*k*unitk[1]*l;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = 0.0;
        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = -l;
        kzvecs[kcount] = 0;
        kvecs[kcount][0] = k * unitk[0];
        kvecs[kcount][1] = -l * unitk[1];
        kvecs[kcount][2] = 0.0;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
        eg[kcount][1] = -2.0*unitk[1]*l*ug[kcount];
        eg[kcount][2] = 0.0;
        vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
        vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
        vg[kcount][2] = 1.0;
        vg[kcount][3] = -vterm*unitk[0]*k*unitk[1]*l;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = 0.0;
        kcount++;;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitk[1]*l) * (unitk[1]*l) + (unitk[2]*m) * (unitk[2]*m);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = m;
        kvecs[kcount][0] = 0.0;
        kvecs[kcount][1] = l * unitk[1];
        kvecs[kcount][2] = m * unitk[2];
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] =  0.0;
        eg[kcount][1] =  2.0*unitk[1]*l*ug[kcount];
        eg[kcount][2] =  2.0*unitk[2]*m*ug[kcount];
        vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
        vg[kcount][0] = 1.0;
        vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
        vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
        vg[kcount][3] = 0.0;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = vterm*unitk[1]*l*unitk[2]*m;
        kcount++;

        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = -m;
        kvecs[kcount][0] = 0.0;
        kvecs[kcount][1] = l * unitk[1];
        kvecs[kcount][2] = -m * unitk[2];
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] =  0.0;
        eg[kcount][1] =  2.0*unitk[1]*l*ug[kcount];
        eg[kcount][2] = -2.0*unitk[2]*m*ug[kcount];
        vg[kcount][0] = 1.0;
        vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
        vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
        vg[kcount][3] = 0.0;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = -vterm*unitk[1]*l*unitk[2]*m;
        kcount++;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[2]*m) * (unitk[2]*m);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = 0;
        kzvecs[kcount] = m;
        kvecs[kcount][0] = k * unitk[0];
        kvecs[kcount][1] = 0.0;
        kvecs[kcount][2] = m * unitk[2];
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] =  2.0*unitk[0]*k*ug[kcount];
        eg[kcount][1] =  0.0;
        eg[kcount][2] =  2.0*unitk[2]*m*ug[kcount];
        vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
        vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
        vg[kcount][1] = 1.0;
        vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
        vg[kcount][3] = 0.0;
        vg[kcount][4] = vterm*unitk[0]*k*unitk[2]*m;
        vg[kcount][5] = 0.0;
        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = 0;
        kzvecs[kcount] = -m;
        kvecs[kcount][0] = k * unitk[0];
        kvecs[kcount][1] = 0.0;
        kvecs[kcount][2] = -m * unitk[2];
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] =  2.0*unitk[0]*k*ug[kcount];
        eg[kcount][1] =  0.0;
        eg[kcount][2] = -2.0*unitk[2]*m*ug[kcount];
        vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
        vg[kcount][1] = 1.0;
        vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
        vg[kcount][3] = 0.0;
        vg[kcount][4] = -vterm*unitk[0]*k*unitk[2]*m;
        vg[kcount][5] = 0.0;
        kcount++;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l) +
          (unitk[2]*m) * (unitk[2]*m);
        if (sqk <= gsqmx) {
          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = m;
          kvecs[kcount][0] = k * unitk[0];
          kvecs[kcount][1] = l * unitk[0];
          kvecs[kcount][2] = m * unitk[0];
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
          eg[kcount][1] = 2.0*unitk[1]*l*ug[kcount];
          eg[kcount][2] = 2.0*unitk[2]*m*ug[kcount];
          vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
          vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
          vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
          vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
          vg[kcount][3] = vterm*unitk[0]*k*unitk[1]*l;
          vg[kcount][4] = vterm*unitk[0]*k*unitk[2]*m;
          vg[kcount][5] = vterm*unitk[1]*l*unitk[2]*m;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = -l;
          kzvecs[kcount] = m;
          kvecs[kcount][0] = k * unitk[0];
          kvecs[kcount][1] = -l * unitk[0];
          kvecs[kcount][2] = m * unitk[0];
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
          eg[kcount][1] = -2.0*unitk[1]*l*ug[kcount];
          eg[kcount][2] = 2.0*unitk[2]*m*ug[kcount];
          vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
          vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
          vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
          vg[kcount][3] = -vterm*unitk[0]*k*unitk[1]*l;
          vg[kcount][4] = vterm*unitk[0]*k*unitk[2]*m;
          vg[kcount][5] = -vterm*unitk[1]*l*unitk[2]*m;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = -m;
          kvecs[kcount][0] = k * unitk[0];
          kvecs[kcount][1] = l * unitk[0];
          kvecs[kcount][2] = -m * unitk[0];
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
          eg[kcount][1] = 2.0*unitk[1]*l*ug[kcount];
          eg[kcount][2] = -2.0*unitk[2]*m*ug[kcount];
          vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
          vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
          vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
          vg[kcount][3] = vterm*unitk[0]*k*unitk[1]*l;
          vg[kcount][4] = -vterm*unitk[0]*k*unitk[2]*m;
          vg[kcount][5] = -vterm*unitk[1]*l*unitk[2]*m;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = -l;
          kzvecs[kcount] = -m;
          kvecs[kcount][0] = k * unitk[0];
          kvecs[kcount][1] = -l * unitk[0];
          kvecs[kcount][2] = -m * unitk[0];
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
          eg[kcount][1] = -2.0*unitk[1]*l*ug[kcount];
          eg[kcount][2] = -2.0*unitk[2]*m*ug[kcount];
          vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
          vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
          vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
          vg[kcount][3] = -vterm*unitk[0]*k*unitk[1]*l;
          vg[kcount][4] = -vterm*unitk[0]*k*unitk[2]*m;
          vg[kcount][5] = vterm*unitk[1]*l*unitk[2]*m;
          kcount++;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   pre-compute coefficients for each Ewald K-vector for a triclinic
   system
------------------------------------------------------------------------- */

void FixFRespEwald::ewald_coeffs_triclinic()
{
  int k,l,m;
  double sqk,vterm;

  double g_ewald_sq_inv = 1.0 / (g_ewald*g_ewald);
  double preu = 4.0*MathConst::MY_PI/volume;

  double unitk_lamda[3];

  kcount = 0;

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = -kymax; l <= kymax; l++) {
      for (m = -kzmax; m <= kzmax; m++) {
        unitk_lamda[0] = 2.0*MathConst::MY_PI*k;
        unitk_lamda[1] = 2.0*MathConst::MY_PI*l;
        unitk_lamda[2] = 2.0*MathConst::MY_PI*m;
        ewald_x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
        sqk = unitk_lamda[0]*unitk_lamda[0] + unitk_lamda[1]*unitk_lamda[1] +
          unitk_lamda[2]*unitk_lamda[2];
        if (sqk <= gsqmx) {
          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = m;
          kvecs[kcount][0] = k * unitk[0];
          kvecs[kcount][1] = l * unitk[0];
          kvecs[kcount][2] = m * unitk[0];
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          eg[kcount][0] = 2.0*unitk_lamda[0]*ug[kcount];
          eg[kcount][1] = 2.0*unitk_lamda[1]*ug[kcount];
          eg[kcount][2] = 2.0*unitk_lamda[2]*ug[kcount];
          vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
          vg[kcount][0] = 1.0 + vterm*unitk_lamda[0]*unitk_lamda[0];
          vg[kcount][1] = 1.0 + vterm*unitk_lamda[1]*unitk_lamda[1];
          vg[kcount][2] = 1.0 + vterm*unitk_lamda[2]*unitk_lamda[2];
          vg[kcount][3] = vterm*unitk_lamda[0]*unitk_lamda[1];
          vg[kcount][4] = vterm*unitk_lamda[0]*unitk_lamda[2];
          vg[kcount][5] = vterm*unitk_lamda[1]*unitk_lamda[2];
          kcount++;
        }
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = -kzmax; m <= kzmax; m++) {
      unitk_lamda[0] = 0.0;
      unitk_lamda[1] = 2.0*MathConst::MY_PI*l;
      unitk_lamda[2] = 2.0*MathConst::MY_PI*m;
      ewald_x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
      sqk = unitk_lamda[1]*unitk_lamda[1] + unitk_lamda[2]*unitk_lamda[2];
      if (sqk <= gsqmx) {
        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = m;
        kvecs[kcount][0] = 0.0;
        kvecs[kcount][1] = l * unitk[0];
        kvecs[kcount][2] = m * unitk[0];
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] =  0.0;
        eg[kcount][1] =  2.0*unitk_lamda[1]*ug[kcount];
        eg[kcount][2] =  2.0*unitk_lamda[2]*ug[kcount];
        vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
        vg[kcount][0] = 1.0;
        vg[kcount][1] = 1.0 + vterm*unitk_lamda[1]*unitk_lamda[1];
        vg[kcount][2] = 1.0 + vterm*unitk_lamda[2]*unitk_lamda[2];
        vg[kcount][3] = 0.0;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = vterm*unitk_lamda[1]*unitk_lamda[2];
        kcount++;
      }
    }
  }

  // (0,0,m)

  for (m = 1; m <= kmax; m++) {
    unitk_lamda[0] = 0.0;
    unitk_lamda[1] = 0.0;
    unitk_lamda[2] = 2.0*MathConst::MY_PI*m;
    ewald_x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
    sqk = unitk_lamda[2]*unitk_lamda[2];
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = m;
      kvecs[kcount][0] = 0.0;
      kvecs[kcount][1] = 0.0;
      kvecs[kcount][2] = m * unitk[0];
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg[kcount][0] = 0.0;
      eg[kcount][1] = 0.0;
      eg[kcount][2] = 2.0*unitk_lamda[2]*ug[kcount];
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg[kcount][0] = 1.0;
      vg[kcount][1] = 1.0;
      vg[kcount][2] = 1.0 + vterm*unitk_lamda[2]*unitk_lamda[2];
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;
      kcount++;
    }
  }
}

/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void FixFRespEwald::ewald_allocate()
{
  kxvecs = new int[kmax3d];
  kyvecs = new int[kmax3d];
  kzvecs = new int[kmax3d];
  memory->create(kvecs, kmax3d, 3, "fresp:kvecs");

  ug = new double[kmax3d];
  memory->create(eg,kmax3d,3,"ewald_fresp:eg");
  memory->create(vg,kmax3d,6,"ewald_fresp:vg");

  sfacrl_qgen = new double[kmax3d];
  sfacim_qgen = new double[kmax3d];
  sfacrl_all_qgen = new double[kmax3d];
  sfacim_all_qgen = new double[kmax3d];
}

/* ---------------------------------------------------------------------
 
------------------------------------------------------------------------ */

void FixFRespEwald::ewald_eik_dot_r_qgen()
{
  int i,k,l,m,n,ic;
  double cstr1,sstr1,cstr2,sstr2,cstr3,sstr3,cstr4,sstr4;
  double sqk,clpm,slpm, arg;

  double **x = atom->x, charge;
  int nlocal = atom->nlocal;

  n = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 3; ic++) {
    sqk = unitk[ic]*unitk[ic];
    if (sqk <= gsqmx) {
      cstr1 = 0.0;
      sstr1 = 0.0;
      for (i = 0; i < nlocal; i++) {
        charge = qgen[types[atom->tag[i] - 1]];
        cs_qgen[0][ic][i] = 1.0;
        sn_qgen[0][ic][i] = 0.0;
        #ifdef __INTEL_MKL__
        arg = unitk[ic] * x[i][ic];
        vdSinCos(1, &arg, &sn_qgen[1][ic][i], &cs_qgen[1][ic][i]);
        #else
        cs_qgen[1][ic][i] = cos(unitk[ic]*x[i][ic]);
        sn_qgen[1][ic][i] = sin(unitk[ic]*x[i][ic]);
        #endif
        cs_qgen[-1][ic][i] = cs_qgen[1][ic][i];
        sn_qgen[-1][ic][i] = -sn_qgen[1][ic][i];
        cstr1 += charge*cs_qgen[1][ic][i];
        sstr1 += charge*sn_qgen[1][ic][i];
      }
      sfacrl_qgen[n] = cstr1;
      sfacim_qgen[n++] = sstr1;
    }
  }

  for (m = 2; m <= kmax; m++) {
    for (ic = 0; ic < 3; ic++) {
      sqk = m*unitk[ic] * m*unitk[ic];
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        for (i = 0; i < nlocal; i++) {
          charge = qgen[types[atom->tag[i] - 1]];
          cs_qgen[m][ic][i] = cs_qgen[m-1][ic][i]*cs_qgen[1][ic][i] -
            sn_qgen[m-1][ic][i]*sn_qgen[1][ic][i];
          sn_qgen[m][ic][i] = sn_qgen[m-1][ic][i]*cs_qgen[1][ic][i] +
            cs_qgen[m-1][ic][i]*sn_qgen[1][ic][i];
          cs_qgen[-m][ic][i] = cs_qgen[m][ic][i];
          sn_qgen[-m][ic][i] = -sn_qgen[m][ic][i];
          cstr1 += charge*cs_qgen[m][ic][i];
          sstr1 += charge*sn_qgen[m][ic][i];
        }
        sfacrl_qgen[n] = cstr1;
        sfacim_qgen[n++] = sstr1;
      }
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          charge = qgen[types[atom->tag[i] - 1]];
          cstr1 += charge*(cs_qgen[k][0][i]*cs_qgen[l][1][i] -
            sn_qgen[k][0][i]*sn_qgen[l][1][i]);
          sstr1 += charge*(sn_qgen[k][0][i]*cs_qgen[l][1][i] +
            cs_qgen[k][0][i]*sn_qgen[l][1][i]);
          cstr2 += charge*(cs_qgen[k][0][i]*cs_qgen[l][1][i] +
            sn_qgen[k][0][i]*sn_qgen[l][1][i]);
          sstr2 += charge*(sn_qgen[k][0][i]*cs_qgen[l][1][i] -
            cs_qgen[k][0][i]*sn_qgen[l][1][i]);
        }
        sfacrl_qgen[n] = cstr1;
        sfacim_qgen[n++] = sstr1;
        sfacrl_qgen[n] = cstr2;
        sfacim_qgen[n++] = sstr2;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (l*unitk[1] * l*unitk[1]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          charge = qgen[types[atom->tag[i] - 1]];
          cstr1 += charge*(cs_qgen[l][1][i]*cs_qgen[m][2][i] -
            sn_qgen[l][1][i]*sn_qgen[m][2][i]);
          sstr1 += charge*(sn_qgen[l][1][i]*cs_qgen[m][2][i] +
            cs_qgen[l][1][i]*sn_qgen[m][2][i]);
          cstr2 += charge*(cs_qgen[l][1][i]*cs_qgen[m][2][i] +
            sn_qgen[l][1][i]*sn_qgen[m][2][i]);
          sstr2 += charge*(sn_qgen[l][1][i]*cs_qgen[m][2][i] -
            cs_qgen[l][1][i]*sn_qgen[m][2][i]);
        }
        sfacrl_qgen[n] = cstr1;
        sfacim_qgen[n++] = sstr1;
        sfacrl_qgen[n] = cstr2;
        sfacim_qgen[n++] = sstr2;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          charge = qgen[types[atom->tag[i] - 1]];
          cstr1 += charge*(cs_qgen[k][0][i]*cs_qgen[m][2][i] -
            sn_qgen[k][0][i]*sn_qgen[m][2][i]);
          sstr1 += charge*(sn_qgen[k][0][i]*cs_qgen[m][2][i] +
            cs_qgen[k][0][i]*sn_qgen[m][2][i]);
          cstr2 += charge*(cs_qgen[k][0][i]*cs_qgen[m][2][i] +
            sn_qgen[k][0][i]*sn_qgen[m][2][i]);
          sstr2 += charge*(sn_qgen[k][0][i]*cs_qgen[m][2][i] -
            cs_qgen[k][0][i]*sn_qgen[m][2][i]);
        }
        sfacrl_qgen[n] = cstr1;
        sfacim_qgen[n++] = sstr1;
        sfacrl_qgen[n] = cstr2;
        sfacim_qgen[n++] = sstr2;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]) +
          (m*unitk[2] * m*unitk[2]);
        if (sqk <= gsqmx) {
          cstr1 = 0.0;
          sstr1 = 0.0;
          cstr2 = 0.0;
          sstr2 = 0.0;
          cstr3 = 0.0;
          sstr3 = 0.0;
          cstr4 = 0.0;
          sstr4 = 0.0;
          for (i = 0; i < nlocal; i++) {
            charge = qgen[types[atom->tag[i] - 1]];
            clpm = cs_qgen[l][1][i]*cs_qgen[m][2][i] -
              sn_qgen[l][1][i]*sn_qgen[m][2][i];
            slpm = sn_qgen[l][1][i]*cs_qgen[m][2][i] +
              cs_qgen[l][1][i]*sn_qgen[m][2][i];
            cstr1 += charge*(cs_qgen[k][0][i]*clpm -
              sn_qgen[k][0][i]*slpm);
            sstr1 += charge*(sn_qgen[k][0][i]*clpm +
              cs_qgen[k][0][i]*slpm);

            clpm = cs_qgen[l][1][i]*cs_qgen[m][2][i] +
              sn_qgen[l][1][i]*sn_qgen[m][2][i];
            slpm = -sn_qgen[l][1][i]*cs_qgen[m][2][i] +
              cs_qgen[l][1][i]*sn_qgen[m][2][i];
            cstr2 += charge*(cs_qgen[k][0][i]*clpm -
              sn_qgen[k][0][i]*slpm);
            sstr2 += charge*(sn_qgen[k][0][i]*clpm +
              cs_qgen[k][0][i]*slpm);

            clpm = cs_qgen[l][1][i]*cs_qgen[m][2][i] +
              sn_qgen[l][1][i]*sn_qgen[m][2][i];
            slpm = sn_qgen[l][1][i]*cs_qgen[m][2][i] -
              cs_qgen[l][1][i]*sn_qgen[m][2][i];
            cstr3 += charge*(cs_qgen[k][0][i]*clpm -
              sn_qgen[k][0][i]*slpm);
            sstr3 += charge*(sn_qgen[k][0][i]*clpm +
              cs_qgen[k][0][i]*slpm);

            clpm = cs_qgen[l][1][i]*cs_qgen[m][2][i] -
              sn_qgen[l][1][i]*sn_qgen[m][2][i];
            slpm = -sn_qgen[l][1][i]*cs_qgen[m][2][i] -
              cs_qgen[l][1][i]*sn_qgen[m][2][i];
            cstr4 += charge*(cs_qgen[k][0][i]*clpm -
              sn_qgen[k][0][i]*slpm);
            sstr4 += charge*(sn_qgen[k][0][i]*clpm +
              cs_qgen[k][0][i]*slpm);
          }
          sfacrl_qgen[n] = cstr1;
          sfacim_qgen[n++] = sstr1;
          sfacrl_qgen[n] = cstr2;
          sfacim_qgen[n++] = sstr2;
          sfacrl_qgen[n] = cstr3;
          sfacim_qgen[n++] = sstr3;
          sfacrl_qgen[n] = cstr4;
          sfacim_qgen[n++] = sstr4;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixFRespEwald::ewald_eik_dot_r_triclinic_qgen()
{
  int i,k,l,m,n,ic;
  double cstr1,sstr1;
  double sqk,clpm,slpm, arg;

  double **x = atom->x;
  int nlocal = atom->nlocal;

  double unitk_lamda[3];

  double max_kvecs[3];
  max_kvecs[0] = kxmax;
  max_kvecs[1] = kymax;
  max_kvecs[2] = kzmax;

  // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 3; ic++) {
    unitk_lamda[0] = 0.0;
    unitk_lamda[1] = 0.0;
    unitk_lamda[2] = 0.0;
    unitk_lamda[ic] = 2.0*MathConst::MY_PI;
    ewald_x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
    sqk = unitk_lamda[ic]*unitk_lamda[ic];
    if (sqk <= gsqmx) {
      for (i = 0; i < nlocal; i++) {
        cs_qgen[0][ic][i] = 1.0;
        sn_qgen[0][ic][i] = 0.0;
        #ifdef __INTEL_MKL__
        arg = unitk_lamda[0]*x[i][0] + unitk_lamda[1]*x[i][1] +
          unitk_lamda[2]*x[i][2];
        vdSinCos(1, &arg, &sn_qgen[1][ic][i], &cs_qgen[1][ic][i]);
        #else
        cs_qgen[1][ic][i] = cos(unitk_lamda[0]*x[i][0] +
          unitk_lamda[1]*x[i][1] + unitk_lamda[2]*x[i][2]);
        sn_qgen[1][ic][i] = sin(unitk_lamda[0]*x[i][0] +
          unitk_lamda[1]*x[i][1] + unitk_lamda[2]*x[i][2]);
        #endif
        cs_qgen[-1][ic][i] = cs_qgen[1][ic][i];
        sn_qgen[-1][ic][i] = -sn_qgen[1][ic][i];
      }
    }
  }

  for (ic = 0; ic < 3; ic++) {
    for (m = 2; m <= max_kvecs[ic]; m++) {
      unitk_lamda[0] = 0.0;
      unitk_lamda[1] = 0.0;
      unitk_lamda[2] = 0.0;
      unitk_lamda[ic] = 2.0*MathConst::MY_PI*m;
      ewald_x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
      sqk = unitk_lamda[ic]*unitk_lamda[ic];
      for (i = 0; i < nlocal; i++) {
        cs_qgen[m][ic][i] = cs_qgen[m-1][ic][i]*cs_qgen[1][ic][i] -
          sn_qgen[m-1][ic][i]*sn_qgen[1][ic][i];
        sn_qgen[m][ic][i] = sn_qgen[m-1][ic][i]*cs_qgen[1][ic][i] +
          cs_qgen[m-1][ic][i]*sn_qgen[1][ic][i];
        cs_qgen[-m][ic][i] = cs_qgen[m][ic][i];
        sn_qgen[-m][ic][i] = -sn_qgen[m][ic][i];
      }
    }
  }

  for (n = 0; n < kcount; n++) {
    k = kxvecs[n];
    l = kyvecs[n];
    m = kzvecs[n];
    cstr1 = 0.0;
    sstr1 = 0.0;
    for (i = 0; i < nlocal; i++) {
      clpm = cs_qgen[l][1][i]*cs_qgen[m][2][i] -
        sn_qgen[l][1][i]*sn_qgen[m][2][i];
      slpm = sn_qgen[l][1][i]*cs_qgen[m][2][i] +
        cs_qgen[l][1][i]*sn_qgen[m][2][i];
      cstr1 += qgen[types[atom->tag[i] - 1]] * (cs_qgen[k][0][i] *
        clpm - sn_qgen[k][0][i]*slpm);
      sstr1 += qgen[types[atom->tag[i] - 1]] * (sn_qgen[k][0][i] *
        clpm + cs_qgen[k][0][i]*slpm);
    }
    sfacrl_qgen[n] = cstr1;
    sfacim_qgen[n] = sstr1;
  }
}

/* ----------------------------------------------------------------------
   deallocate memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void FixFRespEwald::ewald_deallocate()
{
  delete [] kxvecs;
  delete [] kyvecs;
  delete [] kzvecs;
  memory->destroy(kvecs);

  delete [] ug;
  memory->destroy(eg);
  memory->destroy(vg);

  delete [] sfacrl_qgen;
  delete [] sfacim_qgen;
  delete [] sfacrl_all_qgen;
  delete [] sfacim_all_qgen;
}

/* ----------------------------------------------------------------------
   compute RMS accuracy for a dimension
------------------------------------------------------------------------- */

double FixFRespEwald::ewald_rms(int km, double prd, bigint natoms, double q2)
{
  double value = 2.0*q2*g_ewald/prd *
    sqrt(1.0/(MathConst::MY_PI*km*natoms)) *
    exp(-MathConst::MY_PI*MathConst::MY_PI*km*km/(g_ewald*g_ewald*prd*prd));

  return value;
}

/* ---------------------------------------------------------------------- */

void FixFRespEwald::ewald_init()
{
  //Extract short-range Coulombic cutoff from pair style

  triclinic = domain->triclinic;

  //Compute qsum & qsqsum and warn if not charge-neutral

  scale = 1.0;
  ewald_qsum_qsq();
  natoms_original = atom->natoms;

  //Set accuracy (force units) from kspace->accuracy_relative or
  //kspace->accuracy_absolute
  if (force->kspace->accuracy_absolute >= 0.0)
    accuracy = force->kspace->accuracy_absolute;
  else accuracy = force->kspace->accuracy_relative *
    force->kspace->two_charge_force;

  //If not specified in input, g_ewald is put equal to main one.
  if (gewaldflag) g_ewald = force->kspace->g_ewald;

  ewald_setup();
}

/* ----------------------------------------------------------------------
   convert box coords vector to transposed triclinic lamda (0-1) coords
   vector, lamda = [(H^-1)^T] * v, does not preserve vector magnitude
   v and lamda can point to same 3-vector
------------------------------------------------------------------------- */

void FixFRespEwald::ewald_x2lamdaT(double *v, double *lamda)
{
  double *h_inv = domain->h_inv;
  double lamda_tmp[3];

  lamda_tmp[0] = h_inv[0]*v[0];
  lamda_tmp[1] = h_inv[5]*v[0] + h_inv[1]*v[1];
  lamda_tmp[2] = h_inv[4]*v[0] + h_inv[3]*v[1] + h_inv[2]*v[2];

  lamda[0] = lamda_tmp[0];
  lamda[1] = lamda_tmp[1];
  lamda[2] = lamda_tmp[2];
}

/* ----------------------------------------------------------------------
   convert lamda (0-1) coords vector to transposed box coords vector
   lamda = (H^T) * v, does not preserve vector magnitude
   v and lamda can point to same 3-vector
------------------------------------------------------------------------- */

void FixFRespEwald::ewald_lamda2xT(double *lamda, double *v)
{
  double h[6];
  h[0] = domain->h[0];
  h[1] = domain->h[1];
  h[2] = domain->h[2];
  h[3] = fabs(domain->h[3]);
  h[4] = fabs(domain->h[4]);
  h[5] = fabs(domain->h[5]);
  double v_tmp[3];

  v_tmp[0] = h[0]*lamda[0];
  v_tmp[1] = h[5]*lamda[0] + h[1]*lamda[1];
  v_tmp[2] = h[4]*lamda[0] + h[3]*lamda[1] + h[2]*lamda[2];

  v[0] = v_tmp[0];
  v[1] = v_tmp[1];
  v[2] = v_tmp[2];
}


/* ----------------------------------------------------------------------
   adjust Ewald coeffs, called initially and whenever volume has changed
------------------------------------------------------------------------- */

void FixFRespEwald::ewald_setup()
{
  //Volume-dependent factors

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;

  //Adjustment of z dimension for 2d slab Ewald
  //3d Ewald just uses zprd since slab_volfactor = 1.0

  double zprd_slab = zprd * force->kspace->slab_volfactor;
  volume = xprd * yprd * zprd_slab;

  unitk[0] = 2.0*MathConst::MY_PI/xprd;
  unitk[1] = 2.0*MathConst::MY_PI/yprd;
  unitk[2] = 2.0*MathConst::MY_PI/zprd_slab;

  int kmax_old = kmax;
  
  if (kewaldflag == 1) {

  //Determine kmax
  //function of current box size, accuracy, G_ewald (short-range cutoff)

    bigint natoms = atom->natoms;
    double err;
    kxmax = 1;
    kymax = 1;
    kzmax = 1;
   
    err = ewald_rms(kxmax,xprd,natoms,q2);
    while (err > accuracy) {
      kxmax++;
      err = ewald_rms(kxmax,xprd,natoms,q2);
    }
   
    err = ewald_rms(kymax,yprd,natoms,q2);
    while (err > accuracy) {
      kymax++;
      err = ewald_rms(kymax,yprd,natoms,q2);
    }
   
    err = ewald_rms(kzmax,zprd_slab,natoms,q2);
    while (err > accuracy) {
      kzmax++;
      err = ewald_rms(kzmax,zprd_slab,natoms,q2);
    }
   
    kmax = MAX(kxmax,kymax);
    kmax = MAX(kmax,kzmax);
    kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;
   
    double wmaotsetung = unitk[0]*unitk[0]*kxmax*kxmax;
    double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
    double gsqzmx = unitk[2]*unitk[2]*kzmax*kzmax;
    gsqmx = MAX(wmaotsetung,gsqymx);
    gsqmx = MAX(gsqmx,gsqzmx);
   
    kxmax_orig = kxmax;
    kymax_orig = kymax;
    kzmax_orig = kzmax;
   
    //Scale lattice vectors for triclinic skew
   
    if (triclinic) {
      double tmp[3];
      tmp[0] = kxmax/xprd;
      tmp[1] = kymax/yprd;
      tmp[2] = kzmax/zprd;
      ewald_lamda2xT(&tmp[0],&tmp[0]);
      kxmax = MAX(1,static_cast<int>(tmp[0]));
      kymax = MAX(1,static_cast<int>(tmp[1]));
      kzmax = MAX(1,static_cast<int>(tmp[2]));
   
      kmax = MAX(kxmax,kymax);
      kmax = MAX(kmax,kzmax);
      kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;
    }

  } else {

    kxmax_orig = kxmax;
    kymax_orig = kymax;
    kzmax_orig = kzmax;

    kmax = MAX(kxmax,kymax);
    kmax = MAX(kmax,kzmax);
    kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;

    double wmaotsetung = unitk[0]*unitk[0]*kxmax*kxmax;
    double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
    double gsqzmx = unitk[2]*unitk[2]*kzmax*kzmax;
    gsqmx = MAX(wmaotsetung,gsqymx);
    gsqmx = MAX(gsqmx,gsqzmx);
  }

  gsqmx *= 1.00001;

  //If size has grown, reallocate k-dependent and nlocal-dependent arrays

  if (kmax > kmax_old) {
    ewald_deallocate();
    ewald_allocate();
    
    memory->destroy(ek);
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    memory->destroy3d_offset(cs_qgen,-kmax_created);
    memory->destroy3d_offset(sn_qgen,-kmax_created);
    nmax = atom->nmax;
    memory->create(ek,nmax,3,"ewald_fresp:ek");
    memory->create3d_offset(cs,-kmax,kmax,3,nmax,"ewald_fresp:cs");
    memory->create3d_offset(sn,-kmax,kmax,3,nmax,"ewald_fresp:sn");
    memory->create3d_offset(cs_qgen,-kmax,kmax,3,nmax,"ewald_fresp:cs_qgen");
    memory->create3d_offset(sn_qgen,-kmax,kmax,3,nmax,"ewald_fresp:sn_qgen");
    kmax_created = kmax;
  }

  //Pre-compute Ewald coefficients

  if (triclinic == 0)
    ewald_coeffs();
  else
    ewald_coeffs_triclinic();

  //Arrays used by BLAS functions in reciprocal space part of q_update_Efield
  #ifdef __INTEL_MKL__
  memory->grow(bondvskprod_vec, kcount, "fresp:bondkprod_vec");
  memory->grow(xmkprod_vec, kcount, "fresp:xmkprod_vec");
  memory->grow(Im_xm_vec, kcount, "fresp:Im_xm_vec");
  memory->grow(Re_xm_vec, kcount, "fresp:Re_xm_vec");
  memory->grow(Im_prod_vec, kcount, "fresp:Im_prod_vec");
  memory->grow(Re_prod_vec, kcount, "fresp:Re_prod_vec");
  memory->grow(tmp1, kcount, "fresp:tmp1");
  memory->grow(tmp2, kcount, "fresp:tmp2");
  memory->grow(appo2Re_pref_vec, kcount, "fresp:appo2Re_pref_vec");
  memory->grow(appo2Im_pref_vec, kcount, "fresp:appo2Im_pref_vec");
  #endif
}

/* ---------------------------------------------------------------------- */

void FixFRespEwald::ewald_structure_factor()
{
  //Update qsum and qsqsum at each step
  ewald_qsum_qsq();

  if (atom->natoms != natoms_original) {
    natoms_original = atom->natoms;
  }

  //Extend size of per-atom arrays if necessary
  if (atom->nmax > nmax) {
    memory->destroy(ek);
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    memory->destroy3d_offset(cs_qgen,-kmax_created);
    memory->destroy3d_offset(sn_qgen,-kmax_created);
    nmax = atom->nmax;
    memory->create(ek,nmax,3,"ewald_fresp:ek");
    memory->create3d_offset(cs,-kmax,kmax,3,nmax,"ewald_fresp:cs");
    memory->create3d_offset(sn,-kmax,kmax,3,nmax,"ewald_fresp:sn");
    memory->create3d_offset(cs_qgen,-kmax,kmax,3,nmax,"ewald_fresp:cs_qgen");
    memory->create3d_offset(sn_qgen,-kmax,kmax,3,nmax,"ewald_fresp:sn_qgen");
    kmax_created = kmax;
  }

  //Partial structure factors on each processor
  //total structure factor by summing over procs

  if (triclinic == 0) ewald_eik_dot_r_qgen();
  else ewald_eik_dot_r_triclinic_qgen();

  MPI_Allreduce(sfacrl_qgen,sfacrl_all_qgen,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim_qgen,sfacim_all_qgen,kcount,MPI_DOUBLE,MPI_SUM,world);
}

/* ---------------------------------------------------------------------- 
   charges fluctuation due to electric field on bonds
------------------------------------------------------------------------ */

void FixFRespEwald::q_update_Efield_bond()
{
  double xm[3], **x = atom->x, rvml, rvminv, rvminvsq, rvminvcu;
  double rvm[3], rvmvs[3], r0, bondvl, bondvinv, wmarx; 
  bigint atom1, atom2, center, global_center, global_atom1, global_atom2;
  bigint molecule;
  int atom1_t, atom2_t, i, bond, atom1_pos, atom2_pos, molflag;
  double bondv[3], bondvs[3], E[3], E_R[3], E_K[3], Eparallel, ra1[3], ra2[3];
  double grij, expm2, Im_prod, Re_prod, Im_xm, Re_xm, bondkprod, bondvskprod;
  double first_first_half[3], first_second_half[3], erfc, rvmlsq;
  double cutoff3_sqr, arg, first_half[3], second_half[3], appo2Re_pref;
  double appo2Im_pref, kvec[3], minus_square_grij;
  double dEr_par, E_Rpar, bondrvmprod, E_Rpar_red, bondvs_red[3], damping;
  double ddamping[3], pref_first_half, pref_second_half;

  cutoff3_sqr = cutoff3 * cutoff3;

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
    #ifdef __INTEL_MKL__
    bondvl = cblas_dnrm2(3, bondv, 1);
    #else
    bondvl = MathExtra::len3(bondv);
    #endif
    bondvinv = 1.0 / bondvl;

    //This check is here because, if false, bondv has already been calculated
    //and can be used for charge variation due to bond stretching
    if (Efieldflag && qsqsum > 0.0) {

      double E_R_perbond = 0.0, E_B_perbond = 0.0, E_K_perbond = 0.0;

      #ifdef __INTEL_MKL__
      cblas_dcopy(3, x[atom1], 1, xm, 1);
      vdAdd(3, x[atom2], xm, xm);
      cblas_dscal(3, 0.5, xm, 1);
        cblas_dcopy(3, bondv, 1, bondvs, 1);
        cblas_dscal(3, bondvinv, bondvs, 1);
      #else
      xm[0] = (x[atom1][0] + x[atom2][0]) * 0.5;
      xm[1] = (x[atom1][1] + x[atom2][1]) * 0.5;
      xm[2] = (x[atom1][2] + x[atom2][2]) * 0.5;
      MathExtra::normalize3(bondv, bondvs);
      #endif
      domain->minimum_image(xm[0], xm[1], xm[2]);

      atom1_pos = bond_extremes_pos[bond][0];
      atom2_pos = bond_extremes_pos[bond][1];

      //This cycle over all the atoms is absolutely needed
      for (i = 0; i < dEr_indexes[bond][0][0]; i++) dEr_vals[bond][i][0] =
        dEr_vals[bond][i][1] = dEr_vals[bond][i][2] = 0.0;

      //The cycle is done over all the atoms contained in the 
      //Verlet list of bond. Calculate E and its gradient
      //in direct space with bonded correction due to Ewald summation too
      for (i = 0; i < dEr_indexes[bond][0][0]; i++) {
        center = dEr_indexes[bond][i + 1][0];
        global_center = atom->tag[center];
        //molflag = 1 if center is in the same molecule, 0 otherwise
        molflag = atom->molecule[center] == molecule;
        #ifdef __INTEL_MKL__
        vdSub(3, xm, x[center], rvm);
        domain->minimum_image(rvm[0], rvm[1], rvm[2]);
        
        //ra1 and ra2 are needed for force calculation
        vdSub(3, x[atom1], x[center], ra1);
        vdSub(3, x[center], x[atom2], ra2);

        //Dot product with itself is used in order to obtain square lenght
        rvmlsq = cblas_ddot(3, rvm, 1, rvm, 1);
        #else
        rvm[0] = xm[0] - x[center][0];
        rvm[1] = xm[1] - x[center][1];
        rvm[2] = xm[2] - x[center][2];
        domain->minimum_image(rvm[0], rvm[1], rvm[2]);

        //ra1 and ra2 are needed for force calculation
        ra1[0] = x[atom1][0] - x[center][0];
        ra1[1] = x[atom1][1] - x[center][1];
        ra1[2] = x[atom1][2] - x[center][2];
        //domain->minimum_image(ra1[0], ra1[1], ra1[2]);

        ra2[0] = x[center][0] - x[atom2][0];
        ra2[1] = x[center][1] - x[atom2][1];
        ra2[2] = x[center][2] - x[atom2][2];
        //domain->minimum_image(ra2[0], ra2[1], ra2[2]);

        rvmlsq = MathExtra::lensq3(rvm);
        #endif

        if (rvmlsq > cutoff3_sqr) { 
          //In order not to cycle over this atom in pre_reverse function
          dEr_indexes[bond][i + 1][1] = (tagint) -1;
          continue;
        }
        
        dEr_indexes[bond][i + 1][1] = (tagint) 1;

        bondrvmprod = MathExtra::dot3(bondv, rvm);
        #ifdef __INTEL_MKL__
        bondrvmprod = cblas_ddot(3, bondv, 1, rvm, 1);
        cblas_dcopy(3, rvm, 1, E_R, 1);
        rvml = cblas_dnrm2(3, rvm, 1);
        rvminv = 1.0 / rvml;
        cblas_dcopy(3, rvm, 1, rvmvs, 1);
        cblas_dscal(3, rvminv, rvmvs, 1);       
        cblas_dcopy(3, bondvs, 1, bondvs_red, 1);
        cblas_dscal(3, bondvinv, bondvs_red, 1);       
        grij = g_ewald * rvml;
        minus_square_grij = -grij * grij;
        vdExp(1, &minus_square_grij, &expm2);
        vdErfc(1, &grij, &erfc);
        #else
        bondrvmprod = MathExtra::dot3(bondv, rvm);
        MathExtra::copy3(rvm, E_R);
        rvml = MathExtra::len3(rvm);
        rvminv = 1.0 / rvml;
        MathExtra::copy3(rvm, rvmvs);
        MathExtra::scale3(rvminv, rvmvs);
        MathExtra::copy3(bondvs, bondvs_red);
        MathExtra::scale3(bondvinv, bondvs_red);
        grij = g_ewald * rvml;
        expm2 = expmsq(grij);
        erfc = expm2 * my_erfcx(grij);
        #endif
        rvminvsq = rvminv * rvminv;
        rvminvcu = rvminvsq * rvminv;
        if (!molflag) {
          E_Rpar = qgen[types[global_center - 1]] * rvminvsq *
            (erfc * rvminv + TWO_OVER_SQPI * g_ewald * expm2);
          dEr_par = -qgen[types[global_center - 1]] * rvminv * bondvinv *
            (expm2 * TWO_OVER_SQPI * g_ewald * (3.0 * rvminvsq + 2.0 *
            g_ewald * g_ewald) + 3.0 * erfc * rvminvcu);
          if (rvml < cutoff2) {
            damping = Efield_damping(dampflag, rvml, cutoff1, cutoff2);
            //Damping function derivative is initialized as rvmvs
            #ifdef __INTEL_MKL__
            cblas_dcopy(3, rvmvs, 1, ddamping, 1);
            if (dampflag == 0)
              cblas_dscal(3, ((rvml - cutoff2) / (cutoff1 * cutoff1)) *
              bondrvmprod * bondvinv * damping * E_Rpar, ddamping, 1);
            else if (dampflag == 1) 
              cblas_dscal(3, -MathConst::MY_PI / (cutoff2 - cutoff1) *
              sqrt(damping * (1.0 - damping)) * bondrvmprod * bondvinv *
              E_Rpar, ddamping, 1);
            #else
            MathExtra::copy3(rvmvs, ddamping);
            if (dampflag == 0)
              MathExtra::scale3(((rvml - cutoff2) / (cutoff1 * cutoff1)) *
              bondrvmprod * bondvinv * E_Rpar, ddamping);
            else if (dampflag == 1)
              MathExtra::scale3(-MathConst::MY_PI / (cutoff2 - cutoff1) *
              sqrt(damping * (1.0 - damping)) * bondrvmprod * bondvinv *
              E_Rpar, ddamping);
            #endif
            //E_Rpar and dEr_par are scaled
            E_Rpar *= damping;
            dEr_par *= damping;
            #ifdef __INTEL_MKL__
            vdAdd(3, ddamping, dEr_vals[bond][i], dEr_vals[bond][i]);
            //Derivative of damping function for atom1 and atom2 is calculated
            //(simply half the opposite of the previous one) and multiplied
            //times undamped Efield
            cblas_dscal(3, -0.5, ddamping, 1);
            vdAdd(3, ddamping, dEr_vals[bond][atom1_pos],
              dEr_vals[bond][atom1_pos]);
            vdAdd(3, ddamping, dEr_vals[bond][atom2_pos],
              dEr_vals[bond][atom2_pos]);
            #else
            MathExtra::add3(ddamping, dEr_vals[bond][i], dEr_vals[bond][i]);
            //Derivative of damping function for atom1 and atom2 is calculated
            //(simply half the opposite of the previous one) and multiplied
            //times undamped Efield
            MathExtra::scale3(-0.5, ddamping);
            MathExtra::add3(ddamping, dEr_vals[bond][atom1_pos],
              dEr_vals[bond][atom1_pos]);
            MathExtra::add3(ddamping, dEr_vals[bond][atom2_pos],
              dEr_vals[bond][atom2_pos]);
            #endif
          }
          #ifdef __INTEL_MKL__
          cblas_dscal(3, E_Rpar, E_R, 1);
          E_R_perbond += cblas_ddot(3, E_R, 1, bondvs, 1);
          #else
          MathExtra::scale3(E_Rpar, E_R);
          E_R_perbond += MathExtra::dot3(E_R, bondvs);
          #endif
        }
        //Efield is not damped
        else { 
          E_Rpar = qgen[types[global_center - 1]] * rvminvsq *
            (TWO_OVER_SQPI * g_ewald * expm2 - (1.0 - erfc) * rvminv);
          dEr_par = qgen[types[global_center - 1]] * rvminv * bondvinv *
            (3.0 * rvminvcu * (1.0 - erfc) - TWO_OVER_SQPI * g_ewald * expm2 *
            (3.0 * rvminvsq + 2.0 * g_ewald * g_ewald)); 
          #ifdef __INTEL_MKL__
          cblas_dscal(3, E_Rpar, E_R, 1);
          E_B_perbond += cblas_ddot(3, E_R, 1, bondvs, 1);
          #else
          MathExtra::scale3(E_Rpar, E_R);
          E_B_perbond += MathExtra::dot3(E_R, bondvs);
          #endif
        }

        E_Rpar_red = E_Rpar * bondvinv;
        first_half[0] = -bondrvmprod * dEr_par;
        MathExtra::add3(E, E_R, E);
        #ifdef __INTEL_MKL__
        //dEr_vals/d(dEr_indexes[i][0]) components are calculated
        cblas_daxpy(3, first_half[0], rvmvs, 1, dEr_vals[bond][i], 1);
        cblas_daxpy(3, -E_Rpar, bondvs, 1, dEr_vals[bond][i], 1);

        cblas_dcopy(3, bondvs_red, 1, first_first_half, 1);
        cblas_dscal(3, E_Rpar_red, first_first_half, 1);

        cblas_dcopy(3,rvmvs, 1, first_second_half, 1);
        cblas_dscal(3, 0.5 * dEr_par, first_second_half, 1);

        //dEr_vals/d(atom1) components are calculated
        cblas_daxpy(3, -bondrvmprod, first_first_half, 1,
          dEr_vals[bond][atom1_pos], 1);
        cblas_daxpy(3, bondrvmprod, first_second_half, 1,
          dEr_vals[bond][atom1_pos], 1);
        cblas_daxpy(3, E_Rpar_red, ra1, 1,
          dEr_vals[bond][atom1_pos], 1);

        //dEr_vals/d(atom2) components are calculated
        cblas_daxpy(3, bondrvmprod, first_first_half, 1,
          dEr_vals[bond][atom2_pos], 1);
        cblas_daxpy(3, bondrvmprod, first_second_half, 1,
          dEr_vals[bond][atom2_pos], 1);
        cblas_daxpy(3, E_Rpar_red, ra2, 1,
          dEr_vals[bond][atom2_pos], 1);
        #else
        //dEr_vals/d(dEr_indexes[i][0]) components are calculated
        dEr_vals[bond][i][0] += first_half[0] * rvmvs[0] - E_Rpar * bondvs[0];
        dEr_vals[bond][i][1] += first_half[0] * rvmvs[1] - E_Rpar * bondvs[1];
        dEr_vals[bond][i][2] += first_half[0] * rvmvs[2] - E_Rpar * bondvs[2];
   
        first_first_half[0] = bondvs_red[0] * E_Rpar_red;
        first_first_half[1] = bondvs_red[1] * E_Rpar_red;
        first_first_half[2] = bondvs_red[2] * E_Rpar_red;
   
        first_second_half[0] = 0.5 * dEr_par * rvmvs[0];
        first_second_half[1] = 0.5 * dEr_par * rvmvs[1];
        first_second_half[2] = 0.5 * dEr_par * rvmvs[2];
   
        //dEr_vals/d(atom1) components are calculated
        dEr_vals[bond][atom1_pos][0] += (-first_first_half[0] +
          first_second_half[0]) * bondrvmprod + E_Rpar_red * ra1[0];
        dEr_vals[bond][atom1_pos][1] += (-first_first_half[1] +
          first_second_half[1]) * bondrvmprod + E_Rpar_red * ra1[1];
        dEr_vals[bond][atom1_pos][2] += (-first_first_half[2] +
          first_second_half[2]) * bondrvmprod + E_Rpar_red * ra1[2];
   
        //dEr_vals/d(atom2) components are calculated
        dEr_vals[bond][atom2_pos][0] += (first_first_half[0] +
          first_second_half[0]) * bondrvmprod + E_Rpar_red * ra2[0];
        dEr_vals[bond][atom2_pos][1] += (first_first_half[1] +
          first_second_half[1]) * bondrvmprod + E_Rpar_red * ra2[1];
        dEr_vals[bond][atom2_pos][2] += (first_first_half[2] +
          first_second_half[2]) * bondrvmprod + E_Rpar_red * ra2[2];
        #endif
      }

      //Calculate E and its gradient (only wrt atom1 and atom2)
      //in reciprocal space
      for (i = 0; i < kcount; i++) {
        kvec[0] = kvecs[i][0];
        kvec[1] = kvecs[i][1];
        kvec[2] = kvecs[i][2];

        //E is made equal to current k vector times all components except
        //structure factor product
        MathExtra::copy3(eg[i], E_K);

        bondkprod = MathExtra::dot3(bondv, kvec);
        bondvskprod = bondkprod * bondvinv;
        arg = MathExtra::dot3(xm, kvec);
        Im_xm = sin(arg);
        Re_xm = cos(arg);
        Im_prod = Im_xm * sfacrl_all_qgen[i] - Re_xm * sfacim_all_qgen[i];
        Re_prod = Re_xm * sfacrl_all_qgen[i] + Im_xm * sfacim_all_qgen[i];
        MathExtra::scale3(Im_prod, E_K);
        MathExtra::add3(E, E_K, E);
        appo2Re_pref = bondvskprod * Re_xm;
        appo2Im_pref = bondvskprod * Im_xm;
        MathExtra::copy3(eg[i], appo2[bond][i]);
        MathExtra::copy3(eg[i], appo2[bond][i] + 3);
        MathExtra::scale3(appo2Re_pref, appo2[bond][i]);
        MathExtra::scale3(appo2Im_pref, appo2[bond][i] + 3);
        pref_first_half = ug[i] * bondvskprod * Re_prod;
        pref_second_half = 2.0 * ug[i] * Im_prod * bondvinv;
        E_K_perbond += MathExtra::dot3(E_K, bondvs);

        first_half[0] = pref_first_half * kvec[0];
        first_half[1] = pref_first_half * kvec[1];
        first_half[2] = pref_first_half * kvec[2];

        second_half[0] = pref_second_half * (bondvskprod * bondvs[0] - kvec[0]);
        second_half[1] = pref_second_half * (bondvskprod * bondvs[1] - kvec[1]);
        second_half[2] = pref_second_half * (bondvskprod * bondvs[2] - kvec[2]);

        //dEr_vals/d(atom1) components are calculated
        dEr_vals[bond][atom1_pos][0] += first_half[0] - second_half[0];
        dEr_vals[bond][atom1_pos][1] += first_half[1] - second_half[1];
        dEr_vals[bond][atom1_pos][2] += first_half[2] - second_half[2];

        //dEr_vals/d(atom2) components are calculated
        dEr_vals[bond][atom2_pos][0] += first_half[0] + second_half[0];
        dEr_vals[bond][atom2_pos][1] += first_half[1] + second_half[1];
        dEr_vals[bond][atom2_pos][2] += first_half[2] + second_half[2];
      }

    //Calculate component of E along bond
    Eparallel = MathExtra::dot3(E, bondvs);

    if (printEfieldflag) fprintf(stderr, BIGINT_FORMAT " " BIGINT_FORMAT
      " %.14lf %.14lf %.14lf %.14lf\n", global_atom1, global_atom2, E_R_perbond,
      E_B_perbond, E_K_perbond, Eparallel);

      deltaq_update_Efield(molecule, atom1_t, atom2_t, Eparallel);
    }

    if (bondflag) {
      r0 = force->bond->equilibrium_distance(neighbor->bondlist[bond][2]);
      wmarx = bondvl - r0;

      db_vals[bond][0] = bondv[0] * bondvinv;
      db_vals[bond][1] = bondv[1] * bondvinv;
      db_vals[bond][2] = bondv[2] * bondvinv;

      deltaq_update_bond(molecule, atom1_t, atom2_t, wmarx);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixFRespEwald::setup_pre_force(int vflag)
{
  int i, j, bond = 0;
  bigint atom1;

  //Using count_total_bonds(), bonds "freezed" by SHAKE are considered too,
  //otherwise than using nbondlist
  nbond_old = count_total_bonds();

  //If nmax has changed, deltaq and erfc_erf_arr are resized.
  if (atom->nmax != nmax) memory->grow(deltaq, atom->nmax, "fresp:deltaq");

  //K-space solver ewald is initialized.
  ewald_init();

  //An array of nbond double** is allocated in order to store the values of
  //derivatives of E_R * bond unit vector
  //An array of nbond tagint* is allocated in order to store the indexes of
  //atoms wrt the derivatives of dEr_vals are done
  dEr_vals = (double***) calloc(nbond_old, sizeof(double**));
  if (dEr_vals == NULL) {
    char str[128];
    sprintf(str,"Failed to allocate " BIGINT_FORMAT " bytes for array \
      fresp:dEr_vals",
            nbond_old * sizeof(double**));
    error->one(FLERR,str);
  }
  dEr_indexes = (tagint***) calloc(nbond_old, sizeof(tagint**));
  if (dEr_indexes == NULL) {
    char str[128];
    sprintf(str,"Failed to allocate " BIGINT_FORMAT " bytes for array \
      fresp:dEr_indexes",
            nbond_old * sizeof(double**));
    error->one(FLERR,str);
  }
  memory->create(bond_extremes_pos, nbond_old, 2, "fresp:bond_extremes_pos");

  for (i = 0; i < nbond_old; i++) {
    dEr_vals[i] = NULL;
    dEr_indexes[i] = NULL;
  }

  //Building of new neighbor lists needed by F-RESP
  neighbor->build_one(list);

  for (i = 0; i < atom->nlocal; i++) {
    for (j = 0; j < atom->num_bond[i]; j++) {
      atom1 = atom->map(atom->bond_atom[i][j]);
      atom1 = domain->closest_image(i, (int)atom1);
      if (force->newton_bond || i < atom1)
        build_bond_Verlet_list(bond++, i, atom1);
    }
  }

  memory->create(appo2, nbond_old, kcount, 6, "fresp:appo2");

  memory->create(appo3, kcount, 6, "fresp:appo3");

  pre_force(vflag); 
}
 
/* ---------------------------------------------------------------------- */

void FixFRespEwald::post_neighbor()
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
    memory->grow(bond_extremes_pos, nbond_old, 2, "fresp:bond_extremes_pos");
    free(dEr_vals);
    free(dEr_indexes);
    dEr_vals = (double***) calloc(nbond_old, sizeof(double**));
    dEr_indexes = (tagint***) calloc(nbond_old, sizeof(tagint**));
    for (i = 0; i < nbond_old; i++) {
      dEr_vals[i] = NULL;
      dEr_indexes[i] = NULL;
    }
    memory->destroy(appo2);
    memory->create(appo2, nbond_old, kcount, 6, "fresp:appo2");
  }

  //Building of new neighbor lists needed by F-RESP
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

void FixFRespEwald::pre_force(int vflag)
{
  if (update->ntimestep % nevery) return;

  int i;

  if (atom->nmax != nmax) memory->grow(deltaq, atom->nmax, "fresp:deltaq");

  //Activates calculation of kspace->eatom and pair->eatomcoul at each step
  update->eflag_atom = update->eflag_global = update->ntimestep;
  pe->addstep(update->ntimestep + 1);

  //deltaq array is cleared
  for (i = 0; i < atom->nmax; i++) deltaq[i] = 0.0;

  ewald_structure_factor();

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

void FixFRespEwald::pre_reverse(int eflag, int vflag)
{
  int bond, i, j, k, atom1_t, atom2_t, center_t;
  tagint der_atom, global_atom1, global_atom2;
  tagint global_center, center;
  bigint molecule;
  double alpha, Re_xi, Im_xi, arg, alpha_tot_pot;
  static const double prefac =  2.0 * force->kspace->g_ewald /
    MathConst::MY_PIS;
  double v[6], unwrap[3], freal[3], premul[3], gen_q, partial[3];
  double kb, kb_tot_pot, bondv[3], bondvinv;
  int atom1, atom2, atom1_pos, atom2_pos;

  if (update->ntimestep % nevery) return;

  //Energy and virial setup
  //Virial calculation still not correctly implemented!!
  if (vflag) v_setup(vflag);
  else evflag = 0;

  //Communicate force->kspace->eatom for neighboring atoms.
  pack_flag = 1;
  comm->forward_comm_fix(this);
  //Communicate reverse and forward force->pair->eatomcoul
  pack_flag = 3;
  comm->reverse_comm_fix(this);
  comm->forward_comm_fix(this);

  if (Efieldflag || bondflag) {
    //appo3 is filled with zeros
    for (k = 0; k < kcount; k++) appo3[k][0] = appo3[k][1] = appo3[k][2] =
      appo3[k][3] = appo3[k][4] = appo3[k][5] = 0.0;

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
        #ifdef __INTEL_MKL__
        bondvinv = 1.0 / cblas_dnrm2(3, bondv, 1);
        #else
        bondvinv = 1.0 / MathExtra::len3(bondv);
        #endif
      }
      for (i = 1; i <= mol_map[molecule - 1][0]; i++) {
        global_center = mol_map[molecule - 1][i];
        center = atom->map(global_center);
        center_t = types[global_center - 1];
        alpha = k_Efield[atom1_t][atom2_t][center_t];
        alpha_tot_pot = alpha * (2.0 * (force->pair->eatomcoul[center] +
          force->kspace->eatom[center]) / atom->q[center]);
        //Virial calculation still not correctly implemented!!
        //alpha_real_space_pot = alpha * (erfc_erf_arr[center] - 
        //prefac * atom->q[center]);
        for (j = 0; j < dEr_indexes[bond][0][0]; j++) {
          if (dEr_indexes[bond][j + 1][1] == (tagint)-1) continue;
          der_atom = dEr_indexes[bond][j + 1][0];
          //Position of der_atom is put in unwrap
          //Virial calculation still not correctly implemented!!
          //domain->unmap(atom->x[der_atom], atom->image[der_atom], unwrap);

          //Subtraction is because deltaf term is dU/dr and force is -dU/dr
          atom->f[der_atom][0] -= dEr_vals[bond][j][0] * alpha_tot_pot;
          atom->f[der_atom][1] -= dEr_vals[bond][j][1] * alpha_tot_pot;
          atom->f[der_atom][2] -= dEr_vals[bond][j][2] * alpha_tot_pot;

          if (bondflag && (j == atom1_pos || j == atom2_pos)) {
            kb = k_bond[atom1_t][atom2_t][center_t];
            //Force contribution coming from bond stretching is reversed
            //if atom2 is considered
            if (j == atom2_pos) kb *= -1.0;
            kb_tot_pot = kb * (2.0 * (force->pair->eatomcoul[center] +
              force->kspace->eatom[center]) / atom->q[center]);
            atom->f[der_atom][0] -= bondv[0] * kb_tot_pot * bondvinv;
            atom->f[der_atom][1] -= bondv[1] * kb_tot_pot * bondvinv;
            atom->f[der_atom][2] -= bondv[2] * kb_tot_pot * bondvinv;
          }

          //Virial calculation still not correctly implemented!!
          /*if (evflag) {
            freal[0] = dEr_vals[bond][j][0] * alpha_real_space_pot;
            freal[1] = dEr_vals[bond][j][1] * alpha_real_space_pot;
            freal[2] = dEr_vals[bond][j][2] * alpha_real_space_pot;

            v[0] = freal[0] * unwrap[0];
            v[1] = freal[1] * unwrap[1];
            v[2] = freal[2] * unwrap[2];
            v[3] = freal[0] * unwrap[1];
            v[4] = freal[0] * unwrap[2];
            v[5] = freal[1] * unwrap[2];
            v_tally(der_atom, v);
          }*/
        }
        #ifdef __INTEL_MKL__
        mkl_domatadd('r', 'n', 'n', kcount, 6, alpha_tot_pot, &appo2[bond][0][0],
          6, 1.0, &appo3[0][0], 6, &appo3[0][0], 6);
        #else
        for (k = 0; k < kcount; k++) {
          appo3[k][0] += alpha_tot_pot * appo2[bond][k][0];
          appo3[k][1] += alpha_tot_pot * appo2[bond][k][1];
          appo3[k][2] += alpha_tot_pot * appo2[bond][k][2];
          appo3[k][3] += alpha_tot_pot * appo2[bond][k][3];
          appo3[k][4] += alpha_tot_pot * appo2[bond][k][4];
          appo3[k][5] += alpha_tot_pot * appo2[bond][k][5];
        }
        #endif
      }
    }

    for (k = 0; k < kcount; k++) MPI_Allreduce(MPI_IN_PLACE, appo3[k], 6,
      MPI_DOUBLE, MPI_SUM, world);

    for (i = 0; i < atom->nlocal; i++) {
      global_center = atom->tag[i] - 1;
      gen_q = qgen[types[global_center]];
      premul[0] = atom->x[i][0] * unitk[0];
      premul[1] = atom->x[i][1] * unitk[1];
      premul[2] = atom->x[i][2] * unitk[2];
      MathExtra::zero3(partial);
      for (k = 0; k < kcount; k++) {
        arg = kxvecs[k] * premul[0] + kyvecs[k] * premul[1] + kzvecs[k] *
        premul[2];
        #ifdef __INTEL_MKL__
        vdSinCos(1, &arg, &Im_xi, &Re_xi);
        #else
        Re_xi = cos(arg);
        Im_xi = sin(arg);
        #endif
        partial[0] += appo3[k][0] * Re_xi + appo3[k][3] * Im_xi;
        partial[1] += appo3[k][1] * Re_xi + appo3[k][4] * Im_xi;
        partial[2] += appo3[k][2] * Re_xi + appo3[k][5] * Im_xi;
      }
      atom->f[i][0] += gen_q * partial[0];
      atom->f[i][1] += gen_q * partial[1];
      atom->f[i][2] += gen_q * partial[2];
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixFRespEwald::memory_usage()
{
  int bond;
  double bytes = 0.0;
  bytes += atom->natoms * sizeof(int); //types
  bytes += nmolecules * (average_mol_size + 1) * sizeof(bigint); //mol_map
  bytes += 2 * natypes * sizeof(double); //q0 and qgen
  bytes += 1 * atom->nmax * sizeof(double); //deltaq
  if (angleflag) bytes += natypes * natypes * natypes * natypes *
    sizeof(double); //k_angle
  if (dihedralflag) bytes += natypes * natypes * natypes * natypes * natypes *
    sizeof(double); //k_dihedral
  if (improperflag) bytes += natypes * natypes * natypes * natypes * natypes *
    sizeof(double); //k_improper
  if (Efieldflag || bondflag) {
    bytes += kcount * 6 * sizeof(double); //appo3
    bytes += nbond_old * 6 * kcount * sizeof(double); //appo2
    //dEr_vals, dEr_indexes and distances
    for (bond = 0; bond < nbond_old; bond ++)
      bytes += dEr_indexes[bond][0][0] * (2 * (sizeof(tagint) + sizeof(bigint)
      + 3 * sizeof(double))) + 3 * sizeof(bigint);
    #ifdef __INTEL_MKL__
    //arrays used for BLAS functions in reciprocal space part of q_update_Efield
    bytes += 10 * kcount * sizeof(double);
    #endif
    //k_Efield
    if (Efieldflag) bytes += natypes * natypes * natypes * sizeof(double);
    if (bondflag) {
      bytes += nbond_old * sizeof(double); //db_vals
      bytes += natypes * natypes * natypes * sizeof(double); //k_bond
    }
  }
  return bytes;
}

/* ---------------------------------------------------------------------- */

inline double FixFRespEwald::Efield_damping(int dampflag, double r, 
  double cutoff1, double cutoff2)
{
  if (dampflag == 0) return MathSpecial::fm_exp(-0.5 * (r - cutoff2) *
    (r - cutoff2) / (cutoff1 * cutoff1));
  else if (dampflag == 1) {
    if (r > cutoff1)
      return MathSpecial::powint(sin(0.5 *
      MathConst::MY_PI * (r - cutoff1) / (cutoff2 - cutoff1)), 2);
    else return 0.0;
  }
  return 0.0;
}

