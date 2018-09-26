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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fenv.h> //Floating point exceptions
#include <mkl.h> //Not mandatory
#include "fix_fresp.h"
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
#define TWO_OVER_SQPI 1.128379167

/* ---------------------------------------------------------------------- */

FixFResp::FixFResp(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), list(NULL), kxvecs(NULL), kyvecs(NULL), kzvecs(NULL), pe(NULL),
  ug(NULL), eg(NULL), vg(NULL), ek(NULL), sfacrl_qgen(NULL), sfacim_qgen(NULL), sfacrl_all_qgen(NULL), 
  sfacim_all_qgen(NULL), cs(NULL), sn(NULL), cs_qgen(NULL), sn_qgen(NULL)
{
  int i, j, k, iarg;
  bigint *tmp;

  virial_flag = 1;

  tmp = NULL;
  gewaldflag = kewaldflag = 1;
  dampflag = -1;
  printEfieldflag = 0;
  nmolecules = 0;
  average_mol_size = cutoff1 = cutoff2 = 0.0;

  if (!force->newton_bond)
    error->all(FLERR,"Fix fresp can be used only with newton_bond on (for the moment)");
  if (strcmp(atom->atom_style, "full") != 0)
    error->all(FLERR,"Fix fresp can be used only with full atom_style");
  //if (strcmp(force->kspace_style, "ewald") != 0)
  //  error->all(FLERR,"Fix fresp can be used only with ewald kspace_style (for the moment)");

  //nevery = force->inumeric(FLERR,arg[3]); nevery != 1 not yet implemented.
  nevery = 1;
  cutoff3 = force->numeric(FLERR,arg[4]);

  //give the maximum dimension of data communicated per atom
  comm_forward = 1;
  comm_reverse = 3;

  // create arrays for storing FRESP coefficients
  memory->create(types, (int)atom->natoms, "fresp:types");
  for (i = 0; i < atom->nlocal; i++) if (atom->molecule[i] > nmolecules) nmolecules = atom->molecule[i];

  //nmolecules is the number of molecules in the simulation
  MPI_Allreduce(MPI_IN_PLACE, &nmolecules, 1, MPI_LMP_BIGINT, MPI_MAX, world);

  mol_map = (bigint**) calloc(nmolecules, sizeof(bigint*));
  int *counter = (int*) calloc(nmolecules, sizeof(int));

  //Each process counts how many atoms for a given molecule it holds
  for (i = 0; i < atom->nlocal; i++) counter[atom->molecule[i] - 1]++;
  MPI_Barrier(world);

  for (i = 0; i < nmolecules; i++) {
    //After the reduction, each process know the number of atoms contained in each molecule
    MPI_Allreduce(MPI_IN_PLACE, &counter[i], 1, MPI_INT, MPI_SUM, world);
    mol_map[i] = (bigint*)calloc(counter[i] + 1, sizeof(bigint));

    //mol_map[i][0] is equal to the number of atoms contained in molecule i
    mol_map[i][0] = counter[i];
    average_mol_size += counter[i];
    counter[i] = 0;
  }
  MPI_Barrier(world);
  average_mol_size /= (double) nmolecules;

  //Each row of mol_map is filled with global indexes of atoms holded by the process starting by position 1
  for (i = 0; i < atom->nlocal; i++) mol_map[atom->molecule[i] - 1][counter[atom->molecule[i] - 1]++ + 1] = atom->tag[i];
  MPI_Barrier(world);

  int *c_arr, *s_arr;
  for (i = 0; i < nmolecules; i++) {
    memory->create(c_arr, comm->nprocs, "fresp:c_arr");
    memory->create(s_arr, comm->nprocs, "fresp:s_arr");
    for (j = 0; j < comm->nprocs; j++) s_arr[j] = 0; //Without this, it seems not to work

    //After gathering, each element of c_arr is the number of atoms from each molecule holded by each process
    MPI_Allgather(&counter[i], 1, MPI_INT, c_arr, 1, MPI_INT, world);

    //After the cycle, each element of s_arr is the sum of preceding elements in c_arr
    for (j = 1; j < comm->nprocs; j++) {
      for (k = j; k < comm->nprocs; k++) {
        s_arr[k] += c_arr[j - 1];
      }
    }
    MPI_Barrier(world);

    memory->create(tmp, (int)mol_map[i][0], "fresp:tmp");
    /*Without passing through the temporary array tmp, vectors superpose in destination when molecule is shared between boxes
    
    With Allgather, the partial arrays contained in mol_map[i] are joined in order that each process know which
    atoms are contained in each molecule*/
   
    //MPI_Allgatherv(MPI_IN_PLACE, counter[i], MPI_LMP_BIGINT, mol_map[i] + 1, c_arr, s_arr, MPI_LMP_BIGINT, world);
    MPI_Allgatherv(mol_map[i] + 1, counter[i], MPI_LMP_BIGINT, tmp, c_arr, s_arr, MPI_LMP_BIGINT, world);
    memcpy(&mol_map[i][1], tmp, mol_map[i][0] * sizeof(bigint));

    memory->destroy(tmp);
    memory->destroy(c_arr);
    memory->destroy(s_arr);
  }
  free(counter);

  q0 = qgen = NULL;
  k_bond = k_Efield = NULL;
  k_angle = NULL;
  k_dihedral = k_improper = NULL;
 
  bondflag = angleflag = dihedralflag = improperflag = Efieldflag = 0;

  pack_flag = 0;

  //create deltaq and deltaf arrays with atom->natoms length
  memory->create(deltaq, atom->nmax, "fresp:deltaq");
  memory->create(erfc_erf_arr, atom->nmax, "fresp:erfc_erf_arr");
  memory->create(already_cycled, atom->nmax, "fresp:already_cycled");

  warn_nonneutral = warn_nocharge = 1;
  kmax_created = kmax = kcount = nmax = 0;
  kxvecs = kyvecs = kzvecs = NULL;
  eg = vg = kvecs = NULL;
  ek = NULL;
  cs = sn = NULL;

  bondvskprod_vec = xmkprod_vec = Im_xm_vec = Re_xm_vec = tmp1 = tmp2 = NULL;
  appo2Re_pref_vec = appo2Im_pref_vec = Im_prod_vec = Re_prod_vec = NULL;

  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW); //Floating point exceptions

  //Adding this compute here, it should be not necessary to add it in input file.
  id_pe = "fresp_eatom";
  char str1[] = "all";
  char str2[] = "pe/atom";
  char str3[] = "kspace";
  char **str = (char**) calloc(4, sizeof(char*));
  str[0] = const_cast<char *>(id_pe);
  str[1] = str1;
  str[2] = str2;
  str[3] = str3;
  modify->add_compute(4, str);
  free(str);
  //atom->add_callback(0);
  
  if (force->pair->ncoultablebits > 0) {
    force->pair->ncoultablebits = 0;
    printf("In order to correctly use fix fresp, ncoultablebits is set to 0.\n");
  }  
}

/* --------------------------------------------------------------------- 
  Destructor declaration is needed also if it is pure virtual
   --------------------------------------------------------------------- */

FixFResp::~FixFResp() {
  int i, j, end;
  // unregister callbacks to this fix from Atom class
  //atom->delete_callback(id,0); //???

  memory->destroy(q0);
  memory->destroy(qgen);
  memory->destroy(types);
  if (k_bond) memory->destroy(k_bond);
  if (k_angle) memory->destroy(k_angle);
  if (k_dihedral) memory->destroy(k_dihedral);
  if (k_improper) memory->destroy(k_improper);
  if (k_Efield) memory->destroy(k_Efield);
  for (i = 0; i < nmolecules; i++) free(mol_map[i]);
  free(mol_map);
  memory->destroy(deltaq);
  for (i = 0; i < nbond_old; i++) {
    memory->destroy(dEr_vals[i]);
    memory->destroy(distances[i]);
    end = dEr_indexes[i][0][0];
    for (j = 0; j <= end; j++) free(dEr_indexes[i][j]);
    memory->sfree(dEr_indexes[i]);
  }
  free(dEr_vals);
  free(distances);
  free(dEr_indexes);
  memory->destroy(bond_extremes_pos);
  memory->destroy(erfc_erf_arr);
  modify->delete_compute(id_pe);
  memory->destroy(already_cycled);

}

/* ---------------------------------------------------------------------- */

void FixFResp::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* --------------------------------------------------------------------- */

void FixFResp::init()
{
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->newton = 2;
  neighbor->requests[irequest]->ghost = 1;

  // set pe ptr
  int icompute = modify->find_compute(id_pe);
  if (icompute < 0)
    error->all(FLERR,"Potential energy ID for fix fresp does not exist");
  pe = modify->compute[icompute];

}

/* --------------------------------------------------------------------- */

int FixFResp::setmask()
{
  int mask = 0;
  mask |= POST_NEIGHBOR;
  mask |= PRE_FORCE;
  mask |= PRE_REVERSE;
  return mask;
}

/* ---------------------------------------------------------------------- */

int FixFResp::pack_forward_comm(int n, int *list, double *buf,
                          int pbc_flag, int *pbc)
{
  int m;

  if (pack_flag == 1) for(m = 0; m < n; m++) buf[m] = force->kspace->eatom[list[m]];
  else if (pack_flag == 2) for(m = 0; m < n; m++) buf[m] = atom->q[list[m]];
  else if (pack_flag == 3) for(m = 0; m < n; m++) buf[m] = erfc_erf_arr[list[m]];

  return m;
}

/* ---------------------------------------------------------------------- */

void FixFResp::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m;

  if (pack_flag == 1) for(m = 0, i = first; m < n; m++, i++) force->kspace->eatom[i] = buf[m];
  else if (pack_flag == 2) for(m = 0, i = first; m < n; m++, i++) atom->q[i] = buf[m];
  else if (pack_flag == 3) for(m = 0, i = first; m < n; m++, i++) erfc_erf_arr[i] = buf[m];
}

/* ---------------------------------------------------------------------
   charges fluctuation due to angle variation
------------------------------------------------------------------------ */

void FixFResp::q_update_angle()
{
  bigint atom1, atom2, atom3, global_atom1, global_atom2, global_atom3, global_center, molecule;
  double delx1, dely1, delz1, delx2, dely2, delz2, r1, r2, a, a0, da, k, c;
  double **x = atom->x;
  int atype, atom1_t, atom2_t, atom3_t, center_t, m, n;

  for (n = 0; n < neighbor->nanglelist; n++) {
    atom1 = neighbor->anglelist[n][0];
    atom2 = neighbor->anglelist[n][1];
    atom3 = neighbor->anglelist[n][2];
    atype = neighbor->anglelist[n][3];
    global_atom1 = atom->tag[atom1] - 1;
    global_atom2 = atom->tag[atom2] - 1;
    global_atom3 = atom->tag[atom3] - 1;
    molecule = atom->molecule[atom1];
    
    delx1 = x[atom1][0] - x[atom2][0];
    dely1 = x[atom1][1] - x[atom2][1];
    delz1 = x[atom1][2] - x[atom2][2];
    domain->minimum_image(delx1,dely1,delz1);
    r1 = sqrt(delx1*delx1 + dely1*dely1 + delz1*delz1);

    delx2 = x[atom3][0] - x[atom2][0];
    dely2 = x[atom3][1] - x[atom2][1];
    delz2 = x[atom3][2] - x[atom2][2];
    domain->minimum_image(delx2,dely2,delz2);
    r2 = sqrt(delx2*delx2 + dely2*dely2 + delz2*delz2);

    // c = cosine of angle
    c = delx1*delx2 + dely1*dely2 + delz1*delz2;
    c /= r1*r2;
    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;
    a = 180.0*acos(c) / MathConst::MY_PI;
    a0 = force->angle->equilibrium_angle(atype);
    da = a - a0;

    //A cycle over all the atoms in the same molecule of the angle is done in order
    //to correct their charges according to k_angle coefficient
    for (m = 1; m <= mol_map[molecule - 1][0]; m++) { 
      global_center = mol_map[molecule - 1][m] - 1;
      atom1_t = types[global_atom1];
      atom2_t = types[global_atom2];
      atom3_t = types[global_atom3];
      center_t = types[global_center];
      k = k_angle[atom1_t][atom2_t][atom3_t][center_t];
      
      //Charge variation are put in deltaq instead of atom->q in order to permit their communication to other processes
      deltaq[atom->map((int)global_center + 1)] += k * da;
    }
  }
}

/* ---------------------------------------------------------------------
   charges fluctuation due to dihedral variation
------------------------------------------------------------------------ */

void FixFResp::q_update_dihedral()
{
/*  bigint center, atom1, atom2, atom3, atom4, global_atom1, global_atom2, global_atom3, global_atom4, global_center, molecule;
  double vb1x, vb1y, vb1z, vb2x, vb2y, vb2z, vb3x, vb3y, vb3z, vb2xm, vb2ym, vb2zm;
  double ax, ay, az, bx, by, bz, rasq, rbsq, rgsq, rg, ra2inv, rb2inv, rabinv;
  double s, c;
  double d, dd;
  double **x = atom->x;
  double k; //k coeff for dihedral variation
  bigint nlocal = atom->nlocal, nghost = atom->nghost;
  int dtype; //dihedral type

  for (int n = 0; n < neighbor->ndihedrallist; n++) {
    atom1 = neighbor->dihedrallist[n][0];
    atom2 = neighbor->dihedrallist[n][1];
    atom3 = neighbor->dihedrallist[n][2];
    atom4 = neighbor->dihedrallist[n][3];
    dtype = neighbor->dihedrallist[n][4];
    global_atom1 = atom->tag[atom1];
    global_atom2 = atom->tag[atom2];
    global_atom3 = atom->tag[atom3];
    global_atom4 = atom->tag[atom4];
    molecule = atom->molecule[atom1];

    vb1x = x[atom1][0] - x[atom2][0];
    vb1y = x[atom1][1] - x[atom2][1];
    vb1z = x[atom1][2] - x[atom2][2];
    domain->minimum_image(vb1x,vb1y,vb1z); //???

    vb2x = x[atom3][0] - x[atom2][0];
    vb2y = x[atom3][1] - x[atom2][1];
    vb2z = x[atom3][2] - x[atom2][2];
    domain->minimum_image(vb2x,vb2y,vb2z); //???

    vb2xm = -vb2x;
    vb2ym = -vb2y;
    vb2zm = -vb2z;
    domain->minimum_image(vb2xm,vb2ym,vb2zm); //???

    vb3x = x[atom4][0] - x[atom3][0];
    vb3y = x[atom4][1] - x[atom3][1];
    vb3z = x[atom4][2] - x[atom3][2];
    domain->minimum_image(vb3x,vb3y,vb3z); //???

    ax = vb1y*vb2zm - vb1z*vb2ym;
    ay = vb1z*vb2xm - vb1x*vb2zm;
    az = vb1x*vb2ym - vb1y*vb2xm;
    bx = vb3y*vb2zm - vb3z*vb2ym;
    by = vb3z*vb2xm - vb3x*vb2zm;
    bz = vb3x*vb2ym - vb3y*vb2xm;

    rasq = ax*ax + ay*ay + az*az;
    rbsq = bx*bx + by*by + bz*bz;
    rgsq = vb2xm*vb2xm + vb2ym*vb2ym + vb2zm*vb2zm;
    rg = sqrt(rgsq);

    ra2inv = rb2inv = 0.0;
    if (rasq > 0) ra2inv = 1.0/rasq;
    if (rbsq > 0) rb2inv = 1.0/rbsq;
    rabinv = sqrt(ra2inv*rb2inv);

    c = (ax*bx + ay*by + az*bz)*rabinv;
    s = rg*rabinv*(ax*vb3x + ay*vb3y + az*vb3z);

    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;
    d = 180.0*atan2(s,c) / MathConst::MY_PI;
    d0 = force->dihedral->equilibrium_dihedral(dtype); //???
    dd = d - d0;

    for (int m = 1; m < mol_map[molecule - 1][0]; m++) { 
      global_center = mol_map[molecule - 1][m];
      center = atom->map(global_center);
      atom1_t = types[global_atom1] - 1;
      atom2_t = types[global_atom2] - 1;
      atom3_t = types[global_atom3] - 1;
      atom4_t = types[global_atom4] - 1;
      center_t = types[global_center] - 1;
      k = k_dihedral[atom1_t][atom2_t][atom3_t][atom4_t][center_t];

      //Charge variation are put in deltaq instead of atom->q in order to permit their communication to other processes
      deltaq[global_center - 1] += k * dd;
    }
  }
*/}

/* ---------------------------------------------------------------------
   charges fluctuation due to improper variation
------------------------------------------------------------------------ */

void FixFResp::q_update_improper()
{
/*  bigint center, atom1, atom2, atom3, atom4, global_atom1, global_atom2, global_atom3, global_atom4, global_center, molecule;
  double vb1x, vb1y, vb1z, vb2x, vb2y, vb2z, vb3x, vb3y, vb3z;
  double ss1, ss2, ss3, r1, r2, r3, c0, c1, c2, s1, s2;
  double s12, c;
  double im, dim, im0;
  double **x = atom->x;
  double k; //k coeff for improper variation
  int nlocal = atom->nlocal, nghost = atom->nghost;
  int itype; //improper type

  for (int n = 0; n < neighbor->nimproperlist; n++) {
    atom1 = neighbor->improperlist[n][0];
    atom2 = neighbor->improperlist[n][1];
    atom3 = neighbor->improperlist[n][2];
    atom4 = neighbor->improperlist[n][3];
    itype = neighbor->improperlist[n][4];
    global_atom1 = atom->tag[atom1];
    global_atom2 = atom->tag[atom2];
    global_atom3 = atom->tag[atom3];
    global_atom4 = atom->tag[atom4];
    molecule = atom->molecule[atom1];

    vb1x = x[atom1][0] - x[atom2][0];
    vb1y = x[atom1][1] - x[atom2][1];
    vb1z = x[atom1][2] - x[atom2][2];
    domain->minimum_image(vb1x,vb1y,vb1z); //???

    vb2x = x[atom3][0] - x[atom2][0];
    vb2y = x[atom3][1] - x[atom2][1];
    vb2z = x[atom3][2] - x[atom2][2];
    domain->minimum_image(vb2x,vb2y,vb2z); //???

    vb3x = x[atom4][0] - x[atom3][0];
    vb3y = x[atom4][1] - x[atom3][1];
    vb3z = x[atom4][2] - x[atom3][2];
    domain->minimum_image(vb3x,vb3y,vb3z); //???

    ss1 = 1.0 / (vb1x*vb1x + vb1y*vb1y + vb1z*vb1z);
    ss2 = 1.0 / (vb2x*vb2x + vb2y*vb2y + vb2z*vb2z);
    ss3 = 1.0 / (vb3x*vb3x + vb3y*vb3y + vb3z*vb3z);

    r1 = sqrt(ss1);
    r2 = sqrt(ss2);
    r3 = sqrt(ss3);

    c0 = (vb1x * vb3x + vb1y * vb3y + vb1z * vb3z) * r1 * r3;
    c1 = (vb1x * vb2x + vb1y * vb2y + vb1z * vb2z) * r1 * r2;
    c2 = -(vb3x * vb2x + vb3y * vb2y + vb3z * vb2z) * r3 * r2;

    s1 = 1.0 - c1*c1;
    if (s1 < SMALL) s1 = SMALL;
    s1 = 1.0 / s1;

    s2 = 1.0 - c2*c2;
    if (s2 < SMALL) s2 = SMALL;
    s2 = 1.0 / s2;

    s12 = sqrt(s1*s2);
    c = (c1*c2 + c0) * s12;

    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;
    im = 180.0*acos(c) / MathConst::MY_PI;
    im0 = force->improper->equilibrium_improper(itype); //???
    dim = im - im0;

    for (int m = 1; m < mol_map[molecule - 1][0]; m++) { 
      global_center = mol_map[molecule - 1][m];
      center = atom->map(global_center);
      atom1_t = types[global_atom1] - 1;
      atom2_t = types[global_atom2] - 1;
      atom3_t = types[global_atom3] - 1;
      atom4_t = types[global_atom4] - 1;
      center_t = types[global_center] - 1;
      k = k_improper[atom1_t][atom2_t][atom3_t][atom4_t][center_t];

      //Charge variation are put in deltaq instead of atom->q in order to permit their communication to other processes
      deltaq[global_center - 1] += k * dim;
    }
  }
*/}

/* ---------------------------------------------------------------------- */

void FixFResp::setup_pre_reverse(int eflag, int vflag)
{
  pre_reverse(eflag, vflag);
}

/* ---------------------------------------------------------------------- */

void FixFResp::read_file(char *file)
{
  int parseflag = -1, params_per_line = 6, atom1_t, atom2_t, atom3_t, atom4_t, center_t;
  FILE *fp;
  char **words = new char*[params_per_line+1];
  int n, nwords, eof;
  char line[MAXLINE], *ptr;

  eof = 0;
 
  if (comm->me == 0) {
    fp = force->open_potential(file);
    if (fp == NULL) {
      char str[128];
      sprintf(str,"Cannot open fix fresp parameter file %s", file);
      error->one(FLERR,str);
    }
    printf("Reading fix FRESP parameters file %s\n", file);
  }

  while (1) {
    if (comm->me == 0) {
      ptr = fgets(line, MAXLINE, fp);
      if (ptr == NULL) {
        eof = 1;
        fclose(fp);
      } else n = (int)strlen(line) + 1;
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    // strip comment, skip line if blank and update parseflag

    if ((ptr = strchr(line,'#'))) {
      if ((ptr = strstr(line, "q0"))) parseflag = 0;
      else if ((ptr = strstr(line, "qgen"))) parseflag = 1;
      else if ((ptr = strstr(line, "k_bond"))) {

        // if not already existing, create a tensor where the 1st index is atom1
        // of bond, the 2nd is atom2 and the 3rd is center whose charge is changed
        if (!k_bond) memory->create(k_bond, natypes, natypes, natypes, "fresp:k_bond");
        parseflag = 2;
        bondflag = true; 
      }
      else if ((ptr = strstr(line, "k_angle"))) {
  
        // if not already existing, create a tensor where the 1st index is atom1 of angle,
        // the 2nd is atom2, the 3rd is atom3 and the 4th is center whose charge is changed
        if (!k_angle) memory->create(k_angle, natypes, natypes, natypes, natypes, "fresp:k_angle");
        parseflag = 3;
        angleflag = true;
      }
      else if ((ptr = strstr(line, "k_dihedral"))) {

        // if not already existing, create a tensor where the 1st index is atom1 of dihedral, the 2nd
        // is atom2, the 3rd is atom3, the 4th is atom4 and the 5th is center whose charge is changed
        if (!k_dihedral) memory->create(k_dihedral, natypes, natypes, natypes, natypes, natypes, "fresp:k_dihedral");
        parseflag = 4;
        dihedralflag  = true;
      }
      else if ((ptr = strstr(line, "k_improper"))) {

        // if not already existing, create a tensor where the 1st index is atom1 of improper, the 2nd
        // is atom2, the 3rd is atom3, the 4th is atom4 and the 5th is center whose charge is changed
        if (!k_improper) memory->create(k_improper, natypes, natypes, natypes, natypes, natypes, "fresp:k_improper");
        parseflag = 5;
        improperflag = true;
      }
      else if ((ptr = strstr(line, "k_Efield"))) {

        // if not already existing, create a tensor where the 1st index is atom1
        // of bond, the 2nd is atom2 and the 3rd is center whose charge is changed
        if (!k_Efield) memory->create(k_Efield, natypes, natypes, natypes, "fresp:k_Efield");
        parseflag = 6;
        Efieldflag =  true;
        
      }
      //continue;
      *ptr = '\0'; // ??
    }
    nwords = atom->count_words(line);
    if (nwords == 0) continue;

    // words = ptrs to all words in line

    nwords = 0;
    words[nwords++] = strtok(line," \t\n\r\f");
    while ((words[nwords++] = strtok(NULL," \t\n\r\f"))) continue;

    center_t = atoi(words[0]);

    switch (parseflag) {
    
    case 0:
      q0[center_t] = atof(words[1]);
      break;
    
    case 1:
      qgen[center_t] = atof(words[1]);
      break;

    case 2:
      atom1_t = atoi(words[1]);
      atom2_t = atoi(words[2]);
      k_bond[atom1_t][atom2_t][center_t] = atof(words[3]);
      k_bond[atom2_t][atom1_t][center_t] = atof(words[3]);
      break;

    case 3:
      atom1_t = atoi(words[1]);
      atom2_t = atoi(words[2]);
      atom3_t = atoi(words[3]);
      k_angle[atom1_t][atom2_t][atom3_t][center_t] = atof(words[4]);
      k_angle[atom3_t][atom2_t][atom1_t][center_t] = atof(words[4]);
      break;
      
    case 4:
      atom1_t = atoi(words[1]);
      atom2_t = atoi(words[2]);
      atom3_t = atoi(words[3]);
      atom4_t = atoi(words[4]);
      k_dihedral[atom1_t][atom2_t][atom3_t][atom4_t][center_t] = atof(words[5]);
      k_dihedral[atom4_t][atom3_t][atom2_t][atom1_t][center_t] = atof(words[5]);
      break;
    
    case 5:
      atom1_t = atoi(words[1]);
      atom2_t = atoi(words[2]);
      atom3_t = atoi(words[3]);
      atom4_t = atoi(words[4]);
      k_improper[atom1_t][atom2_t][atom3_t][atom4_t][center_t] = atof(words[5]);
      k_improper[atom4_t][atom3_t][atom2_t][atom1_t][center_t] = atof(words[5]);
      break;
    
    case 6:
      atom1_t = atoi(words[1]);
      atom2_t = atoi(words[2]);
      k_Efield[atom1_t][atom2_t][center_t] = atof(words[3]);
      k_Efield[atom2_t][atom1_t][center_t] = atof(words[3]);
      break;
    }
  }

  delete [] words;
}

/* ---------------------------------------------------------------------- */

void FixFResp::read_file_types(char *file)
{

  int parseflag = -1, params_per_line = 6;
  FILE *fp;
  bigint center;
  char **words = new char*[params_per_line+1];
  
  if (comm->me == 0) {
    fp = force->open_potential(file);
    if (fp == NULL) {
      char str[128];
      sprintf(str,"Cannot open fix fresp types file %s",file);
      error->one(FLERR,str);
    }
  }

  int n, nwords, eof;
  char line[MAXLINE], *ptr;

  eof = 0;

  if (comm->me == 0) printf("Reading fix FRESP types file %s\n", file);
    
  while (1) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fp);
      if (ptr == NULL) {
        eof = 1;
        fclose(fp);
      } else n = (int)strlen(line) + 1;
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    // strip comment, skip line if blank and update parseflag

    if ((ptr = strchr(line, '#'))) {
      if ((ptr = strstr(line, "natypes"))) parseflag = 0;
      else if ((ptr = strstr(line, "atypes"))) parseflag = 1;
      //*ptr = '\0';
    }
    nwords = atom->count_words(line);
    if (nwords == 0) continue;

    // words = ptrs to all words in line

    nwords = 0;
    words[nwords++] = strtok(line," \t\n\r\f");
    while ((words[nwords++] = strtok(NULL," \t\n\r\f"))) continue;

    switch (parseflag) {

      case 0:
        natypes = atoi(words[0]);
        break;
 
      case 1:      
        center = atoi(words[0]) - 1;
        types[center] = atoi(words[1]);
        break;
    }

  }
  delete [] words;

}

/* ---------------------------------------------------------------------- */

void FixFResp::build_bond_Verlet_list(int bond, tagint atom1, tagint atom2)
{
  char bflag;
  int i, n, atom1_counter, counter;
  tagint j;
  int total_size = list->numneigh[atom1] + list->numneigh[atom2];
  tagint verlet_list_union[total_size];

  atom1_counter = counter = list->numneigh[atom1];
  for (i = 0; i < atom1_counter; i++) {
    verlet_list_union[i] = list->firstneigh[atom1][i] & NEIGHMASK;
    if (atom->tag[verlet_list_union[i]] == atom->tag[atom2]) bond_extremes_pos[bond][1] = i;
  }

  for (i = 0; i < list->numneigh[atom2]; i++) {
    bflag = 0;
    j = list->firstneigh[atom2][i] & NEIGHMASK;
    for (n = 0; n < atom1_counter; n++) {
      if (atom->tag[j] == atom->tag[verlet_list_union[n]]) {
        bflag = 1;
        break;
      }
    }
    if (!bflag) {
      if (atom->tag[j] == atom->tag[atom1]) bond_extremes_pos[bond][0] = counter;
      verlet_list_union[counter++] = j;
    }
  }

  //Verlet list contains bonded atoms too
  //array of derivatives for direct Efield * bond unit vector is allocated with first dimension that is
  //number of atoms in Verlet list of bond
  memory->create(dEr_vals[bond], counter, 3, "fresp:dEr_vals comp");
  memory->create(distances[bond], counter, 2, "fresp:distances comp");
  dEr_indexes[bond] = (tagint**) memory->smalloc((counter + 1) * sizeof(tagint*), "fresp:dEr_indexes comp");
  dEr_indexes[bond][0] = (tagint*) calloc(3, sizeof(tagint));
  for (i = 1; i <= counter; i++) dEr_indexes[bond][i] = (tagint*) calloc(2, sizeof(tagint));

  //dEr_indexes[bond][0][0] counts the number of atoms in bond Verlet list
  dEr_indexes[bond][0][0] = (tagint)counter;
  //dEr_indexes[bond][0][1] is atom1 (as neighbor->bondlist[bond][0])
  dEr_indexes[bond][0][1] = (tagint)atom1;
  //dEr_indexes[bond][0][2] is atom2 (as neighbor->bondlist[bond][2])
  dEr_indexes[bond][0][2] = (tagint)atom2;

  //verlet_list_union elements are copied into dEr_indexes in order to use the latter in the cycle of q_update_Efield
  for (i = 1; i <= counter; i++) dEr_indexes[bond][i][0] = verlet_list_union[i - 1];
}

/* ---------------------------------------------------------------------- 
   Counts the total number of bonds (so works with SHAKE too) in the process
   ---------------------------------------------------------------------- */
 
int FixFResp::count_total_bonds() {
  int i, j, n = 0;
  bigint atom1;
  for (i = 0; i < atom->nlocal; i++) {
    for (j = 0; j < atom->num_bond[i]; j++) {
      atom1 = atom->map(atom->bond_atom[i][j]);
      atom1 = domain->closest_image(i, (int)atom1);
      if (force->newton_bond || i < atom1) n++;
    }
  }
  return n;
}

/* ----------------------------------------------------------------------
   erfc_erf_arr is built cycling over ends of local bonds
   ---------------------------------------------------------------------- */

void FixFResp::build_erfc_erf_arr()
{
  int bond, i, atom1_type, atom2_type, center_type;
  int *type = atom->type;
  char molflag;
  tagint atom1, atom2, global_atom1, global_atom2, center;
  double ra1l, ra2l, ra1lsq, ra2lsq, grij, erfc;
  const double main_gewald = force->kspace->g_ewald, qscale = force->qqrd2e * 1.0; //1.0 is scale

  for (bond = 0; bond < nbond_old; bond++) {
    atom1 = dEr_indexes[bond][0][1];
    atom2 = dEr_indexes[bond][0][2];
    if (already_cycled[atom1] && already_cycled[atom2]) continue;
    global_atom1 = atom->tag[atom1];
    global_atom2 = atom->tag[atom2];

    //Needed because, in some cases, atom? is very big and num_bond[atom?] returns strange results
    //Maybe domain->closest_image() can be useful.
    if (atom1 > atom->natoms) atom1 = atom->map(global_atom1);
    if (atom2 > atom->natoms) atom2 = atom->map(global_atom2);
    atom1_type = type[atom1];
    atom2_type = type[atom2];

    for (i = 1; i <= dEr_indexes[bond][0][0]; i++) {
      center = dEr_indexes[bond][i][0];
      center_type = type[center];
      molflag = atom->molecule[atom1] == atom->molecule[center];
      ra1lsq = distances[bond][i - 1][0];
      ra2lsq = distances[bond][i - 1][1];
      //Check if ra1lsq > 0.0 is needed because atom1 itself can be contained in dEr_indexes[bond]
      if (!already_cycled[atom1] && ra1lsq < force->pair->cutsq[atom1_type][center_type] && ra1lsq > 0.0) {
        ra1l = sqrt(ra1lsq);
        grij = main_gewald * ra1l;
        #ifdef __INTEL_MKL__
        vdErfc(1, &grij, &erfc);
        #else
        erfc = MathSpecial::my_erfcx(grij) * MathSpecial::expmsq(grij);
        #endif
        if (!molflag) erfc_erf_arr[atom1] += atom->q[center] * erfc / ra1l;
        else erfc_erf_arr[atom1] -= atom->q[center] * (1.0 - erfc) / ra1l;
      }
      //Check if ra2lsq > 0.0 is needed because atom2 itself can be contained in dEr_indexes[bond]
      if (!already_cycled[atom2] && ra2lsq < force->pair->cutsq[atom2_type][center_type] && ra2lsq > 0.0) {
        ra2l = sqrt(ra2lsq);
        grij = main_gewald * ra2l;
        #ifdef __INTEL_MKL__
        vdErfc(1, &grij, &erfc);
        #else
        erfc = MathSpecial::my_erfcx(grij) * MathSpecial::expmsq(grij);
        #endif
        if (!molflag) erfc_erf_arr[atom2] += atom->q[center] * erfc / ra2l;
        else erfc_erf_arr[atom2] -= atom->q[center] * (1.0 - erfc) / ra2l;
      }
    }
    if (!already_cycled[atom1]) {
      erfc_erf_arr[atom1] *= qscale;
      already_cycled[atom1] = (short)1;
    }
    if (!already_cycled[atom2]) {
      erfc_erf_arr[atom2] *= qscale;
      already_cycled[atom2] = (short)1;
    }
  }
}