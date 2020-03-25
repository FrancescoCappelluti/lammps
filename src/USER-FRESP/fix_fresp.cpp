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
  Fix(lmp, narg, arg), list(NULL), kxvecs(NULL), kyvecs(NULL), kzvecs(NULL),
  pe(NULL), ug(NULL), eg(NULL), vg(NULL), ek(NULL), sfacrl_qgen(NULL),
  sfacim_qgen(NULL), sfacrl_all_qgen(NULL), sfacim_all_qgen(NULL), cs(NULL),
  sn(NULL), cs_qgen(NULL), sn_qgen(NULL)
{
  int i, j, k;
  bigint *tmp;

  virial_flag = 1;

  tmp = NULL;
  gewaldflag = kewaldflag = 1;
  dampflag = -1;
  printEfieldflag = 0;
  nmolecules = 0;
  average_mol_size = cutoff1 = cutoff2 = 0.0;

  if (!force->newton_bond)
    error->all(FLERR,"Fix fresp can be used only with newton_bond on \
      (for the moment)");
  if (strcmp(atom->atom_style, "full") != 0)
    error->all(FLERR,"Fix fresp can be used only with full atom_style");
  //if (strcmp(force->kspace_style, "ewald") != 0)
  //  error->all(FLERR,"Fix fresp can be used only with ewald kspace_style
  //    (for the moment)");

  nevery = force->inumeric(FLERR,arg[3]);
  cutoff3 = force->numeric(FLERR,arg[4]);

  //give the maximum dimension of data communicated per atom
  comm_forward = 1;
  comm_reverse = 3;

  // create arrays for storing FRESP coefficients
  memory->create(types, (int)atom->natoms, "fresp:types");
  for (i = 0; i < atom->nlocal; i++) if (atom->molecule[i] > nmolecules)
    nmolecules = atom->molecule[i];

  //nmolecules is the number of molecules in the simulation
  MPI_Allreduce(MPI_IN_PLACE, &nmolecules, 1, MPI_LMP_BIGINT, MPI_MAX, world);

  mol_map = (bigint**) calloc(nmolecules, sizeof(bigint*));
  int *counter = (int*) calloc(nmolecules, sizeof(int));

  //Each process counts how many atoms for a given molecule it holds
  for (i = 0; i < atom->nlocal; i++) counter[atom->molecule[i] - 1]++;
  MPI_Barrier(world);

  for (i = 0; i < nmolecules; i++) {
    //After the reduction, each process know the number of atoms contained
    //  in each molecule
    MPI_Allreduce(MPI_IN_PLACE, &counter[i], 1, MPI_INT, MPI_SUM, world);
    mol_map[i] = (bigint*)calloc(counter[i] + 1, sizeof(bigint));

    //mol_map[i][0] is equal to the number of atoms contained in molecule i
    mol_map[i][0] = counter[i];
    average_mol_size += counter[i];
    counter[i] = 0;
  }
  MPI_Barrier(world);
  average_mol_size /= (double)nmolecules;

  //Each row of mol_map is filled with global indexes of atoms holded
  //  by the process starting by position 1
  for (i = 0; i < atom->nlocal; i++)
    mol_map[atom->molecule[i] - 1][counter[atom->molecule[i] - 1]++ + 1] =
    atom->tag[i];
  MPI_Barrier(world);

  int *c_arr, *s_arr;
  for (i = 0; i < nmolecules; i++) {
    memory->create(c_arr, comm->nprocs, "fresp:c_arr");
    memory->create(s_arr, comm->nprocs, "fresp:s_arr");
    for (j = 0; j < comm->nprocs; j++)
      s_arr[j] = 0; //Without this, it seems not to work

    //After gathering, each element of c_arr is the number of atoms
    //  from each molecule holded by each process
    MPI_Allgather(&counter[i], 1, MPI_INT, c_arr, 1, MPI_INT, world);

    //After the cycle, each element of s_arr is the sum of preceding elements
    //  in c_arr
    for (j = 1; j < comm->nprocs; j++) {
      for (k = j; k < comm->nprocs; k++) {
        s_arr[k] += c_arr[j - 1];
      }
    }
    MPI_Barrier(world);

    memory->create(tmp, (int)mol_map[i][0], "fresp:tmp");
    /*Without passing through the temporary array tmp, vectors superpose
        in destination when molecule is shared between boxes
    
    With Allgather, the partial arrays contained in mol_map[i] are joined
      in order that each process know which atoms are contained
      in each molecule*/
   
    //MPI_Allgatherv(MPI_IN_PLACE, counter[i], MPI_LMP_BIGINT, mol_map[i] + 1,
    //  c_arr, s_arr, MPI_LMP_BIGINT, world);
    MPI_Allgatherv(mol_map[i] + 1, counter[i], MPI_LMP_BIGINT, tmp, c_arr,
      s_arr, MPI_LMP_BIGINT, world);
    memcpy(&mol_map[i][1], tmp, mol_map[i][0] * sizeof(bigint));

    memory->destroy(tmp);
    memory->destroy(c_arr);
    memory->destroy(s_arr);
  }
  free(counter);

  q0 = qgen = thoascal = NULL;
  thobscal = NULL;
  k_bond = k_Efield = NULL;
  k_angle = phi0_improper = NULL;
  k_dihedral = k_improper = NULL;
 
  bondflag = angleflag = dihedralflag = improperflag = phi0improperflag=
    Efieldflag = 0;

  pack_flag = 0;

  memory->create(deltaq, atom->nmax, "fresp:deltaq");

  warn_nonneutral = warn_nocharge = 1;
  kmax_created = kmax = kcount = nmax = 0;
  kxvecs = kyvecs = kzvecs = NULL;
  eg = vg = kvecs = NULL;
  ek = NULL;
  cs = sn = NULL;

  bondvskprod_vec = xmkprod_vec = Im_xm_vec = Re_xm_vec = tmp1 = tmp2 = NULL;
  appo2Re_pref_vec = appo2Im_pref_vec = Im_prod_vec = Re_prod_vec = NULL;

  //Adding this compute here, it is not necessary to add it in input file.
  id_pe = "fresp_eatom";
  char str1[] = "all";
  char str2[] = "pe/atom";
  char str3[] = "kspace";
  char str4[] = "pair";
  char **str = (char**) calloc(5, sizeof(char*));
  str[0] = const_cast<char *>(id_pe);
  str[1] = str1;
  str[2] = str2;
  str[3] = str3;
  str[4] = str4;
  modify->add_compute(5, str);
  free(str);
  //atom->add_callback(0); ???
}

/* --------------------------------------------------------------------- 
  Destructor declaration is needed also if it is pure virtual
   --------------------------------------------------------------------- */

FixFResp::~FixFResp() {
  int i, j, end;
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
  if (Efieldflag || bondflag) {
    for (i = 0; i < nbond_old; i++) {
      memory->destroy(dEr_vals[i]);
      end = dEr_indexes[i][0][0];
      for (j = 0; j <= end; j++) free(dEr_indexes[i][j]);
      memory->sfree(dEr_indexes[i]);
    }
    memory->destroy(bond_extremes_pos);
    free(dEr_vals);
    free(dEr_indexes);
  }
  if (improperflag) memory->destroy(dimp_vals);
  if (angleflag) memory->destroy(da_vals);
  modify->delete_compute(id_pe);
  if (thoascal) memory->destroy(thoascal);
  if (thobscal) memory->destroy(thobscal);
}

/* ---------------------------------------------------------------------- */

void FixFResp::min_post_neighbor()
{
  post_neighbor();
}

/* ---------------------------------------------------------------------- */

void FixFResp::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* --------------------------------------------------------------------- */

void FixFResp::init()
{
  //In following line, 1.0 is considered to be a maximal half-bond length
  double cutghost, mycutneigh = cutoff3 + neighbor->skin + 1.0;
  if (force->pair)
    cutghost = MAX(force->pair->cutforce + neighbor->skin, comm->cutghostuser);
  else
    cutghost = comm->cutghostuser;

  if (mycutneigh > cutghost) {
    //ceil(mycutneigh) is the smallest integer equal or bigger to mycutneigh
    comm->cutghostuser = ceil(mycutneigh);
    if (comm->me == 0) {
      if (screen)
        fprintf(screen, "cutghostuser set to %lf in order to correctly\
 use fix fresp\n", ceil(mycutneigh));
      if (logfile)
        fprintf(logfile, "cutghostuser set to %lf in order to correctly\
 use fix fresp\n", ceil(mycutneigh));
    }
  }
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->newton = 2;
  neighbor->requests[irequest]->occasional = 1;
  if (cutoff3 > force->pair->cutforce) {
    neighbor->requests[irequest]->cut = 1;
    neighbor->requests[irequest]->cutoff = cutoff3 + neighbor->skin + 1.0;
  }

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
  mask |= MIN_POST_NEIGHBOR;
  mask |= PRE_FORCE;
  mask |= MIN_PRE_FORCE;
  mask |= PRE_REVERSE;
  mask |= MIN_PRE_REVERSE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixFResp::min_pre_force(int vflag)
{
  pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

int FixFResp::pack_forward_comm(int n, int *list, double *buf,
                          int pbc_flag, int *pbc)
{
  int m;

  if (pack_flag == 1) for(m = 0; m < n; m++) buf[m] =
    force->kspace->eatom[list[m]];
  else if (pack_flag == 2) for(m = 0; m < n; m++) buf[m] = atom->q[list[m]];
  else if (pack_flag == 3) for(m = 0; m < n; m++) buf[m] =
    force->pair->eatomcoul[list[m]];
  return m;
}

/* ---------------------------------------------------------------------- */

void FixFResp::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m;

  if (pack_flag == 1) for(m = 0, i = first; m < n; m++, i++)
    force->kspace->eatom[i] = buf[m];
  else if (pack_flag == 2) for(m = 0, i = first; m < n; m++, i++)
    atom->q[i] = buf[m];
  else if (pack_flag == 3) for(m = 0, i = first; m < n; m++, i++)
    force->pair->eatomcoul[i] = buf[m];
}

/* ---------------------------------------------------------------------- */

int FixFResp::pack_reverse_comm(int n, int first, double *buf)
{
  int i, m;
  if (pack_flag == 1) for (m = 0, i = first; m < n; m++, i++)
    buf[m] = deltaq[i];
  else if (pack_flag == 3) for (m = 0, i = first; m < n; m++, i++)
    buf[m] = force->pair->eatomcoul[i];
  return m;
}

/* ---------------------------------------------------------------------- */

void FixFResp::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m;

  if (pack_flag == 1) for(m = 0; m < n; m++) deltaq[list[m]] += buf[m];
  else if (pack_flag == 3) for (m = 0; m < n; m++)
    force->pair->eatomcoul[list[m]] += buf[m];
}

/* ---------------------------------------------------------------------
   charges fluctuation due to angle variation
------------------------------------------------------------------------ */

void FixFResp::q_update_angle()
{
  bigint atom1, atom2, atom3, global_atom1, global_atom2, global_atom3,
    molecule;
  double r1inv, r2inv, a, a0, da, num, invden, cosa, fvpi[3], fvpk[3];
  double rij[3], rkj[3], rijuv[3], rkjuv[3], dadri[3], dadrj[3], dadrk[3];
  double vpi[3], vpk[3], pref, prefi, prefk, **x = atom->x;
  int atype, atom1_t, atom2_t, atom3_t, an;

  for (an = 0; an < neighbor->nanglelist; an++) {
    atom1 = neighbor->anglelist[an][0];
    atom2 = neighbor->anglelist[an][1];
    atom3 = neighbor->anglelist[an][2];
    atype = neighbor->anglelist[an][3];
    global_atom1 = atom->tag[atom1];
    global_atom2 = atom->tag[atom2];
    global_atom3 = atom->tag[atom3];
    molecule = atom->molecule[atom1];
    atom1_t = types[global_atom1 - 1];
    atom2_t = types[global_atom2 - 1];
    atom3_t = types[global_atom3 - 1];
    
    MathExtra::sub3(x[atom1], x[atom2], rij);
    domain->minimum_image(rij[0], rij[1], rij[2]);
    r1inv = 1. / MathExtra::len3(rij);

    MathExtra::sub3(x[atom3], x[atom2], rkj);
    domain->minimum_image(rkj[0], rkj[1], rkj[2]);
    r2inv = 1. / MathExtra::len3(rkj);

    num = MathExtra::dot3(rij, rkj);
    invden = r1inv * r2inv;
    cosa = num * invden;
    a = acos(cosa);

    a0 = force->angle->equilibrium_angle(atype);

    da = a - a0;

    //charge variation is proportional to a - a0
    deltaq_update_angle(molecule, atom1_t, atom2_t, atom3_t, da);
    
    pref = 1. / sqrt(1. - cosa * cosa);
    prefi = pref * r1inv;
    prefk = pref * r2inv;

    //Unit vectors of rij and rkj are calculated
    MathExtra::copy3(rij, rijuv);
    MathExtra::copy3(rkj, rkjuv);
    MathExtra::scale3(r1inv, rijuv);
    MathExtra::scale3(r2inv, rkjuv);

    MathExtra::copy3(rijuv, fvpi);
    MathExtra::copy3(rkjuv, fvpk);
    MathExtra::scale3(cosa, fvpi);
    MathExtra::scale3(cosa, fvpk);

    MathExtra::sub3(fvpi, rkjuv, vpi);
    MathExtra::sub3(fvpk, rijuv, vpk);

    MathExtra::copy3(vpi, dadri);
    MathExtra::copy3(vpk, dadrk);
    MathExtra::scale3(prefi, dadri);
    MathExtra::scale3(prefk, dadrk);
    
    MathExtra::copy3(dadri, da_vals[an][0]);
    MathExtra::copy3(dadrk, da_vals[an][2]);

    MathExtra::negate3(dadri);
    MathExtra::sub3(dadri, dadrk, dadrj);

    MathExtra::copy3(dadrj, da_vals[an][1]);
  }
}

/* ---------------------------------------------------------------------
   charges fluctuation due to dihedral variation
------------------------------------------------------------------------ */

void FixFResp::q_update_dihedral()
{
/*  bigint center, atom1, atom2, atom3, atom4, global_atom1, global_atom2;
  bigint global_atom3, global_atom4, global_center, molecule;
  double vb1x, vb1y, vb1z, vb2x, vb2y, vb2z, vb3x, vb3y, vb3z, vb2xm, vb2ym;
  double ax, ay, az, bx, by, bz, rasq, rbsq, rgsq, rg, ra2inv, rb2inv, rabinv;
  double s, c, vb2zm;
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
    domain->minimum_image(vb1x,vb1y,vb1z);

    vb2x = x[atom3][0] - x[atom2][0];
    vb2y = x[atom3][1] - x[atom2][1];
    vb2z = x[atom3][2] - x[atom2][2];
    domain->minimum_image(vb2x,vb2y,vb2z);

    vb2xm = -vb2x;
    vb2ym = -vb2y;
    vb2zm = -vb2z;
    domain->minimum_image(vb2xm,vb2ym,vb2zm);

    vb3x = x[atom4][0] - x[atom3][0];
    vb3y = x[atom4][1] - x[atom3][1];
    vb3z = x[atom4][2] - x[atom3][2];
    domain->minimum_image(vb3x,vb3y,vb3z);

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

      //Charge variation are put in deltaq instead of atom->q in order
      //to permit their communication to other processes
      deltaq[global_center - 1] += k * dd;
    }
  }
*/}

/* ---------------------------------------------------------------------
   charges fluctuation due to improper variation
------------------------------------------------------------------------ */

void FixFResp::q_update_improper()
{
  bigint atom1, atom2, atom3, atom4, global_atom1, global_atom2;
  bigint global_atom3, global_atom4, molecule;
  int improper, atom1_t, atom2_t, atom3_t, atom4_t, itype;
  double im, absim;
  double rij[3], rkj[3], rik[3], rkl[3], rjl[3], casnafn[3], dasnafn;
  double pref, a[3], b[3], afn[3], asn[3], af[3], as[3], afd, asd; 
  double cosim, cosimas[3], cosimaf[3], dcosimdri[3], dcosimdrl[3];
  double dcosimdrj[3], dcosimdrk[3], dcosimdrj1[3], dcosimdrj2[3], ddimdri[3];
  double dcosimdrk1[3], dcosimdrk2[3], ddimdrj[3], ddimdrk[3], ddimdrl[3];
  double **x = atom->x;

  for (improper = 0; improper < nimproper_old; improper++) {
    atom1 = neighbor->improperlist[improper][0];
    atom2 = neighbor->improperlist[improper][1];
    atom3 = neighbor->improperlist[improper][2];
    atom4 = neighbor->improperlist[improper][3];
    itype = neighbor->improperlist[improper][4];
    global_atom1 = atom->tag[atom1];
    global_atom2 = atom->tag[atom2];
    global_atom3 = atom->tag[atom3];
    global_atom4 = atom->tag[atom4];
    molecule = atom->molecule[atom1];
    atom1_t = types[global_atom1 - 1];
    atom2_t = types[global_atom2 - 1];
    atom3_t = types[global_atom3 - 1];
    atom4_t = types[global_atom4 - 1];

    MathExtra::sub3(x[atom1], x[atom2], rij);
    domain->minimum_image(rij[0], rij[1], rij[2]);

    MathExtra::sub3(x[atom3], x[atom2], rkj);
    domain->minimum_image(rkj[0], rkj[1], rkj[2]);

    MathExtra::sub3(x[atom3], x[atom4], rkl);
    domain->minimum_image(rkl[0], rkl[1], rkl[2]);

    MathExtra::sub3(x[atom1], x[atom3], rik);
    domain->minimum_image(rik[0], rik[1], rik[2]);

    MathExtra::sub3(x[atom2], x[atom4], rjl);
    domain->minimum_image(rjl[0], rjl[1], rjl[2]);

    MathExtra::cross3(rkj, rkl, afn);
    MathExtra::cross3(rij, rkj, asn);
    afd = 1. / MathExtra::len3(afn);
    asd = 1. / MathExtra::len3(asn);

    MathExtra::cross3(asn, afn, casnafn);
    dasnafn = MathExtra::dot3(asn, afn);
    //Being cosine symmetric with respect to y-axis, following product already
    //is the cosine of im: cos(im) = cos(+/-acos(arg)) = cos(acos(arg)) = arg.
    cosim = dasnafn * asd * afd;
    im = acos(cosim);
    if (MathExtra::dot3(rkj, casnafn) < 0.) im *= -1.;

    //pref = d(im)/d(cos(im))
    pref = -1. / sin(im);

    //abs(im) is employed for charge variation
    //pref is multiplied times sign(im)
    if (im > 0.) {
      absim = im;
    }
    else {
      absim = -im;
      pref *= -1.;
    }

    //differently from the others "deltaq_update" functions, only |im| is
    //passed as argument because im0 will be retrieved by the function
    deltaq_update_improper(molecule, atom1_t, atom2_t, atom3_t, atom4_t, absim);

    MathExtra::copy3(afn, af);
    MathExtra::copy3(asn, as);
    MathExtra::scale3(afd, af);
    MathExtra::scale3(asd, as);

    MathExtra::copy3(af, cosimaf);
    MathExtra::copy3(as, cosimas);
    MathExtra::scale3(cosim, cosimaf);
    MathExtra::scale3(cosim, cosimas);
    
    MathExtra::sub3(af, cosimas, a);
    MathExtra::sub3(as, cosimaf, b);
    MathExtra::scale3(asd, a);
    MathExtra::scale3(afd, b);

    MathExtra::cross3(rik, a, dcosimdrj1);
    MathExtra::cross3(rkl, b, dcosimdrj2);
    MathExtra::cross3(rjl, b, dcosimdrk1);
    MathExtra::cross3(rij, a, dcosimdrk2);

    MathExtra::cross3(rkj, a, dcosimdri);
    MathExtra::cross3(rkj, b, dcosimdrl);
    MathExtra::sub3(dcosimdrj1, dcosimdrj2, dcosimdrj);
    MathExtra::sub3(dcosimdrk1, dcosimdrk2, dcosimdrk);

    MathExtra::copy3(dcosimdri, ddimdri);
    MathExtra::copy3(dcosimdrj, ddimdrj);
    MathExtra::copy3(dcosimdrk, ddimdrk);
    MathExtra::copy3(dcosimdrl, ddimdrl);

    MathExtra::scale3(pref, ddimdri);
    MathExtra::scale3(pref, ddimdrj);
    MathExtra::scale3(pref, ddimdrk);
    MathExtra::scale3(pref, ddimdrl);
    
    MathExtra::copy3(ddimdri, dimp_vals[improper][0]);
    MathExtra::copy3(ddimdrj, dimp_vals[improper][1]);
    MathExtra::copy3(ddimdrk, dimp_vals[improper][2]);
    MathExtra::copy3(ddimdrl, dimp_vals[improper][3]);
  }
}

/* ---------------------------------------------------------------------- */

void FixFResp::min_pre_reverse(int eflag, int vflag)
{
  pre_reverse(eflag, vflag);
}

/* ---------------------------------------------------------------------- */

void FixFResp::setup_pre_reverse(int eflag, int vflag)
{
  pre_reverse(eflag, vflag);
}

/* ---------------------------------------------------------------------- */

void FixFResp::read_file(char *file)
{
  int parseflag = -1, params_per_line = 6, atom1_t, atom2_t, atom3_t,
    atom4_t, center_t;
  double pho, apol;
  FILE *fp;
  char **words = new char*[params_per_line+1];
  int nwords, eof, i, j, k, l, m, n;
  char line[MAXLINE], *ptr;
  static double deg2rad = MathConst::MY_PI / 180.;

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

        //if not already existing, create a tensor where the 1st index
        //is atom1 of bond, the 2nd is atom2 and the 3rd is center
        //whose charge is changed
        if (!k_bond) memory->create(k_bond, natypes, natypes, natypes,
          "fresp:k_bond");
        for (i = 0; i < natypes; i++) {
          for (j = 0; j < natypes; j++) {
            for (k = 0; k < natypes; k++) k_bond[i][j][k] = 0.0;
          }
        }
        parseflag = 2;
        bondflag = true; 
      }
      else if ((ptr = strstr(line, "k_angle"))) {
  
        //if not already existing, create a tensor where the 1st index
        //is atom1 of angle, the 2nd is atom2, the 3rd is atom3
        //and the 4th is center whose charge is changed
        if (!k_angle) memory->create(k_angle, natypes, natypes, natypes,
          natypes, "fresp:k_angle");
        for (i = 0; i < natypes; i++) {
          for (j = 0; j < natypes; j++) {
            for (k = 0; k < natypes; k++) {
              for (l = 0; l < natypes; l++) k_angle[i][j][k][l] = 0.0;
            }
          }
        }
        parseflag = 3;
        angleflag = true;
      }
      else if ((ptr = strstr(line, "k_dihedral"))) {

        //if not already existing, create a tensor where the 1st index
        //is atom1 of dihedral, the 2nd is atom2, the 3rd is atom3,
        //the 4th is atom4 and the 5th is center whose charge is changed
        if (!k_dihedral) memory->create(k_dihedral, natypes, natypes,
          natypes, natypes, natypes, "fresp:k_dihedral");
        for (i = 0; i < natypes; i++) {
          for (j = 0; j < natypes; j++) {
            for (k = 0; k < natypes; k++) {
              for (l = 0; l < natypes; l++) {
                for (m = 0; m < natypes; m++) k_dihedral[i][j][k][l][m] = 0.0;
              }
            }
          }
        }
        parseflag = 4;
        dihedralflag  = true;
      }
      else if ((ptr = strstr(line, "k_improper"))) {

        //if not already existing, create a tensor where the 1st index
        //is atom1 of improper, the 2nd is atom2, the 3rd is atom3,
        //the 4th is atom4 and the 5th is center whose charge is changed
        if (!k_improper) memory->create(k_improper, natypes, natypes,
          natypes, natypes, natypes, "fresp:k_improper");
        for (i = 0; i < natypes; i++) {
          for (j = 0; j < natypes; j++) {
            for (k = 0; k < natypes; k++) {
              for (l = 0; l < natypes; l++) {
                for (m = 0; m < natypes; m++) k_improper[i][j][k][l][m] = 0.0;
              }
            }
          }
        }
        parseflag = 5;
        improperflag = true;
      }
      else if ((ptr = strstr(line, "phi0_improper"))) {

        //if not already existing, create a tensor where the 1st index
        //is atom1 of improper, the 2nd is atom2, the 3rd is atom3 and
        //the 4th is atom4
        if (!phi0_improper) memory->create(phi0_improper, natypes, natypes,
          natypes, natypes, "fresp:phi0_improper");
        for (i = 0; i < natypes; i++) {
          for (j = 0; j < natypes; j++) {
            for (k = 0; k < natypes; k++) {
              for (l = 0; l < natypes; l++) phi0_improper[i][j][k][l] = 180.0;
            }
          }
        }
        parseflag = 6;
        phi0improperflag = true;
      }
      else if ((ptr = strstr(line, "k_Efield"))) {

        //if not already existing, create a tensor where the 1st index
        //is atom1 of bond, the 2nd is atom2 and the 3rd is center
        //whose charge is changed
        if (!k_Efield) memory->create(k_Efield, natypes, natypes,
          natypes, "fresp:k_Efield");
        for (i = 0; i < natypes; i++) {
          for (j = 0; j < natypes; j++) {
            for (k = 0; k < natypes; k++) k_Efield[i][j][k] = 0.0;
          }
        }
        parseflag = 7;
        Efieldflag =  true;
        
      }
      else if ((ptr = strstr(line, "atom_pol"))) {

        //Create an array where atom polarizability is associated with
	//atom global indexes
        if (!thoascal) memory->create(thoascal, natypes, "fresp:thoascal");
	parseflag = 8;
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

    center_t = atoi(words[0]) - 1;

    switch (parseflag) {
    
    case 0:
      q0[center_t] = atof(words[1]);
      break;
    
    case 1:
      qgen[center_t] = atof(words[1]);
      break;

    case 2:
      atom1_t = atoi(words[1]) - 1;
      atom2_t = atoi(words[2]) - 1;
      pho = atof(words[3]);
      k_bond[atom1_t][atom2_t][center_t] = pho;
      k_bond[atom2_t][atom1_t][center_t] = pho;
      break;

    case 3:
      atom1_t = atoi(words[1]) - 1;
      atom2_t = atoi(words[2]) - 1;
      atom3_t = atoi(words[3]) - 1;
      pho = atof(words[4]);
      k_angle[atom1_t][atom2_t][atom3_t][center_t] = pho;
      k_angle[atom3_t][atom2_t][atom1_t][center_t] = pho;
      break;
      
    case 4:
      atom1_t = atoi(words[1]) - 1;
      atom2_t = atoi(words[2]) - 1;
      atom3_t = atoi(words[3]) - 1;
      atom4_t = atoi(words[4]) - 1;
      pho = atof(words[5]);
      k_dihedral[atom1_t][atom2_t][atom3_t][atom4_t][center_t] =
        pho;
      k_dihedral[atom4_t][atom3_t][atom2_t][atom1_t][center_t] =
        pho;
      break;
    
    case 5:
      atom1_t = atoi(words[1]) - 1;
      atom2_t = atoi(words[2]) - 1;
      atom3_t = atoi(words[3]) - 1;
      atom4_t = atoi(words[4]) - 1;
      pho = atof(words[5]);
      k_improper[atom1_t][atom2_t][atom3_t][atom4_t][center_t] =
        pho;
      k_improper[atom4_t][atom3_t][atom2_t][atom1_t][center_t] =
        pho;
      break;
    
    case 6:
      atom1_t = center_t;
      atom2_t = atoi(words[1]) - 1;
      atom3_t = atoi(words[2]) - 1;
      atom4_t = atoi(words[3]) - 1;
      //phi is converted from degrees to radians
      pho = atof(words[4]) * deg2rad;
      phi0_improper[atom1_t][atom2_t][atom3_t][atom4_t] = pho;
      phi0_improper[atom4_t][atom3_t][atom2_t][atom1_t] = pho;
      break;
    
    case 7:
      atom1_t = atoi(words[1]) - 1;
      atom2_t = atoi(words[2]) - 1;
      pho = atof(words[3]);
      k_Efield[atom1_t][atom2_t][center_t] = pho;
      k_Efield[atom2_t][atom1_t][center_t] = pho;
      break;
    
    case 8:
      apol = atof(words[1]);
      //Atomic polarizabilities are stored as 2.6 / a**(1/6) because they
      //will always be used in this form in the calculation of s coefficient,
      //as stated in lammps.sandia.gov/doc/pair_thole.html
      thoascal[center_t] = 2.6 * pow(apol, -SIXTH);
      break;
    }
  }

  if (!thoascal && (dampflag == THO))
    error->all(FLERR, "Atomic polarizabilities have to be defined in FRESP \
      parameters file in order to use Thole damping");

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
        types[center] = atoi(words[1]) - 1;
        break;
    }

  }
  delete [] words;

}

/* ---------------------------------------------------------------------- */

void FixFResp::build_bond_Verlet_list(int bond, tagint atom1, tagint atom2)
{
  int i;

  //+1 is needed in order to include atom1 too, which is obviously not 
  //contained in its Verlet list
  int total_size = list->numneigh[atom1] + 1;

  //Verlet list contains bonded atoms too
  //array of derivatives for direct Efield * bond unit vector is allocated 
  //with first dimension that is number of atoms in Verlet list of bond
  memory->create(dEr_vals[bond], total_size, 3, "fresp:dEr_vals comp");
  dEr_indexes[bond] = (tagint**) memory->smalloc((total_size + 1) * 
    sizeof(tagint*), "fresp:dEr_indexes comp");
  dEr_indexes[bond][0] = (tagint*) calloc(3, sizeof(tagint));
  for (i = 1; i <= total_size; i++) 
    dEr_indexes[bond][i] = (tagint*) calloc(2, sizeof(tagint));

  //dEr_indexes[bond][0][0] counts the number of atoms in bond Verlet list
  dEr_indexes[bond][0][0] = (tagint)total_size;
  //dEr_indexes[bond][0][1] is atom1 (as neighbor->bondlist[bond][0])
  dEr_indexes[bond][0][1] = (tagint)atom1;
  //dEr_indexes[bond][0][2] is atom2 (as neighbor->bondlist[bond][2])
  dEr_indexes[bond][0][2] = (tagint)atom2;

  //atoms from occasional Verlet list of atom1 are copied into dEr_indexes 
  //in order to use the latter in the cycle of q_update_Efield
  for (i = 1; i < total_size; i++) {
    dEr_indexes[bond][i][0] = list->firstneigh[atom1][i - 1];
    if (atom->tag[dEr_indexes[bond][i][0] & NEIGHMASK] == atom->tag[atom2])
      bond_extremes_pos[bond][1] = i - 1;
  }
  dEr_indexes[bond][total_size][0] = atom1;
  bond_extremes_pos[bond][0] = total_size - 1;
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
   deltaq array is updated considering electric field polarization
     contribution
   ---------------------------------------------------------------------- */
 
void FixFResp::deltaq_update_Efield(bigint molecule, int atom1_t, int atom2_t, 
  double Eparallel) {
  bigint i, global_center, center;
  int center_t;
  double k;
  
  //The cycle is done over all the atoms contained in the same molecule of
  //the bond
  #pragma vector
  for (i = 1; i <= mol_map[molecule - 1][0]; i++) {
    global_center = mol_map[molecule - 1][i];
    center = atom->map((int)global_center);
    center_t = types[global_center - 1];
    k = k_Efield[atom1_t][atom2_t][center_t];
    //Charge variation are put in deltaq instead of atom->q in order to
    //permit their communication to other processes
    deltaq[center] += k * Eparallel;
  }
}

/* ---------------------------------------------------------------------- 
   deltaq array is updated considering bond stretching polarization
     contributions
   ---------------------------------------------------------------------- */
 
void FixFResp::deltaq_update_bond(bigint molecule, int atom1_t, int atom2_t, 
  double dr) {
  bigint i, global_center, center;
  int center_t;
  double k;
  
  //The cycle is done over all the atoms contained in the same molecule of
  //the bond
  #pragma vector
  for (i = 1; i <= mol_map[molecule - 1][0]; i++) {
    global_center = mol_map[molecule - 1][i];
    center = atom->map((int)global_center);
    center_t = types[global_center - 1];
    k = k_bond[atom1_t][atom2_t][center_t];
    //Charge variation are put in deltaq instead of atom->q in order to
    //permit their communication to other processes
    deltaq[center] += k * dr;
  }
}

/* ---------------------------------------------------------------------- 
   deltaq array is updated considering angle bending polarization
     contribution
   ---------------------------------------------------------------------- */
 
void FixFResp::deltaq_update_angle(bigint molecule, int atom1_t, int atom2_t, 
  int atom3_t, double da) {
  bigint i, global_center, center;
  int center_t;
  double k;
  
  //The cycle is done over all the atoms contained in the same molecule of
  //the angle
  #pragma vector
  for (i = 1; i <= mol_map[molecule - 1][0]; i++) {
    global_center = mol_map[molecule - 1][i];
    center = atom->map((int)global_center);
    center_t = types[global_center - 1];
    k = k_angle[atom1_t][atom2_t][atom3_t][center_t];
    //Charge variation are put in deltaq instead of atom->q in order to
    //permit their communication to other processes
    deltaq[center] += k * da;
  }
}

/* ---------------------------------------------------------------------- 
   deltaq array is updated considering improper change polarization
     contribution
   ---------------------------------------------------------------------- */
 
void FixFResp::deltaq_update_improper(bigint molecule, int atom1_t, int atom2_t, 
  int atom3_t, int atom4_t, double absim) {
  bigint i, global_center, center;
  int center_t;
  double k, im0, dim;
  
  //The cycle is done over all the atoms contained in the same molecule of
  //the improper
  #pragma vector
  for (i = 1; i <= mol_map[molecule - 1][0]; i++) {
    global_center = mol_map[molecule - 1][i];
    center = atom->map((int)global_center);
    center_t = types[global_center - 1];
    k = k_improper[atom1_t][atom2_t][atom3_t][atom4_t][center_t];
    im0 = phi0_improper[atom1_t][atom2_t][atom3_t][atom4_t];
    dim = absim - im0;
    //Charge variation are put in deltaq instead of atom->q in order to
    //permit their communication to other processes
    deltaq[center] += k * dim;
  }
}
