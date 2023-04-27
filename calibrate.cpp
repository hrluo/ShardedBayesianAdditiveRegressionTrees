//     calibrate.cpp: Implement command-line model interface to OpenBT (BART-based) computer model calibration.
//     Copyright (C) 2020 Matthew T. Pratola
//
//     This file is part of OpenBT.
//
//     OpenBT is free software: you can redistribute it and/or modify
//     it under the terms of the GNU Affero General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.
//
//     OpenBT is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//     GNU Affero General Public License for more details.
//
//     You should have received a copy of the GNU Affero General Public License
//     along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//     Author contact information
//     Matthew T. Pratola: mpratola@gmail.com


#include <chrono>
#include <iostream>
#include <string>
#include <ctime>
#include <sstream>

#include <fstream>
#include <vector>
#include <limits>
#include <algorithm>

#include "crn.h"
#include "tree.h"
#include "brt.h"
#include "brtfuns.h"
#include "dinfo.h"
#include "mbrt.h"
#include "ambrt.h"
#include "psbrt.h"
#include "tnorm.h"

using std::cout;
using std::endl;

#define MODEL_BT 1
#define MODEL_BINOMIAL 2
#define MODEL_POISSON 3
#define MODEL_BART 4
#define MODEL_HBART 5
#define MODEL_PROBIT 6
#define MODEL_MODIFIEDPROBIT 7
#define MODEL_MERCK_TRUNCATED 8
#define MODEL_CALIBRATE 9

#define MPI_TAG_CAL_ACCEPT 9001
#define MPI_TAG_CAL_REJECT 9002



void updateall(std::vector<double>& xpred, std::vector<double>& theta, std::vector<size_t>& thetaidx, size_t pthetas, size_t p, size_t nf)
{
   size_t index;
   for(size_t j=0;j<pthetas;j++) {
      for(size_t i=0;i<nf;i++)
      {
         index=i*p+thetaidx[j];
         xpred[index]=theta[j];
      }
   }
}

double updatetheta(std::vector<double>& xpred, size_t j, double thetaprime, std::vector<size_t>& thetaidx, size_t pthetas, size_t p, size_t nf)
{
   size_t index;
   double thetaold=xpred[thetaidx[j]];
   for(size_t i=0;i<nf;i++)
   {
      index=i*p+thetaidx[j];
      xpred[index]=thetaprime;
   }

   return thetaold;
}


int main(int argc, char* argv[])
{
   std::string folder("");

   if(argc>1)
   {
      std::string confopt("--conf");
      if(confopt.compare(argv[1])==0) {
#ifdef _OPENMPI
         return 101;
#else
         return 100;
#endif
      }

      //otherwise argument on the command line is path to conifg file.
      folder=std::string(argv[1]);
      folder=folder+"/";
   }


   //-----------------------------------------------------------
   //random number generation
   crn gen;
   gen.set_seed(static_cast<long long>(std::chrono::high_resolution_clock::now()
                                   .time_since_epoch()
                                   .count()));

   //--------------------------------------------------
   //process args
   std::ifstream conf(folder+"config.calibrate");

   // model type
   int modeltypec,modeltypef;
   conf >> modeltypec;
   conf >> modeltypef;

   // core filenames for xc,yc input
   std::string xcore,ycore;
   conf >> xcore;
   conf >> ycore;

   // core filenames for xf,yf input
   std::string xfcore,yfcore;
   conf >> xfcore;
   conf >> yfcore;

   // core filenames for xpred at the unknown theta
   std::string xpcore;
   conf >> xpcore;

   // sample mean (offset) of yc and yf that was removed by detrending in caller function.
   double fmeanc,fmeanf;
   conf >> fmeanc;
   conf >> fmeanf;

   //offset
   double off;
   off = fmeanf-fmeanc;//fmeanc-fmeanf;

   // number of trees for the BART model of eta(x,theta) and delta(x)
   size_t mc,mhc,mf,mhf;
   conf >> mc;
   conf >> mhc;
   conf >> mf;
   conf >> mhf;

   //number of draws to save, burn, adapt and adaptevery
   size_t nd,burn,nadapt,adaptevery;
   conf >> nd;
   conf >> burn;
   conf >> nadapt;
   conf >> adaptevery;

   //mu prior (tau, ambrt) and sigma prior (lambda,nu, psbrt) for eta(x,theta)
   double tauc;
   double overalllambdac;
   double overallnuc;
   conf >> tauc;
   conf >> overalllambdac;
   conf >> overallnuc;

   //mu prior (tau, ambrt) and sigma prior (lambda,nu, psbrt) for delta(x)
   double tauf;
   double overalllambdaf;
   double overallnuf;
   conf >> tauf;
   conf >> overalllambdaf;
   conf >> overallnuf;

   //tree prior for both eta(x,theta) and delta(x)
   //I assume same prior for both functions for now.
   double alpha;
   double mybeta;
   double alphah;
   double mybetah;
   conf >> alpha;
   conf >> mybeta;
   conf >> alphah;
   conf >> mybetah;

   //thread count
   int tc;
   conf >> tc;

   //sigma vector for eta(x,theta) and delta(x)
   std::string scorec,scoref;
   conf >> scorec;
   conf >> scoref;

   //change variable for eta(x,theta) and delta(x)
   std::string chgvcorec,chgvcoref;
   conf >> chgvcorec;
   conf >> chgvcoref;

   //control for eta(x,theta) and delta(x)
   //I assume pb is the same for both functions for now.
   double pbdc;
   double pbdch;
   double pb;
   double pbdf;
   double pbdfh;
   double stepwpert;
   double stepwperth;
   double probchv;
   double probchvh;
   size_t minnumbot;
   size_t minnumboth;
   size_t printevery;
   conf >> pb;
   conf >> pbdc;
   conf >> pbdch;
   conf >> pbdf;
   conf >> pbdfh;
   conf >> stepwpert;
   conf >> stepwperth;
   conf >> probchv;
   conf >> probchvh;
   conf >> minnumbot;
   conf >> minnumboth;
   conf >> printevery;

   //cut info for eta(x,theta) and delta(x)
   std::string xicorec,xicoref;
   conf >> xicorec;
   conf >> xicoref;

   //model name
   std::string modelname;
   conf >> modelname;

   //summary statistics yes/no
   bool summarystats;
   conf >> summarystats;

   //number of calibration parameters (theta's) for eta(x,theta), must be >0 obviously.
   size_t pthetas;
   conf >> pthetas;

   //index of calibration parameters for eta(x,theta)
   size_t thtemp;
   std::vector<size_t> thetaidx(pthetas);
   for(size_t i=0;i<pthetas;i++) {
      conf >> thtemp;
      thetaidx[i]=thtemp;
   }

   //vector of calibration parameter prior type (1=Normal,0=Uniform)
   bool thtemp2;
   std::vector<bool> th_type(pthetas);
   for(size_t i=0;i<pthetas;i++) {
      conf >> thtemp2;
      th_type[i]=thtemp2;
   }

   //vector of calibration parameter prior means/lower bound
   double thtemp3;
   std::vector<double> th_prior_a(pthetas);
   for(size_t i=0;i<pthetas;i++){
      conf >> thtemp3;
      th_prior_a[i]=thtemp3;
   }

   //vector of calibration parameter prior sd/upper bound
   std::vector<double> th_prior_b(pthetas);
   for(size_t i=0;i<pthetas;i++){
      conf >> thtemp3;
      th_prior_b[i]=thtemp3;
   }

   bool dopert=true;
   bool doperth=true;
   if(probchv<0) dopert=false;
   if(probchvh<0) doperth=false;

   // done loading configuration
   conf.close();


   //folder
//   std::string folder("." + modelname + "/");

   //MPI initialization
   int mpirank=0;
#ifdef _OPENMPI
   int mpitc;
   MPI_Init(NULL,NULL);
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Comm_rank(MPI_COMM_WORLD,&mpirank);
   MPI_Comm_size(MPI_COMM_WORLD,&mpitc);
#ifndef SILENT
   cout << "\nMPI: node " << mpirank << " of " << mpitc << " processes." << endl;
#endif
   if(tc<=1) return 0; //need at least 2 processes!
   if(tc!=mpitc) return 0; //mismatch between how MPI was started and how the data is prepared according to tc.
// #else
//    if(tc!=1) return 0; //serial mode should have no slave threads!
#endif


   //--------------------------------------------------
   // Banner
   if(mpirank==0) {
      cout << endl;
      cout << "-----------------------------------" << endl;
      cout << "OpenBT calibration model CLI (calibrate)" << endl;
      cout << "Loading config file at " << folder << endl;
   }


   //--------------------------------------------------
   //setup calibration parameter matrix, initialize the first row.
   std::vector<std::vector<double> > thetas_adapt(nadapt,std::vector<double>(pthetas));
   std::vector<std::vector<double> > thetas_burn(burn,std::vector<double>(pthetas));
   std::vector<std::vector<double> > thetas(nd,std::vector<double>(pthetas));
   std::vector<unsigned int> accept_thetas(pthetas,0);
   std::vector<unsigned int> reject_thetas(pthetas,0);

   for(size_t i=0;i<pthetas;i++) {
      if(th_type[i]) //Normal prior, initialize to prior mean
         thetas_adapt[0][i]=th_prior_a[i];
      else           //Uniform prior, initialize to prior mean
         thetas_adapt[0][i]=th_prior_a[i]+(th_prior_b[i]-th_prior_a[i])/2.0;
      if(mpirank==0) {
         cout << "Initialized theta" << i << "=" << thetas_adapt[0][i];
         cout << " (range=" << th_prior_a[i] << "," << th_prior_b[i] << ")" << endl;
      }
   }

   //--------------------------------------------------
   //read in yc
   std::vector<double> yc;
   double ytemp;
   size_t n=0;
#ifdef _OPENMPI
   if(mpirank>0) { //only load data on slaves
#endif
      std::stringstream yfss;
      std::string yfs;
      yfss << folder << ycore << mpirank;
      yfs=yfss.str();
      std::ifstream ystream(yfs);
      while(ystream >> ytemp)
         yc.push_back(ytemp);
      n=yc.size();
#ifndef SILENT
      cout << "node " << mpirank << " loaded " << n << " from " << yfs <<endl;
#endif
#ifdef _OPENMPI
   }
#endif


   //--------------------------------------------------
   //read in xc
   std::vector<double> xc;
   double xtemp;
   size_t p=0;
#ifdef _OPENMPI
   if(mpirank>0) {
#endif
      std::stringstream xfss;
      std::string xfs;
      xfss << folder << xcore << mpirank;
      xfs=xfss.str();
      std::ifstream xstream(xfs);
      while(xstream >> xtemp)
         xc.push_back(xtemp);
      p = xc.size()/n;
#ifndef SILENT
      cout << "node " << mpirank << " loaded " << n << " inputs of dimension " << p << " from " << xfs << endl;
#endif
#ifdef _OPENMPI
   }
   int tempp = (unsigned int) p;
   MPI_Allreduce(MPI_IN_PLACE,&tempp,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
   if(mpirank>0 && p != ((size_t) tempp)) { cout << "PROBLEM LOADING DATA (xc)" << endl; MPI_Finalize(); return 0;}
   p=(size_t)tempp;
#endif


   //--------------------------------------------------
   //read in yf
   std::vector<double> yf;
   size_t nf=0;
#ifdef _OPENMPI
   if(mpirank>0) { //only load data on slaves
#endif
      std::stringstream yfss;
      std::string yfs;
      yfss << folder << yfcore << mpirank;
      yfs=yfss.str();
      std::ifstream ystream(yfs);
      while(ystream >> ytemp)
         yf.push_back(ytemp);
      nf=yf.size();
#ifndef SILENT
      cout << "node " << mpirank << " loaded " << nf << " from " << yfs <<endl;
#endif
#ifdef _OPENMPI
   }
#endif


   //--------------------------------------------------
   //read in xf
   std::vector<double> xf;
   size_t pdelta=0;
#ifdef _OPENMPI
   if(mpirank>0) {
#endif
      std::stringstream xfss;
      std::string xfs;
      xfss << folder << xfcore << mpirank;
      xfs=xfss.str();
      std::ifstream xstream(xfs);
      while(xstream >> xtemp)
         xf.push_back(xtemp);
      pdelta = xf.size()/nf;
#ifndef SILENT
      cout << "node " << mpirank << " loaded " << nf << " inputs of dimension " << pdelta << " from " << xfs << endl;
#endif
#ifdef _OPENMPI
   }
   tempp = (unsigned int) pdelta;
   MPI_Allreduce(MPI_IN_PLACE,&tempp,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
   if(mpirank>0 && pdelta != ((size_t) tempp)) { cout << "PROBLEM LOADING DATA (xf)" << endl; MPI_Finalize(); return 0;}
   pdelta=(size_t)tempp;
   if(p!=pthetas+pdelta) { cout << "PROBLEM LOADING DATA (xf)" << endl; MPI_Finalize(); return 0;}
#endif


   //--------------------------------------------------
   //dinfo's
   dinfo dic;
   dic.n=0;dic.p=p,dic.x = NULL;dic.y=NULL;dic.tc=tc;
#ifdef _OPENMPI
   if(mpirank>0) { 
#endif
      dic.n=n; dic.x = &xc[0]; dic.y = &yc[0];
#ifdef _OPENMPI
   }
#endif

   dinfo dif;
   dif.n=0;dif.p=pdelta,dif.x = NULL;dif.y=NULL;dif.tc=tc;
#ifdef _OPENMPI
   if(mpirank>0) { 
#endif
      dif.n=nf; dif.x = &xf[0]; dif.y = &yf[0];
#ifdef _OPENMPI
   }
#endif


   //--------------------------------------------------
   //xfpred and dinfo for prediction of eta(xfpred,theta)
   //(xfpred is just xf but in the p-dim space not the pdelta-dim space)
   std::vector<double> xpred;
   size_t ppred=0;
#ifdef _OPENMPI
   if(mpirank>0) {
#endif
      std::stringstream xfsspred;
      std::string xfspred;
      xfsspred << folder << xpcore << mpirank;
      xfspred=xfsspred.str();
      std::ifstream xpredstream(xfspred);
      while(xpredstream >> xtemp)
         xpred.push_back(xtemp);
      ppred = xpred.size()/nf;
#ifndef SILENT
      cout << "node " << mpirank << " loaded " << nf << " inputs of dimension " << ppred << " from " << xfspred << endl;
#endif
#ifdef _OPENMPI
   }
   tempp = (unsigned int) ppred;
   MPI_Allreduce(MPI_IN_PLACE,&tempp,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
   if(mpirank>0 && ppred != ((size_t) tempp)) { cout << "PROBLEM LOADING DATA (xpred)" << endl; MPI_Finalize(); return 0;}
   ppred=(size_t)tempp;
   if(p!=ppred) { cout << "PROBLEM LOADING DATA (xpred)" << endl; MPI_Finalize(); return 0;}
#endif

   // Initialize appropriate columns of xpred to the initialized calibration parameters.
#ifdef _OPENMPI
   if(mpirank>0) {
#endif
      updateall(xpred,thetas_adapt[0],thetaidx,pthetas,p,nf);
#ifdef _OPENMPI
   }
#endif

//   double *fp = new double[nf];
   std::vector<double> fp(nf);
   dinfo dip;
   dip.x = NULL; dip.y=NULL; dip.p = p; dip.n=0; dip.tc=1;
#ifdef _OPENMPI
   if(mpirank>0) {
#endif
   dip.x = &xpred[0]; dip.y=&fp[0]; dip.n=nf;

#ifdef _OPENMPI      
   }
#endif

//   double *fp2 = new double[nf];
   std::vector<double> fp2(nf);
   dinfo dip2;
   dip2.x = NULL; dip2.y=NULL; dip2.p = pdelta; dip2.n=0; dip2.tc=1;
#ifdef _OPENMPI
   if(mpirank>0) {
#endif
   dip2.x = &xf[0]; dip2.y=&fp2[0]; dip2.n=nf;
#ifdef _OPENMPI      
   }
#endif

//   double *sigfp = new double[nf];
   std::vector<double> sigfp(nf);
   dinfo dip3;
   dip3.x = NULL; dip3.y=NULL; dip3.p = pdelta; dip3.n=0; dip3.tc=1;
#ifdef _OPENMPI
   if(mpirank>0) {
#endif
   dip3.x = &xf[0]; dip3.y=&sigfp[0]; dip3.n=nf;
#ifdef _OPENMPI      
   }
#endif

   //--------------------------------------------------
   //read in sigmavc  -- same as above.
   std::vector<double> sigmavc;
   double stemp;
   size_t nsigc=0;
#ifdef _OPENMPI
   if(mpirank>0) { //only load data on slaves
#endif
      std::stringstream sfss;
      std::string sfs;
      sfss << folder << scorec << mpirank;
      sfs=sfss.str();
      std::ifstream sf(sfs);
      while(sf >> stemp)
         sigmavc.push_back(stemp);
      nsigc=sigmavc.size();
#ifndef SILENT
      cout << "node " << mpirank << " loaded " << nsigc << " from " << sfs <<endl;
#endif
#ifdef _OPENMPI
      if(n!=nsigc) { cout << "PROBLEM LOADING SIGMAV" << endl; MPI_Finalize(); return 0; }
   }
#else
   if(n!=nsigc) { cout << "PROBLEM LOADING SIGMAV" << endl; return 0; }
#endif

//   double *sigc=&sigmavc[0];
   dinfo disigc;
   disigc.n=0; disigc.p=p; disigc.x=NULL; disigc.y=NULL; disigc.tc=tc;
#ifdef _OPENMPI
   if(mpirank>0) { 
#endif
      disigc.n=n; disigc.x=&xc[0]; disigc.y=&sigmavc[0]; 
#ifdef _OPENMPI
   }
#endif


   //--------------------------------------------------
   //read in sigmavf  -- same as above.
   std::vector<double> sigmavf;
   size_t nsigf=0;
#ifdef _OPENMPI
   if(mpirank>0) { //only load data on slaves
#endif
      std::stringstream sfss;
      std::string sfs;
      sfss << folder << scoref << mpirank;
      sfs=sfss.str();
      std::ifstream sf(sfs);
      while(sf >> stemp)
         sigmavf.push_back(stemp);
      nsigf=sigmavf.size();
#ifndef SILENT
      cout << "node " << mpirank << " loaded " << nsigf << " from " << sfs <<endl;
#endif
#ifdef _OPENMPI
      if(nf!=nsigf) { cout << "PROBLEM LOADING SIGMAV" << endl; MPI_Finalize(); return 0; }
   }
#else
   if(nf!=nsigf) { cout << "PROBLEM LOADING SIGMAV" << endl; return 0; }
#endif

//   double *sigf=&sigmavf[0];
   dinfo disigf;
   disigf.n=0; disigf.p=pdelta; disigf.x=NULL; disigf.y=NULL; disigf.tc=tc;
#ifdef _OPENMPI
   if(mpirank>0) { 
#endif
      disigf.n=nf; disigf.x=&xf[0]; disigf.y=&sigmavf[0]; 
#ifdef _OPENMPI
   }
#endif


   //--------------------------------------------------
   // read in the initial change of variable rank correlation matrix for eta(x,theta)
   std::vector<std::vector<double>> chgvc;
   std::vector<double> cvvtemp;
   double cvtemp;
   std::ifstream chgvstream(folder + chgvcorec);
   for(size_t i=0;i<dic.p;i++) {
      cvvtemp.clear();
      for(size_t j=0;j<dic.p;j++) {
         chgvstream >> cvtemp;
         cvvtemp.push_back(cvtemp);
      }
      chgvc.push_back(cvvtemp);
   }
#ifndef SILENT
   cout << "mpirank=" << mpirank << ": change of variable rank correlation matrix loaded:" << endl;
#endif
   // if(mpirank==0) //print it out:
   //    for(size_t i=0;i<dic.p;i++) {
   //       for(size_t j=0;j<dic.p;j++)
   //          cout << "(" << i << "," << j << ")" << chgvc[i][j] << "  ";
   //       cout << endl;
   //    }



   //--------------------------------------------------
   // read in the initial change of variable rank correlation matrix for delta(x)
   std::vector<std::vector<double>> chgvf;
   std::ifstream chgvfstream(folder + chgvcoref);
   for(size_t i=0;i<dif.p;i++) {
      cvvtemp.clear();
      for(size_t j=0;j<dif.p;j++) {
         chgvfstream >> cvtemp;
         cvvtemp.push_back(cvtemp);
      }
      chgvf.push_back(cvvtemp);
   }
#ifndef SILENT
   cout << "mpirank=" << mpirank << ": change of variable rank correlation matrix loaded:" << endl;
#endif
   // if(mpirank==0) //print it out:
   //    for(size_t i=0;i<dif.p;i++) {
   //       for(size_t j=0;j<dif.p;j++)
   //          cout << "(" << i << "," << j << ")" << chgvf[i][j] << "  ";
   //       cout << endl;
   //    }



   //--------------------------------------------------
   // decide what variables each slave node will update in change-of-variable proposals.
#ifdef _OPENMPI
   int* lwrc=new int[tc];
   int* uprc=new int[tc];
   lwrc[0]=-1; uprc[0]=-1;
   for(size_t i=1;i<(size_t)tc;i++) { 
      lwrc[i]=-1; uprc[i]=-1; 
      calcbegend(p,i-1,tc-1,&lwrc[i],&uprc[i]);
      if(p>1 && lwrc[i]==0 && uprc[i]==0) { lwrc[i]=-1; uprc[i]=-1; }
   }

#ifndef SILENT
   if(mpirank>0) cout << "Slave node " << mpirank << " will update eta variables " << lwrc[mpirank] << " to " << uprc[mpirank]-1 << endl;
#endif
#endif

#ifdef _OPENMPI
   int* lwrf=new int[tc];
   int* uprf=new int[tc];
   lwrf[0]=-1; uprf[0]=-1;
   for(size_t i=1;i<(size_t)tc;i++) { 
      lwrf[i]=-1; uprf[i]=-1; 
      calcbegend(pdelta,i-1,tc-1,&lwrf[i],&uprf[i]);
      if(pdelta>1 && lwrf[i]==0 && uprf[i]==0) { lwrf[i]=-1; uprf[i]=-1; }
   }

#ifndef SILENT
   if(mpirank>0) cout << "Slave node " << mpirank << " will update delta variables " << lwrf[mpirank] << " to " << uprf[mpirank]-1 << endl;
#endif
#endif


   //--------------------------------------------------
   //print args
#ifndef SILENT
   cout << "**********************\n";
   cout << "n: " << n << endl;
   cout << "p: " << p << endl;
   cout << "nf: " << nf << endl;
   cout << "pdelta: " << pdelta << endl;
   cout << "pthetas: " << pthetas << endl;
   if(mpirank>0) cout << "first row: " << xc[0] << ", " << xc[p-1] << endl;
   if(mpirank>0) cout << "second row: " << xc[p] << ", " << xc[2*p-1] << endl;
   if(mpirank>0) cout << "last row: " << xc[(n-1)*p] << ", " << xc[n*p-1] << endl;
   if(mpirank>0) cout << "first and last y: " << yc[0] << ", " << yc[n-1] << endl;
   if(mpirank>0) cout << "first row: " << xf[0] << ", " << xf[pdelta-1] << endl;
   if(mpirank>0) cout << "second row: " << xf[pdelta] << ", " << xf[2*pdelta-1] << endl;
   if(mpirank>0) cout << "last row: " << xf[(nf-1)*pdelta] << ", " << xf[nf*pdelta-1] << endl;
   if(mpirank>0) cout << "first and last y: " << yf[0] << ", " << yf[nf-1] << endl;
   cout << "number of trees mean: " << mc << endl;
   cout << "number of trees stan dev: " << mhc << endl;
   cout << "number of trees mean: " << mf << endl;
   cout << "number of trees stan dev: " << mhf << endl;
   cout << "tauc: " << tauc << endl;
   cout << "overalllambdac: " << overalllambdac << endl;
   cout << "overallnuc: " << overallnuc << endl;
   cout << "tauf: " << tauf << endl;
   cout << "overalllambdaf: " << overalllambdaf << endl;
   cout << "overallnuf: " << overallnuf << endl;
   cout << "burn (nskip): " << burn << endl;
   cout << "nd (ndpost): " << nd << endl;
   cout << "nadapt: " << nadapt << endl;
   cout << "adaptevery: " << adaptevery << endl;
   cout << "mean tree prior base: " << alpha << endl;
   cout << "mean tree prior power: " << mybeta << endl;
   cout << "variance tree prior base: " << alphah << endl;
   cout << "variance tree prior power: " << mybetah << endl;
   cout << "thread count: " << tc << endl;
   if(mpirank>0) cout << "first and last sigmavc: " << sigmavc[0] << ", " << sigmavc[n-1] << endl;
   if(mpirank>0) cout << "first and last sigmavf: " << sigmavf[0] << ", " << sigmavf[nf-1] << endl;
   cout << "chgvc first row: " << chgvc[0][0] << ", " << chgvc[0][p-1] << endl;
   cout << "chgvc last row: " << chgvc[p-1][0] << ", " << chgvc[p-1][p-1] << endl;
   cout << "chgvf first row: " << chgvf[0][0] << ", " << chgvf[0][pdelta-1] << endl;
   cout << "chgvf last row: " << chgvf[pdelta-1][0] << ", " << chgvf[pdelta-1][pdelta-1] << endl;
   cout << "mean trees prob birth/death (eta): " << pbdc << endl;
   cout << "mean trees prob birth/death (delta): " << pbdf << endl;
   cout << "mean trees prob birth: " << pb << endl;
   cout << "variance trees prob birth/death (eta): " << pbdch << endl;
   cout << "variance trees prob birth/death (delta): " << pbdfh << endl;
//   cout << "variance trees prob birth: " << pbh << endl;
   cout << "mean trees initial step width pert move: " << stepwpert << endl;
   cout << "variance trees initial step width pert move: " << stepwperth << endl;
   cout << "mean trees prob of a change var move : " << probchv << endl;
   cout << "variance trees prob of a change var move : " << probchvh << endl;
   cout << "mean trees min num obs in bottom node: " << minnumbot << endl;
   cout << "variance trees min num obs in bottom node: " << minnumboth << endl;
   cout << "*****printevery: " << printevery << endl;
#endif


   //--------------------------------------------------
   //make xinfo for eta(x,theta)
   xinfo xic;
   xic.resize(p);

   for(size_t i=0;i<p;i++) {
      std::vector<double> xivec;
      double xitemp;

      std::stringstream xifss;
      std::string xifs;
      xifss << folder << xicorec << (i+1);
      xifs=xifss.str();
      std::ifstream xistream(xifs);
      while(xistream >> xitemp)
         xivec.push_back(xitemp);
      xic[i]=xivec;
   }
#ifndef SILENT
   cout << "&&& made xinfo\n";
#endif

   //summarize input variables:
#ifndef SILENT
   for(size_t i=0;i<p;i++)
   {
      cout << "Variable " << i << " has numcuts=" << xic[i].size() << " : ";
      cout << xic[i][0] << " ... " << xic[i][xic[i].size()-1] << endl;
   }
#endif


   //--------------------------------------------------
   //make xinfo for delta(x)
   xinfo xif;
   xif.resize(pdelta);

   for(size_t i=0;i<pdelta;i++) {
      std::vector<double> xivec;
      double xitemp;

      std::stringstream xifss;
      std::string xifs;
      xifss << folder << xicorec << (i+1);
      xifs=xifss.str();
      std::ifstream xistream(xifs);
      while(xistream >> xitemp)
         xivec.push_back(xitemp);
      xif[i]=xivec;
   }
#ifndef SILENT
   cout << "&&& made xinfo\n";
#endif

   //summarize input variables:
#ifndef SILENT
   for(size_t i=0;i<pdelta;i++)
   {
      cout << "Variable " << i << " has numcuts=" << xif[i].size() << " : ";
      cout << xif[i][0] << " ... " << xif[i][xif[i].size()-1] << endl;
   }
#endif



   //--------------------------------------------------
   // set up emulator objects
   ambrt ambm(mc);

   //cutpoints
   ambm.setxi(&xic);    //set the cutpoints for this model object
   //data objects
   ambm.setdata(&dic);  //set the data
   //thread count
   ambm.settc(tc-1);      //set the number of slaves when using MPI.
   //mpi rank
#ifdef _OPENMPI
   ambm.setmpirank(mpirank);  //set the rank when using MPI.
   ambm.setmpicvrange(lwrc,uprc); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
   //tree prior
   ambm.settp(alpha, //the alpha parameter in the tree depth penalty prior
         mybeta,     //the beta parameter in the tree depth penalty prior
         false,
         0.0,
         0.5,
         false
         );
   //MCMC info
   ambm.setmi(
         pbdc,  //probability of birth/death
         pb,  //probability of birth
         minnumbot,    //minimum number of observations in a bottom node
         dopert, //do perturb/change variable proposal?
         stepwpert,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
         probchv,  //probability of doing a change of variable proposal.  perturb prob=1-this.
         &chgvc  //initialize the change of variable correlation matrix.
         );
   ambm.setci(tauc,&sigmavc[0]);//sigc);

   //setup psbrt object
   psbrt psbm(mhc,overalllambdac);

   //make di for psbrt object
   dinfo dipsc;
   dipsc.n=0; dipsc.p=p; dipsc.x=NULL; dipsc.y=NULL; dipsc.tc=tc;
   std::vector<double> rc;
   // double *rc=NULL;
#ifdef _OPENMPI
   if(mpirank>0) {
#endif
      // rc = new double[n];
      rc.resize(n);
      rc=sigmavc;
      // for(size_t i=0;i<n;i++) rc[i]=sigmavc[i];
      dipsc.x=&xc[0]; dipsc.y=&rc[0]; dipsc.n=n;
#ifdef _OPENMPI
   }
#endif

   double opm=1.0/((double)mhc);
   double nu=2.0*pow(overallnuc,opm)/(pow(overallnuc,opm)-pow(overallnuc-2.0,opm));
   double lambdac=pow(overalllambdac,opm);

   //cutpoints
   psbm.setxi(&xic);    //set the cutpoints for this model object
   //data objects
   psbm.setdata(&dipsc);  //set the data
   //thread count
   psbm.settc(tc-1); 
   //mpi rank
#ifdef _OPENMPI
   psbm.setmpirank(mpirank);  //set the rank when using MPI.
   psbm.setmpicvrange(lwrc,uprc); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
   //tree prior
   psbm.settp(alphah, //the alpha parameter in the tree depth penalty prior
         mybetah,     //the beta parameter in the tree depth penalty prior
         false,
         0.0,
         0.5,
         false
         );
   psbm.setmi(
         pbdch,  //probability of birth/death
         pb,  //probability of birth
         minnumboth,    //minimum number of observations in a bottom node
         doperth, //do perturb/change variable proposal?
         stepwperth,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
         probchvh,  //probability of doing a change of variable proposal.  perturb prob=1-this.
         &chgvc  //initialize the change of variable correlation matrix.
         );
   psbm.setci(nu,lambdac);


   //--------------------------------------------------
   // set up discrepancy objects
   ambrt ambmf(mf);

   //cutpoints
   ambmf.setxi(&xif);    //set the cutpoints for this model object
   //data objects
   ambmf.setdata(&dif);  //set the data
   //thread count
   ambmf.settc(tc-1);      //set the number of slaves when using MPI.
   //mpi rank
#ifdef _OPENMPI
   ambmf.setmpirank(mpirank);  //set the rank when using MPI.
   ambmf.setmpicvrange(lwrf,uprf); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
   //tree prior
   ambmf.settp(alpha, //the alpha parameter in the tree depth penalty prior
         mybeta,     //the beta parameter in the tree depth penalty prior
         false,
         0.0,
         0.5,
         false
         );
   //MCMC info
   ambmf.setmi(
         pbdf,  //probability of birth/death
         pb,  //probability of birth
         minnumbot,    //minimum number of observations in a bottom node
         dopert, //do perturb/change variable proposal?
         stepwpert,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
         probchv,  //probability of doing a change of variable proposal.  perturb prob=1-this.
         &chgvf  //initialize the change of variable correlation matrix.
         );
   ambmf.setci(tauf,&sigmavf[0]);//sigf);

   //setup psbrt object
   psbrt psbmf(mhf,overalllambdaf);

   //make di for discrepancy psbrt object
   dinfo dipsf;
   dipsf.n=0; dipsf.p=pdelta; dipsf.x=NULL; dipsf.y=NULL; dipsf.tc=tc;
   // double *rf=NULL;
   std::vector<double> rf;
#ifdef _OPENMPI
   if(mpirank>0) {
#endif
      // rf = new double[nf];
      // for(size_t i=0;i<nf;i++) rf[i]=sigmavf[i];
      rf.resize(nf);
      rf=sigmavf;
      dipsf.x=&xf[0]; dipsf.y=&rf[0]; dipsf.n=nf;
#ifdef _OPENMPI
   }
#endif

   double opmf=1.0/((double)mhf);
   double nuf=2.0*pow(overallnuf,opmf)/(pow(overallnuf,opmf)-pow(overallnuf-2.0,opmf));
   double lambdaf=pow(overalllambdaf,opmf);
   double valformarg=0.5*overallnuf*overalllambdaf;//0.1;//2.0/nuf/lambdaf; // need this for calibration param MH step when using mariginal likelihood
cout << "opmf=" << opmf << " nuf=" << nuf << " lambdaf=" << lambdaf << "valformarg=" << valformarg << endl;
   //cutpoints
   psbmf.setxi(&xif);    //set the cutpoints for this model object
   //data objects
   psbmf.setdata(&dipsf);  //set the data
   //thread count
   psbmf.settc(tc-1); 
   //mpi rank
#ifdef _OPENMPI
   psbmf.setmpirank(mpirank);  //set the rank when using MPI.
   psbmf.setmpicvrange(lwrf,uprf); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
   //tree prior
   psbmf.settp(alphah, //the alpha parameter in the tree depth penalty prior
         mybetah,     //the beta parameter in the tree depth penalty prior
         false,
         0.0,
         0.5,
         false
         );
   psbmf.setmi(
         pbdfh,  //probability of birth/death
         pb,  //probability of birth
         minnumboth,    //minimum number of observations in a bottom node
         doperth, //do perturb/change variable proposal?
         stepwperth,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
         probchvh,  //probability of doing a change of variable proposal.  perturb prob=1-this.
         &chgvf  //initialize the change of variable correlation matrix.
         );
   psbmf.setci(nuf,lambdaf);



   //--------------------------------------------------
   //run mcmc
   std::vector<int> onn(nd*mc,1);
   std::vector<std::vector<int> > oid(nd*mc, std::vector<int>(1));
   std::vector<std::vector<int> > ovar(nd*mc, std::vector<int>(1));
   std::vector<std::vector<int> > oc(nd*mc, std::vector<int>(1));
   std::vector<std::vector<double> > otheta(nd*mc, std::vector<double>(1));
   std::vector<int> snn(nd*mhc,1);
   std::vector<std::vector<int> > sid(nd*mhc, std::vector<int>(1));
   std::vector<std::vector<int> > svar(nd*mhc, std::vector<int>(1));
   std::vector<std::vector<int> > sc(nd*mhc, std::vector<int>(1));
   std::vector<std::vector<double> > stheta(nd*mhc, std::vector<double>(1));
   std::vector<int> onnf(nd*mf,1);
   std::vector<std::vector<int> > oidf(nd*mf, std::vector<int>(1));
   std::vector<std::vector<int> > ovarf(nd*mf, std::vector<int>(1));
   std::vector<std::vector<int> > ocf(nd*mf, std::vector<int>(1));
   std::vector<std::vector<double> > othetaf(nd*mf, std::vector<double>(1));
   std::vector<int> snnf(nd*mhf,1);
   std::vector<std::vector<int> > sidf(nd*mhf, std::vector<int>(1));
   std::vector<std::vector<int> > svarf(nd*mhf, std::vector<int>(1));
   std::vector<std::vector<int> > scf(nd*mhf, std::vector<int>(1));
   std::vector<std::vector<double> > sthetaf(nd*mhf, std::vector<double>(1));

   brtMethodWrapper fambm(&brt::f,ambm);
   brtMethodWrapper fpsbm(&brt::f,psbm);
   brtMethodWrapper fambmf(&brt::f,ambmf);
   brtMethodWrapper fpsbmf(&brt::f,psbmf);

   std::vector<double> propwidth(pthetas,0.25);

   std::vector<double> buf(nf);
   double ssold,ssnew;
   double lalpha;
   double logprior;

   for(size_t i=0;i<pthetas;i++) {
      if(th_type[i]) //Normal prior, initialize to 25% of +/- 2sd
         propwidth[i]=4.0*th_prior_b[i]*0.25;
      else           //Uniform prior, initialize to 25% of range
         propwidth[i]=(th_prior_b[i]-th_prior_a[i])*0.25;

      if(mpirank==0) cout << "Initialized proposal width for theta" << i << " to " << propwidth[i] << endl;
   }


#ifdef _OPENMPI
   double tstart=0.0,tend=0.0;
   if(mpirank==0) tstart=MPI_Wtime();
   if(mpirank==0) cout << "Starting MCMC..." << endl;
#else
   cout << "Starting MCMC..." << endl;
#endif

   // Adapt Iterations
   for(size_t i=1;i<nadapt;i++) { 
      if((i % printevery) ==0 && mpirank==0) cout << "Adapt iteration " << i << endl;

      // Draw BART model for eta(x,theta)
#ifdef _OPENMPI
      if(mpirank==0) ambm.draw(gen); else ambm.draw_mpislave(gen);
#else
      ambm.draw(gen);
#endif
      dipsc = dic;
      dipsc -= fambm;
      if((i+1)%adaptevery==0 && mpirank==0) ambm.adapt();
      if(modeltypec!=MODEL_PROBIT) {
#ifdef _OPENMPI
         if(mpirank==0) psbm.draw(gen); else psbm.draw_mpislave(gen);
#else
         psbm.draw(gen);
#endif
         disigc = fpsbm;
      }
      if((i+1)%adaptevery==0 && mpirank==0) psbm.adapt();      

      // Draw theta
#ifdef _OPENMPI
      thetas_adapt[i]=thetas_adapt[i-1];
      updateall(xpred,thetas_adapt[i],thetaidx,pthetas,p,nf);
     // size_t j=(size_t)std::floor(gen.uniform()*pthetas);
      // for(size_t j=0;j<pthetas;j++) {
         if(mpirank>0) {
            MPI_Status status;
            std::fill(fp.begin(),fp.end(),0.0);
            std::fill(sigfp.begin(),sigfp.end(),0.0);
            ambm.predict(&dip);   //eta(x,theta) -> g(x,theta)
            ambmf.predict(&dip2);
            psbmf.predict(&dip3);  //sigmaf(x)

            // (yf-eta(x,theta))/sigf
            std::transform(yf.begin(),yf.end(),fp.begin(),buf.begin(),std::minus<double>());
            std::transform(buf.begin(),buf.end(),fp2.begin(),buf.begin(),std::minus<double>());
            // std::transform(buf.begin(),buf.end(),sigfp.begin(),buf.begin(),std::divides<double>());
//            buf=yf-fp; buf/=sigfp;

            // criterion calculation for current theta
            ssold=0.0;
            // for(size_t k=0;k<nf;k++) ssold+=buf[k];
            // ssold=-1.0*ssold*ssold/4.0*(1.0/(nf/dip3.y[0]/dip3.y[0]+1.0/tauf/tauf));
            for(size_t k=0;k<nf;k++) ssold+=buf[k]*buf[k];

            // draw proposed theta_j
            for(size_t j=0;j<pthetas;j++) {
               if(th_type[j]) //Normal prior
                  thetas_adapt[i][j]=thetas_adapt[i-1][j]+propwidth[j]*gen.normal();
               else //Uniform prior
                  thetas_adapt[i][j]=thetas_adapt[i-1][j]+propwidth[j]*(gen.uniform()-0.5);
            }
            updateall(xpred,thetas_adapt[i],thetaidx,pthetas,p,nf);

            std::fill(fp.begin(),fp.end(),0.0);
            ambm.predict(&dip);   //eta(x,thetaprime) -> g(x,thetaprime)

            // (yf-eta(x,theta))/sigf
            std::transform(yf.begin(),yf.end(),fp.begin(),buf.begin(),std::minus<double>());
            std::transform(buf.begin(),buf.end(),fp2.begin(),buf.begin(),std::minus<double>());
            // std::transform(buf.begin(),buf.end(),sigfp.begin(),buf.begin(),std::divides<double>());
//            buf=yf-fp; buf/=sigfp;

            // criterion calculation for proposed theta
            ssnew=0.0;
            // for(size_t k=0;k<nf;k++) ssnew+=buf[k];
            // ssnew=-1.0*ssnew*ssnew/4.0*(1.0/(nf/dip3.y[0]/dip3.y[0]+1.0/tauf/tauf));
            for(size_t k=0;k<nf;k++) ssnew+=buf[k]*buf[k];

            // lalpha=(ssnew-ssold);
            MPI_Reduce(&ssold,NULL,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&ssnew,NULL,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Recv(NULL,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            if(status.MPI_TAG==MPI_TAG_CAL_ACCEPT) {
               for(size_t j=0;j<pthetas;j++) accept_thetas[j]++;
            }
            else {
               thetas_adapt[i]=thetas_adapt[i-1];
               updateall(xpred,thetas_adapt[i],thetaidx,pthetas,p,nf);
               for(size_t j=0;j<pthetas;j++) reject_thetas[j]++;
            }
         }
         if(mpirank==0) {
            MPI_Request *request = new MPI_Request[tc];
               logprior=0.0;
               for(size_t j=0;j<pthetas;j++) {
                  if(th_type[j]) //Normal prior
                     thetas_adapt[i][j]=thetas_adapt[i-1][j]+propwidth[j]*gen.normal();
                  else //Uniform prior
                     thetas_adapt[i][j]=thetas_adapt[i-1][j]+propwidth[j]*(gen.uniform()-0.5);

                  if(th_type[j]) { //Normal prior
                     logprior += -1.0/2.0/th_prior_b[j]/th_prior_b[j]*(thetas_adapt[i][j]-th_prior_a[j])*(thetas_adapt[i][j]-th_prior_a[j])
                     +1.0/2.0/th_prior_b[j]/th_prior_b[j]*(thetas_adapt[i-1][j]-th_prior_a[j])*(thetas_adapt[i-1][j]-th_prior_a[j]);
                  }
                  else { //Uniform prior
                     logprior += 0.0;
                     if(thetas_adapt[i][j]<th_prior_a[j] || thetas_adapt[i][j]>th_prior_b[j])
                        logprior=-std::numeric_limits<double>::infinity();
                  }
               }
               lalpha=0.0; ssold=0.0; ssnew=0.0;
               MPI_Reduce(MPI_IN_PLACE,&ssold,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
               MPI_Reduce(MPI_IN_PLACE,&ssnew,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
               //tempered: lalpha=-1.0/2.0*lalpha*((double)(1.0/(std::log(nadapt)-std::log(i)+1.0)))+logprior;
               // lalpha=-1.0/2.0*lalpha+logprior;
               lalpha=(overallnuf+nf)/2.0*std::log(valformarg+0.5*ssold)
                        -(overallnuf+nf)/2.0*std::log(valformarg+0.5*ssnew);
               lalpha=lalpha+logprior;
               lalpha=std::min(0.0,lalpha);

               double alpha=gen.uniform();
               if(log(alpha)<lalpha) //accept
               {
                  for(size_t j=0;j<pthetas;j++) accept_thetas[j]++;
                  const int tag=MPI_TAG_CAL_ACCEPT;
                  for(size_t k=1; k<(size_t)tc; k++) {
                     MPI_Isend(NULL,0,MPI_PACKED,k,tag,MPI_COMM_WORLD,&request[k-1]);
                  }
               }
               else { //reject
                  thetas_adapt[i]=thetas_adapt[i-1];
                  const int tag=MPI_TAG_CAL_REJECT;
                  for(size_t k=1; k<(size_t)tc; k++) {
                     MPI_Isend(NULL,0,MPI_PACKED,k,tag,MPI_COMM_WORLD,&request[k-1]);
                  }
                  for(size_t j=0;j<pthetas;j++) reject_thetas[j]++;
               }
               MPI_Waitall(tc-1,request,MPI_STATUSES_IGNORE);
               delete[] request;
         }
      // }
      if((i+1)%adaptevery==0) {
         double accrate;
         for(size_t j=0;j<pthetas;j++) {
            if(mpirank==0) cout << "Acceptance rate=" << ((double)accept_thetas[j])/((double)(accept_thetas[j]+reject_thetas[j]));
            accrate=((double)accept_thetas[j])/((double)(accept_thetas[j]+reject_thetas[j]));
            if(accrate>0.29 || accrate<0.19) propwidth[j]*=accrate/0.24;
            if(mpirank==0) cout << " (adapted propwidth to " << propwidth[j] << ")" << endl;
         }
         std::fill(accept_thetas.begin(),accept_thetas.end(),0);
         std::fill(reject_thetas.begin(),reject_thetas.end(),0);
      }
#else
     // openmp version of draw thetas.
#endif

      // Draw BART model for delta(x)
      std::fill(fp.begin(),fp.end(),0.0);
      updateall(xpred,thetas_adapt[i],thetaidx,pthetas,p,nf);
      ambm.predict(&dip); // predict eta(x,theta[i]) and store it in dip
      std::transform(yf.begin(),yf.end(),fp.begin(),yf.begin(),std::minus<double>());
//      yf-=fp;

#ifdef _OPENMPI
      if(mpirank==0) ambmf.draw(gen); else ambmf.draw_mpislave(gen);
#else
      ambmf.draw(gen);
#endif
      dipsf = dif;
      dipsf -= fambmf;
      std::transform(yf.begin(),yf.end(),fp.begin(),yf.begin(),std::plus<double>());
//      yf+=fp; //reset it

      if((i+1)%adaptevery==0 && mpirank==0) ambmf.adapt();
      if(modeltypef!=MODEL_PROBIT) {
#ifdef _OPENMPI
         if(mpirank==0) psbmf.draw(gen); else psbmf.draw_mpislave(gen);
#else
         psbmf.draw(gen);
#endif
         disigf = fpsbmf;
      }
      if((i+1)%adaptevery==0 && mpirank==0) psbmf.adapt();      
   }



   // Burn Iterations
   thetas_burn[0]=thetas_adapt[nadapt-1];
   for(size_t i=1;i<burn;i++) {

      // Draw BART model for eta(x,theta)     
      if((i % printevery) ==0 && mpirank==0) cout << "Burn iteration " << i << endl;
#ifdef _OPENMPI
      if(mpirank==0) ambm.draw(gen); else ambm.draw_mpislave(gen);
#else
      ambm.draw(gen);
#endif
      dipsc = dic;
      dipsc -= fambm;      
      if(modeltypec!=MODEL_PROBIT) {
#ifdef _OPENMPI
         if(mpirank==0) psbm.draw(gen); else psbm.draw_mpislave(gen);
#else
      psbm.draw(gen);
#endif
        disigc = fpsbm;
      }

      // Draw theta
#ifdef _OPENMPI 
      thetas_burn[i]=thetas_burn[i-1];
      updateall(xpred,thetas_burn[i],thetaidx,pthetas,p,nf);
     // size_t j=(size_t)std::floor(gen.uniform()*pthetas);
      // for(size_t j=0;j<pthetas;j++) {
         if(mpirank>0) {
            MPI_Status status;
            std::fill(fp.begin(),fp.end(),0.0);
            std::fill(sigfp.begin(),sigfp.end(),0.0);
            ambm.predict(&dip);   //eta(x,theta) -> g(x,theta)
            ambmf.predict(&dip2);
            psbmf.predict(&dip3);  //sigmaf(x)

            // (yf-eta(x,theta))/sigf
            std::transform(yf.begin(),yf.end(),fp.begin(),buf.begin(),std::minus<double>());
            std::transform(buf.begin(),buf.end(),fp2.begin(),buf.begin(),std::minus<double>());
            // std::transform(buf.begin(),buf.end(),sigfp.begin(),buf.begin(),std::divides<double>());
//            buf=yf-fp; buf/=sigfp;

            // criterion calculation for current theta
            ssold=0.0;
            // for(size_t k=0;k<nf;k++) ssold+=buf[k];
            // ssold=-1.0*ssold*ssold/4.0*(1.0/(nf/dip3.y[0]/dip3.y[0]+1.0/tauf/tauf));
            for(size_t k=0;k<nf;k++) ssold+=buf[k]*buf[k];//dip.y[k]*dip.y[k];

            for(size_t j=0;j<pthetas;j++) {
               if(th_type[j]) //Normal prior
                  thetas_burn[i][j]=thetas_burn[i-1][j]+propwidth[j]*gen.normal();
               else //Uniform prior
                  thetas_burn[i][j]=thetas_burn[i-1][j]+propwidth[j]*(gen.uniform()-0.5);
            }
            updateall(xpred,thetas_burn[i],thetaidx,pthetas,p,nf);

            std::fill(fp.begin(),fp.end(),0.0);
            ambm.predict(&dip);   //eta(x,thetaprime) -> g(x,thetaprime)

            // (yf-eta(x,theta))/sigf
            std::transform(yf.begin(),yf.end(),fp.begin(),buf.begin(),std::minus<double>());
            std::transform(buf.begin(),buf.end(),fp2.begin(),buf.begin(),std::minus<double>());
            // std::transform(buf.begin(),buf.end(),sigfp.begin(),buf.begin(),std::divides<double>());
//            buf=yf-fp; buf/=sigfp;

            ssnew=0.0;
            // for(size_t k=0;k<nf;k++) ssnew+=buf[k];
            // ssnew=-1.0*ssnew*ssnew/4.0*(1.0/(nf/dip3.y[0]/dip3.y[0]+1.0/tauf/tauf));
            for(size_t k=0;k<nf;k++) ssnew+=buf[k]*buf[k];//dip.y[k]*dip.y[k];

            // lalpha=(ssnew-ssold);
            lalpha=ssold-ssnew;
            MPI_Reduce(&ssold,NULL,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&ssnew,NULL,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Recv(NULL,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            if(status.MPI_TAG==MPI_TAG_CAL_ACCEPT) {
               for(size_t j=0;j<pthetas;j++) accept_thetas[j]++;
            }
            else {
               thetas_burn[i]=thetas_burn[i-1];
               updateall(xpred,thetas_burn[i],thetaidx,pthetas,p,nf);
               for(size_t j=0;j<pthetas;j++) reject_thetas[j]++;
            }
         }
         if(mpirank==0) {
            MPI_Request *request = new MPI_Request[tc];
               logprior=0.0;
               for(size_t j=0;j<pthetas;j++) {
                  if(th_type[j]) //Normal prior
                     thetas_burn[i][j]=thetas_burn[i-1][j]+propwidth[j]*gen.normal();
                  else //Uniform prior
                     thetas_burn[i][j]=thetas_burn[i-1][j]+propwidth[j]*(gen.uniform()-0.5);

                  if(th_type[j]) { //Normal prior
                     logprior += -1.0/2.0/th_prior_b[j]/th_prior_b[j]*(thetas_burn[i][j]-th_prior_a[j])*(thetas_burn[i][j]-th_prior_a[j])
                     +1.0/2.0/th_prior_b[j]/th_prior_b[j]*(thetas_burn[i-1][j]-th_prior_a[j])*(thetas_burn[i-1][j]-th_prior_a[j]);
                  }
                  else { //Uniform prior
                     logprior += 0.0;
                     if(thetas_burn[i][j]<th_prior_a[j] || thetas_burn[i][j]>th_prior_b[j])
                        logprior=-std::numeric_limits<double>::infinity();
                  }
               }
               lalpha=0.0; ssold=0.0; ssnew=0.0;
               MPI_Reduce(MPI_IN_PLACE,&ssold,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
               MPI_Reduce(MPI_IN_PLACE,&ssnew,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
               // lalpha=-1.0/2.0*lalpha+logprior;
               lalpha=(overallnuf+nf)/2.0*std::log(valformarg+0.5*ssold)
                        -(overallnuf+nf)/2.0*std::log(valformarg+0.5*ssnew);
               lalpha=lalpha+logprior;
               lalpha=std::min(0.0,lalpha);

               double alpha=gen.uniform();
               if(log(alpha)<lalpha) //accept
               {
                  for(size_t j=0;j<pthetas;j++) accept_thetas[j]++;
                  const int tag=MPI_TAG_CAL_ACCEPT;
                  for(size_t k=1; k<(size_t)tc; k++) {
                     MPI_Isend(NULL,0,MPI_PACKED,k,tag,MPI_COMM_WORLD,&request[k-1]);
                  }
               }
               else { //reject
                  thetas_burn[i]=thetas_burn[i-1];
                  const int tag=MPI_TAG_CAL_REJECT;
                  for(size_t k=1; k<(size_t)tc; k++) {
                     MPI_Isend(NULL,0,MPI_PACKED,k,tag,MPI_COMM_WORLD,&request[k-1]);
                  }
                  for(size_t j=0;j<pthetas;j++) reject_thetas[j]++;
               }
               MPI_Waitall(tc-1,request,MPI_STATUSES_IGNORE);
               delete[] request;
         }
      // }
      if((i+1)%burn==0) {
         if(mpirank==0){
            for(size_t j=0;j<pthetas;j++)
               if(mpirank==0) cout << "Acceptance rate theta" << j << "=" << ((double)accept_thetas[j])/((double)(accept_thetas[j]+reject_thetas[j])) << endl;
         }
         std::fill(accept_thetas.begin(),accept_thetas.end(),0);
         std::fill(reject_thetas.begin(),reject_thetas.end(),0);
      }

#else
     // openmp version of draw.
#endif

      // Draw BART model for delta(x)
      std::fill(fp.begin(),fp.end(),0.0);
      updateall(xpred,thetas_burn[i],thetaidx,pthetas,p,nf);
      ambm.predict(&dip); // predict eta(x,theta[i]) and store it in dip
      std::transform(yf.begin(),yf.end(),fp.begin(),yf.begin(),std::minus<double>());
//      yf-=fp;

#ifdef _OPENMPI
      if(mpirank==0) ambmf.draw(gen); else ambmf.draw_mpislave(gen);
#else
      ambmf.draw(gen);
#endif
      dipsf = dif;
      dipsf -= fambmf;
      std::transform(yf.begin(),yf.end(),fp.begin(),yf.begin(),std::plus<double>());
//      yf+=fp; //reset it

      if(modeltypef!=MODEL_PROBIT) {
#ifdef _OPENMPI
         if(mpirank==0) psbmf.draw(gen); else psbmf.draw_mpislave(gen);
#else
         psbmf.draw(gen);
#endif
         disigf = fpsbmf;
      }
   }

   if(summarystats) {
      ambm.setstats(true);
      psbm.setstats(true);
   }



   // Draw iterations
   //save tree to vec format
   if(mpirank==0) {
      ambm.savetree(0,mc,onn,oid,ovar,oc,otheta);
      psbm.savetree(0,mhc,snn,sid,svar,sc,stheta);
      ambmf.savetree(0,mf,onnf,oidf,ovarf,ocf,othetaf);
      psbmf.savetree(0,mhf,snnf,sidf,svarf,scf,sthetaf);
   }

   thetas[0]=thetas_burn[burn-1];
   for(size_t i=1;i<nd;i++) {

      // Draw BART model for eta(x,theta)     
      if((i % printevery) ==0 && mpirank==0) cout << "Draw iteration " << i << endl;
#ifdef _OPENMPI
      if(mpirank==0) ambm.draw(gen); else ambm.draw_mpislave(gen);
#else
      ambm.draw(gen);
#endif
      dipsc = dic;
      dipsc -= fambm;
      if(modeltypec!=MODEL_PROBIT) {
#ifdef _OPENMPI
         if(mpirank==0) psbm.draw(gen); else psbm.draw_mpislave(gen);
#else
      psbm.draw(gen);
#endif
         disigc = fpsbm;
      }

      // Draw theta
#ifdef _OPENMPI
      thetas[i]=thetas[i-1];
      updateall(xpred,thetas[i],thetaidx,pthetas,p,nf);
     // size_t j=(size_t)std::floor(gen.uniform()*pthetas);
     // for(size_t j=0;j<pthetas;j++) {
         if(mpirank>0) {
            MPI_Status status;
            std::fill(fp.begin(),fp.end(),0.0);
            std::fill(sigfp.begin(),sigfp.end(),0.0);
            ambm.predict(&dip);   //eta(x,theta) -> g(x,theta)
            ambmf.predict(&dip2);
            psbmf.predict(&dip3);  //sigmaf(x)

            // (yf-eta(x,theta))/sigf
            std::transform(yf.begin(),yf.end(),fp.begin(),buf.begin(),std::minus<double>());
            std::transform(buf.begin(),buf.end(),fp2.begin(),buf.begin(),std::minus<double>());
            // std::transform(buf.begin(),buf.end(),sigfp.begin(),buf.begin(),std::divides<double>());
//            buf=yf-fp; buf/=sigfp;

            // criterion calculation for current theta
            ssold=0.0;
            // for(size_t k=0;k<nf;k++) ssold+=buf[k];
            // ssold=-1.0*ssold*ssold/4.0*(1.0/(nf/dip3.y[0]/dip3.y[0]+1.0/tauf/tauf));
            for(size_t k=0;k<nf;k++) ssold+=buf[k]*buf[k];//dip.y[k]*dip.y[k];

            for(size_t j=0;j<pthetas;j++) {
               if(th_type[j]) //Normal prior
                  thetas[i][j]=thetas[i-1][j]+propwidth[j]*gen.normal();
               else //Uniform prior
                  thetas[i][j]=thetas[i-1][j]+propwidth[j]*(gen.uniform()-0.5);
            }
            updateall(xpred,thetas[i],thetaidx,pthetas,p,nf);

            std::fill(fp.begin(),fp.end(),0.0);
            ambm.predict(&dip);   //eta(x,thetaprime) -> g(x,thetaprime)

            // (yf-eta(x,theta))/sigf
            std::transform(yf.begin(),yf.end(),fp.begin(),buf.begin(),std::minus<double>());
            std::transform(buf.begin(),buf.end(),fp2.begin(),buf.begin(),std::minus<double>());
            // std::transform(buf.begin(),buf.end(),sigfp.begin(),buf.begin(),std::divides<double>());
//            buf=yf-fp; buf/=sigfp;

            // criterion calculation for proposed theta
            ssnew=0.0;
            // for(size_t k=0;k<nf;k++) ssnew+=buf[k];
            // ssnew=-1.0*ssnew*ssnew/4.0*(1.0/(nf/dip3.y[0]/dip3.y[0]+1.0/tauf/tauf));
            for(size_t k=0;k<nf;k++) ssnew+=buf[k]*buf[k];//dip.y[k]*dip.y[k];

            // lalpha=(ssnew-ssold);
            MPI_Reduce(&ssold,NULL,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&ssnew,NULL,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Recv(NULL,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            if(status.MPI_TAG==MPI_TAG_CAL_ACCEPT) {
               for(size_t j=0;j<pthetas;j++) accept_thetas[j]++;
            }
            else {
               thetas[i]=thetas[i-1];
               updateall(xpred,thetas[i],thetaidx,pthetas,p,nf);
               for(size_t j=0;j<pthetas;j++) reject_thetas[j]++;
            }
         }
         if(mpirank==0) {
            MPI_Request *request = new MPI_Request[tc];
            logprior=0.0;
            for(size_t j=0;j<pthetas;j++) {
               if(th_type[j]) //Normal prior
                  thetas[i][j]=thetas[i-1][j]+propwidth[j]*gen.normal();
               else //Uniform prior
                  thetas[i][j]=thetas[i-1][j]+propwidth[j]*(gen.uniform()-0.5);

               if(th_type[j]) { //Normal prior
                  logprior += -1.0/2.0/th_prior_b[j]/th_prior_b[j]*(thetas[i][j]-th_prior_a[j])*(thetas[i][j]-th_prior_a[j])
                  +1.0/2.0/th_prior_b[j]/th_prior_b[j]*(thetas[i-1][j]-th_prior_a[j])*(thetas[i-1][j]-th_prior_a[j]);
               }
               else { //Uniform prior
                  logprior += 0.0;
                  if(thetas[i][j]<th_prior_a[j] || thetas[i][j]>th_prior_b[j])
                     logprior=-std::numeric_limits<double>::infinity();
               }
            }
               lalpha=0.0; ssold=0.0; ssnew=0.0;
               MPI_Reduce(MPI_IN_PLACE,&ssold,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
               MPI_Reduce(MPI_IN_PLACE,&ssnew,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
               // lalpha=-1.0/2.0*lalpha+logprior;
               lalpha=(overallnuf+nf)/2.0*std::log(valformarg+0.5*ssold)
                        -(overallnuf+nf)/2.0*std::log(valformarg+0.5*ssnew);
               lalpha=lalpha+logprior;
               lalpha=std::min(0.0,lalpha);

               double alpha=gen.uniform();
               if(log(alpha)<lalpha) //accept
               {
                  for(size_t j=0;j<pthetas;j++) accept_thetas[j]++;
                  const int tag=MPI_TAG_CAL_ACCEPT;
                  for(size_t k=1; k<(size_t)tc; k++) {
                     MPI_Isend(NULL,0,MPI_PACKED,k,tag,MPI_COMM_WORLD,&request[k-1]);
                  }
               }
               else { //reject
                  thetas[i]=thetas[i-1];
                  const int tag=MPI_TAG_CAL_REJECT;
                  for(size_t k=1; k<(size_t)tc; k++) {
                     MPI_Isend(NULL,0,MPI_PACKED,k,tag,MPI_COMM_WORLD,&request[k-1]);
                  }
                  for(size_t j=0;j<pthetas;j++) reject_thetas[j]++;
               }
               MPI_Waitall(tc-1,request,MPI_STATUSES_IGNORE);
               delete[] request;
         }
      // }
      if((i+1)%nd==0) {
         if(mpirank==0){
            for(size_t j=0;j<pthetas;j++)
               if(mpirank==0) cout << "Acceptance rate theta" << j << "=" << ((double)accept_thetas[j])/((double)(accept_thetas[j]+reject_thetas[j])) << endl;
         }
         std::fill(accept_thetas.begin(),accept_thetas.end(),0);
         std::fill(reject_thetas.begin(),reject_thetas.end(),0);
      }

#else
     // openmp version of draw.
#endif

      // Draw BART model for delta(x)
      std::fill(fp.begin(),fp.end(),0.0);
      updateall(xpred,thetas[i],thetaidx,pthetas,p,nf);
      ambm.predict(&dip); // predict eta(x,theta[i]) and store it in dip
      std::transform(yf.begin(),yf.end(),fp.begin(),yf.begin(),std::minus<double>());
//      yf-=fp;

#ifdef _OPENMPI
      if(mpirank==0) ambmf.draw(gen); else ambmf.draw_mpislave(gen);
#else
      ambmf.draw(gen);
#endif
      dipsf = dif;
      dipsf -= fambmf;
      std::transform(yf.begin(),yf.end(),fp.begin(),yf.begin(),std::plus<double>());
//      yf+=fp; //reset it

      if(modeltypef!=MODEL_PROBIT) {
#ifdef _OPENMPI
         if(mpirank==0) psbmf.draw(gen); else psbmf.draw_mpislave(gen);
#else
         psbmf.draw(gen);
#endif
         disigf = fpsbmf;
      }

      //save tree to vec format
      if(mpirank==0) {
         ambm.savetree(i,mc,onn,oid,ovar,oc,otheta);
         psbm.savetree(i,mhc,snn,sid,svar,sc,stheta);
         ambmf.savetree(i,mf,onnf,oidf,ovarf,ocf,othetaf);
         psbmf.savetree(i,mhf,snnf,sidf,svarf,scf,sthetaf);
      }
   }
#ifdef _OPENMPI
   if(mpirank==0) {
      tend=MPI_Wtime();
      cout << "Training time was " << (tend-tstart)/60.0 << " minutes." << endl;
   }
#endif

   //Flatten posterior trees to a few (very long) vectors so we can just pass pointers
   //to these vectors back to R (which is much much faster than copying all the data back).
   if(mpirank==0) {
      cout << "Returning posterior, please wait...";

      // eta(x,theta) part of model
      std::vector<int>* e_ots=new std::vector<int>(nd*mc);
      std::vector<int>* e_oid=new std::vector<int>;
      std::vector<int>* e_ovar=new std::vector<int>;
      std::vector<int>* e_oc=new std::vector<int>;
      std::vector<double>* e_otheta=new std::vector<double>;
      std::vector<int>* e_sts=new std::vector<int>(nd*mhc);
      std::vector<int>* e_sid=new std::vector<int>;
      std::vector<int>* e_svar=new std::vector<int>;
      std::vector<int>* e_sc=new std::vector<int>;
      std::vector<double>* e_stheta=new std::vector<double>;
      for(size_t i=0;i<nd;i++)
         for(size_t j=0;j<mc;j++) {
            e_ots->at(i*mc+j)=static_cast<int>(oid[i*mc+j].size());
            e_oid->insert(e_oid->end(),oid[i*mc+j].begin(),oid[i*mc+j].end());
            e_ovar->insert(e_ovar->end(),ovar[i*mc+j].begin(),ovar[i*mc+j].end());
            e_oc->insert(e_oc->end(),oc[i*mc+j].begin(),oc[i*mc+j].end());
            e_otheta->insert(e_otheta->end(),otheta[i*mc+j].begin(),otheta[i*mc+j].end());
         }
      for(size_t i=0;i<nd;i++)
         for(size_t j=0;j<mhc;j++) {
            e_sts->at(i*mhc+j)=static_cast<int>(sid[i*mhc+j].size());
            e_sid->insert(e_sid->end(),sid[i*mhc+j].begin(),sid[i*mhc+j].end());
            e_svar->insert(e_svar->end(),svar[i*mhc+j].begin(),svar[i*mhc+j].end());
            e_sc->insert(e_sc->end(),sc[i*mhc+j].begin(),sc[i*mhc+j].end());
            e_stheta->insert(e_stheta->end(),stheta[i*mhc+j].begin(),stheta[i*mhc+j].end());
         }

      //write out to file
      std::ofstream omc(folder + modelname + ".eta.fit");
      omc << nd << endl;
      omc << mc << endl;
      omc << mhc << endl;
      omc << e_ots->size() << endl;
      for(size_t i=0;i<e_ots->size();i++) omc << e_ots->at(i) << endl;
      omc << e_oid->size() << endl;
      for(size_t i=0;i<e_oid->size();i++) omc << e_oid->at(i) << endl;
      omc << e_ovar->size() << endl;
      for(size_t i=0;i<e_ovar->size();i++) omc << e_ovar->at(i) << endl;
      omc << e_oc->size() << endl;
      for(size_t i=0;i<e_oc->size();i++) omc << e_oc->at(i) << endl;
      omc << e_otheta->size() << endl;
      for(size_t i=0;i<e_otheta->size();i++) omc << std::scientific << e_otheta->at(i) << endl;
      omc << e_sts->size() << endl;
      for(size_t i=0;i<e_sts->size();i++) omc << e_sts->at(i) << endl;
      omc << e_sid->size() << endl;
      for(size_t i=0;i<e_sid->size();i++) omc << e_sid->at(i) << endl;
      omc << e_svar->size() << endl;
      for(size_t i=0;i<e_svar->size();i++) omc << e_svar->at(i) << endl;
      omc << e_sc->size() << endl;
      for(size_t i=0;i<e_sc->size();i++) omc << e_sc->at(i) << endl;
      omc << e_stheta->size() << endl;
      for(size_t i=0;i<e_stheta->size();i++) omc << std::scientific << e_stheta->at(i) << endl;
      omc.close();

      // delta(x) part of model
      std::vector<int>* e_otsf=new std::vector<int>(nd*mf);
      std::vector<int>* e_oidf=new std::vector<int>;
      std::vector<int>* e_ovarf=new std::vector<int>;
      std::vector<int>* e_ocf=new std::vector<int>;
      std::vector<double>* e_othetaf=new std::vector<double>;
      std::vector<int>* e_stsf=new std::vector<int>(nd*mhf);
      std::vector<int>* e_sidf=new std::vector<int>;
      std::vector<int>* e_svarf=new std::vector<int>;
      std::vector<int>* e_scf=new std::vector<int>;
      std::vector<double>* e_sthetaf=new std::vector<double>;
      for(size_t i=0;i<nd;i++)
         for(size_t j=0;j<mf;j++) {
            e_otsf->at(i*mf+j)=static_cast<int>(oidf[i*mf+j].size());
            e_oidf->insert(e_oidf->end(),oidf[i*mf+j].begin(),oidf[i*mf+j].end());
            e_ovarf->insert(e_ovarf->end(),ovarf[i*mf+j].begin(),ovarf[i*mf+j].end());
            e_ocf->insert(e_ocf->end(),ocf[i*mf+j].begin(),ocf[i*mf+j].end());
            e_othetaf->insert(e_othetaf->end(),othetaf[i*mf+j].begin(),othetaf[i*mf+j].end());
         }
      for(size_t i=0;i<nd;i++)
         for(size_t j=0;j<mhf;j++) {
            e_stsf->at(i*mhf+j)=static_cast<int>(sidf[i*mhf+j].size());
            e_sidf->insert(e_sidf->end(),sidf[i*mhf+j].begin(),sidf[i*mhf+j].end());
            e_svarf->insert(e_svarf->end(),svarf[i*mhf+j].begin(),svarf[i*mhf+j].end());
            e_scf->insert(e_scf->end(),scf[i*mhf+j].begin(),scf[i*mhf+j].end());
            e_sthetaf->insert(e_sthetaf->end(),sthetaf[i*mhf+j].begin(),sthetaf[i*mhf+j].end());
         }

      //write out to file
      std::ofstream omf(folder + modelname + ".delta.fit");
      omf << nd << endl;
      omf << mf << endl;
      omf << mhf << endl;
      omf << e_otsf->size() << endl;
      for(size_t i=0;i<e_otsf->size();i++) omf << e_otsf->at(i) << endl;
      omf << e_oidf->size() << endl;
      for(size_t i=0;i<e_oidf->size();i++) omf << e_oidf->at(i) << endl;
      omf << e_ovarf->size() << endl;
      for(size_t i=0;i<e_ovarf->size();i++) omf << e_ovarf->at(i) << endl;
      omf << e_ocf->size() << endl;
      for(size_t i=0;i<e_ocf->size();i++) omf << e_ocf->at(i) << endl;
      omf << e_othetaf->size() << endl;
      for(size_t i=0;i<e_othetaf->size();i++) omf << std::scientific << e_othetaf->at(i) << endl;
      omf << e_stsf->size() << endl;
      for(size_t i=0;i<e_stsf->size();i++) omf << e_stsf->at(i) << endl;
      omf << e_sidf->size() << endl;
      for(size_t i=0;i<e_sidf->size();i++) omf << e_sidf->at(i) << endl;
      omf << e_svarf->size() << endl;
      for(size_t i=0;i<e_svarf->size();i++) omf << e_svarf->at(i) << endl;
      omf << e_scf->size() << endl;
      for(size_t i=0;i<e_scf->size();i++) omf << e_scf->at(i) << endl;
      omf << e_sthetaf->size() << endl;
      for(size_t i=0;i<e_sthetaf->size();i++) omf << std::scientific << e_sthetaf->at(i) << endl;
      omf.close();

      //write calibration parameters out to file
      std::ofstream omt(folder + modelname + ".eta.theta.fit");
      for(size_t i=0;i<nd;i++) {
         for(size_t j=0;j<pthetas;j++)
            omt << std::scientific << thetas[i][j] << " ";
         omt << endl;
      }

      cout << " done." << endl;
   }

/*
   // summary statistics
   if(summarystats) {
      cout << "Calculating summary statistics" << endl;
      unsigned int varcount[p];
      for(size_t i=0;i<p;i++) varcount[i]=0;
      unsigned int tmaxd=0;
      unsigned int tmind=0;
      double tavgd=0.0;

      ambm.getstats(&varcount[0],&tavgd,&tmaxd,&tmind);
      tavgd/=(double)(nd*m);
      cout << "Average tree depth (ambm): " << tavgd << endl;
      cout << "Maximum tree depth (ambm): " << tmaxd << endl;
      cout << "Minimum tree depth (ambm): " << tmind << endl;
      cout << "Vartivity summary (ambm)" << endl;
      for(size_t i=0;i<p;i++)
         cout << "Var " << i << ": " << varcount[i] << endl;

      for(size_t i=0;i<p;i++) varcount[i]=0;
      tmaxd=0; tmind=0; tavgd=0.0;
      psbm.getstats(&varcount[0],&tavgd,&tmaxd,&tmind);
      tavgd/=(double)(nd*mh);
      cout << "Average tree depth (psbm): " << tavgd << endl;
      cout << "Maximum tree depth (psbm): " << tmaxd << endl;
      cout << "Minimum tree depth (psbm): " << tmind << endl;
      cout << "Vartivity summary (psbm)" << endl;
      for(size_t i=0;i<p;i++)
         cout << "Var " << i << ": " << varcount[i] << endl;
   }
*/

   //-------------------------------------------------- 
   // Cleanup.
#ifdef _OPENMPI
   delete[] lwrf;
   delete[] uprf;
   delete[] lwrc;
   delete[] uprc;
   // delete[] fp;
   // delete[] fp2;
   // delete[] sigfp;
   // if(mpirank>0) delete[] rc;
   // if(mpirank>0) delete[] rf;
   MPI_Finalize();
#else
   delete[] rf;
   delete[] rc;
#endif

   return 0;
}
