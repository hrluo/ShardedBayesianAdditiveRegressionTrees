//     emulate.cpp: Implement emulation interface for OpenBT (BART-based) computer model calibration.
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


#include <iostream>
#include <string>
#include <ctime>
#include <sstream>

#include <fstream>
#include <vector>
#include <limits>

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



void updatethetavec(std::vector<double>& xe, std::vector<double>& thetavec, std::vector<size_t>& thetaidx, size_t pthetas, size_t p, size_t n)
{
   size_t index;
   for(size_t i=0;i<n;i++)
      for(size_t j=0;j<pthetas;j++)
      {
         index=i*p+thetaidx[j];
         xe[index]=thetavec[j];
      }

   return;
}


// Draw predictive realizations at the prediciton points, xp.
int main(int argc, char* argv[])
{
   std::string folder("");

   if(argc>1)
   {
      //argument on the command line is path to config file.
      folder=std::string(argv[1]);
      folder=folder+"/";
   }


   //--------------------------------------------------
   //process args
   std::ifstream conf(folder+"config.emulate");

   //model name
   std::string modelname;
   conf >> modelname;

   // model type
   int modeltypec,modeltypef;
   conf >> modeltypec;
   conf >> modeltypef;

   //cut info for eta(x,theta)
   std::string xicorec;
   conf >> xicorec;

   // core filenames for eta(x,theta) inputs
   std::string xecore;
   conf >> xecore;

   //number of saved draws and number of trees
   size_t nd;
   size_t mc;
   size_t mhc;

   conf >> nd;
   conf >> mc;
   conf >> mhc;

   //number of predictors
   size_t p;
   size_t pdelta;
   size_t pthetas;
   conf >> p;
   conf >> pdelta;
   conf >> pthetas;
 
    //index of calibration parameters for eta(x,theta)
   size_t thtemp;
   std::vector<size_t> thetaidx(pthetas);
   for(size_t i=0;i<pthetas;i++) {
      conf >> thtemp;
      thetaidx[i]=thtemp;
   }

   //thread count
   int tc;
   conf >> tc;

   //mean offset
   double fmeanc;
   conf >> fmeanc;

   // done loading configuration
   conf.close();




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
   if(tc<=1){
      cout << "Error: tc=" << tc << endl;
      MPI_Finalize();
      return 0; //need at least 2 processes! 
   } 
   if(tc!=mpitc) {
      cout << "Error: tc does not match mpitc" << endl;
      MPI_Finalize();
      return 0; //mismatch between how MPI was started and how the data is prepared according to tc.
   }
// #else
//    if(tc!=1) return 0; //serial mode should have no slave threads!
#endif


   //--------------------------------------------------
   // Banner
   if(mpirank==0) {
      cout << endl;
      cout << "-----------------------------------" << endl;
      cout << "OpenBT emulation CLI" << endl;
      cout << "Loading config file at " << folder << endl;
   }


   //--------------------------------------------------
   //read in xe.
   std::vector<double> xe;
   double xtemp;
   size_t n;

   std::stringstream xfsse;
   std::string xfse;
   xfsse << folder << xecore << mpirank;
   xfse=xfsse.str();
   std::ifstream xfe(xfse);
   while(xfe >> xtemp)
      xe.push_back(xtemp);
   n = xe.size()/p;
#ifndef SILENT
   cout << "node " << mpirank << " loaded " << n << " inputs of dimension " << p << " from " << xfse << endl;
#endif



   //--------------------------------------------------
   //make xinfos
   xinfo xic;
   xic.resize(p);

   for(size_t i=0;i<p;i++) {
      std::vector<double> xicvec;
      double xictemp;

      std::stringstream xicfss;
      std::string xicfs;
      xicfss << folder << xicorec << (i+1);
      xicfs=xicfss.str();
      std::ifstream xistream(xicfs);
      while(xistream >> xictemp)
         xicvec.push_back(xictemp);
      xic[i]=xicvec;
   }
#ifndef SILENT
   cout << "&&& made xinfo\n";
#endif


  //summarize input variables:
#ifndef SILENT
   if(mpirank==0)
      for(size_t i=0;i<p;i++)
      {
         cout << "Variable " << i << " has numcuts=" << xic[i].size() << " : ";
         cout << xic[i][0] << " ... " << xic[i][xic[i].size()-1] << endl;
      }
#endif


   // set up ambrt object
   ambrt ambm(mc);
   ambm.setxi(&xic); //set the cutpoints for this model object

   //setup psbrt object
   psbrt psbm(mhc);
   psbm.setxi(&xic); //set the cutpoints for this model object



   //load eta(x,theta) tree posterior from file
#ifndef SILENT
   if(mpirank==0) cout << "Loading saved posterior tree draws" << endl;
#endif
   size_t ind,im,imh;
   std::ifstream imf(folder + modelname + ".fit");
   imf >> ind;
   imf >> im;
   imf >> imh;
#ifdef _OPENMPI
   if(nd!=ind) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
   if(mc!=im) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
   if(mhc!=imh) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
#else
   if(nd!=ind) { cout << "Error loading posterior trees" << endl; return 0; }
   if(mc!=im) { cout << "Error loading posterior trees" << endl; return 0; }
   if(mhc!=imh) { cout << "Error loading posterior trees" << endl; return 0; }
#endif

   size_t temp=0;
   imf >> temp;
   std::vector<int> e_ots(temp);
   for(size_t i=0;i<temp;i++) imf >> e_ots.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_oid(temp);
   for(size_t i=0;i<temp;i++) imf >> e_oid.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_ovar(temp);
   for(size_t i=0;i<temp;i++) imf >> e_ovar.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_oc(temp);
   for(size_t i=0;i<temp;i++) imf >> e_oc.at(i);

   temp=0;
   imf >> temp;
   std::vector<double> e_otheta(temp);
   for(size_t i=0;i<temp;i++) imf >> std::scientific >> e_otheta.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_sts(temp);
   for(size_t i=0;i<temp;i++) imf >> e_sts.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_sid(temp);
   for(size_t i=0;i<temp;i++) imf >> e_sid.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_svar(temp);
   for(size_t i=0;i<temp;i++) imf >> e_svar.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e_sc(temp);
   for(size_t i=0;i<temp;i++) imf >> e_sc.at(i);

   temp=0;
   imf >> temp;
   std::vector<double> e_stheta(temp);
   for(size_t i=0;i<temp;i++) imf >> std::scientific >> e_stheta.at(i);

   imf.close();


   // load theta posterior from file
   double thetatemp;
   std::vector<std::vector<double> > thetas(nd,std::vector<double>(pthetas));
cout << "Loading thetas from " << folder << modelname << ".theta.fit" << endl;
   std::ifstream imt(folder + modelname + ".theta.fit");
   for(size_t i=0;i<nd;i++) {
      for(size_t j=0;j<pthetas;j++) {
         imt >> thetatemp;
         thetas[i][j]=thetatemp;
      }
   }


   //objects where we'll store the realizations
   std::vector<std::vector<double> > tedraw(nd,std::vector<double>(n));
   std::vector<std::vector<double> > tedrawh(nd,std::vector<double>(n));
   // std::vector<std::vector<double> > tedrawp(nd,std::vector<double>(n));
   double *fp = new double[n];
   dinfo dip;
   dip.x = &xe[0]; dip.y=fp; dip.p = p; dip.n=n; dip.tc=1;

   // Temporary vectors used for loading one model realization at a time.
   std::vector<int> onn(mc,1);
   std::vector<std::vector<int> > oid(mc, std::vector<int>(1));
   std::vector<std::vector<int> > ov(mc, std::vector<int>(1));
   std::vector<std::vector<int> > oc(mc, std::vector<int>(1));
   std::vector<std::vector<double> > otheta(mc, std::vector<double>(1));
   std::vector<int> snn(mhc,1);
   std::vector<std::vector<int> > sid(mhc, std::vector<int>(1));
   std::vector<std::vector<int> > sv(mhc, std::vector<int>(1));
   std::vector<std::vector<int> > sc(mhc, std::vector<int>(1));
   std::vector<std::vector<double> > stheta(mhc, std::vector<double>(1));

   // Draw realizations of the posterior predictive.
   size_t curdx=0;
   size_t cumdx=0;
#ifdef _OPENMPI
   double tstart=0.0,tend=0.0;
   if(mpirank==0) tstart=MPI_Wtime();
#endif


   // Mean trees first
   if(mpirank==0) cout << "Drawing mean response from posterior predictive" << endl;
   for(size_t i=0;i<nd;i++) {
      curdx=0;
      for(size_t j=0;j<mc;j++) {
         onn[j]=e_ots.at(i*mc+j);
         oid[j].resize(onn[j]);
         ov[j].resize(onn[j]);
         oc[j].resize(onn[j]);
         otheta[j].resize(onn[j]);
         for(size_t k=0;k< (size_t)onn[j];k++) {
            oid[j][k]=e_oid.at(cumdx+curdx+k);
            ov[j][k]=e_ovar.at(cumdx+curdx+k);
            oc[j][k]=e_oc.at(cumdx+curdx+k);
            otheta[j][k]=e_otheta.at(cumdx+curdx+k);
         }
         curdx+=(size_t)onn[j];
      }
      cumdx+=curdx;

      ambm.loadtree(0,mc,onn,oid,ov,oc,otheta);

      // load this iteration's theta draws from saved posterior
      updatethetavec(xe,thetas[i],thetaidx,pthetas,p,n);

      // draw realization
      ambm.predict(&dip);
      for(size_t j=0;j<n;j++) tedraw[i][j] = fp[j] + fmeanc;
   }


   // Variance trees second
   if(mpirank==0) cout << "Drawing sd response from posterior predictive" << endl;
   cumdx=0;
   curdx=0;
   for(size_t i=0;i<nd;i++) {
      curdx=0;
      for(size_t j=0;j<mhc;j++) {
         snn[j]=e_sts.at(i*mhc+j);
         sid[j].resize(snn[j]);
         sv[j].resize(snn[j]);
         sc[j].resize(snn[j]);
         stheta[j].resize(snn[j]);
         for(size_t k=0;k< (size_t)snn[j];k++) {
            sid[j][k]=e_sid.at(cumdx+curdx+k);
            sv[j][k]=e_svar.at(cumdx+curdx+k);
            sc[j][k]=e_sc.at(cumdx+curdx+k);
            stheta[j][k]=e_stheta.at(cumdx+curdx+k);
         }
         curdx+=(size_t)snn[j];
      }
      cumdx+=curdx;

      psbm.loadtree(0,mhc,snn,sid,sv,sc,stheta);

      // load this iteration's theta draws from saved posterior
      updatethetavec(xe,thetas[i],thetaidx,pthetas,p,n);

      // draw realization
      psbm.predict(&dip);
      for(size_t j=0;j<n;j++) tedrawh[i][j] = fp[j];
   }

   // // For probit models we'll also construct probabilities
   // if(modeltype==MODEL_PROBIT || modeltype==MODEL_MODIFIEDPROBIT) {
   //    if(mpirank==0) cout << "Drawing posterior predictive probabilities" << endl;
   //    for(size_t i=0;i<nd;i++)
   //       for(size_t j=0;j<np;j++)
   //          tedrawp[i][j]=normal_01_cdf(tedraw[i][j]/tedrawh[i][j]);
   // }

#ifdef _OPENMPI
   if(mpirank==0) {
      tend=MPI_Wtime();
      cout << "Posterior predictive draw time was " << (tend-tstart)/60.0 << " minutes." << endl;
   }
#endif

   // Save the draws.
   if(mpirank==0) cout << "Saving posterior predictive draws...";
   std::ofstream omf(folder + modelname + ".mdraws" + std::to_string(mpirank));
   for(size_t i=0;i<nd;i++) {
      for(size_t j=0;j<n;j++)
         omf << std::scientific << tedraw[i][j] << " ";
      omf << endl;
   }
   omf.close();

   std::ofstream smf(folder + modelname + ".sdraws" + std::to_string(mpirank));
   for(size_t i=0;i<nd;i++) {
      for(size_t j=0;j<n;j++)
         smf << std::scientific << tedrawh[i][j] << " ";
      smf << endl;
   }
   smf.close();

   // if(modeltype==MODEL_PROBIT || modeltype==MODEL_MODIFIEDPROBIT) {
   //    std::ofstream pmf(folder + modelname + ".pdraws" + std::to_string(mpirank));
   //    for(size_t i=0;i<nd;i++) {
   //       for(size_t j=0;j<np;j++)
   //          pmf << std::scientific << tedrawp[i][j] << " ";
   //       pmf << endl;
   //    }
   //    pmf.close();
   // }

   if(mpirank==0) cout << " done." << endl;

   //-------------------------------------------------- 
   // Cleanup.
#ifdef _OPENMPI
   MPI_Finalize();
#endif

   return 0;
}

