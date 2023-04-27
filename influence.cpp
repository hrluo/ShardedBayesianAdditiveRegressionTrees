//     influence.cpp: Implement calculation of influential observations for OpenBT.
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


// Calculate hyperrectangles of influence for each observation given as
// well as the weight.
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
   std::ifstream conf(folder+"config.influence");
   std::string modelname;
   std::string xicore;

   //model name, xi
   conf >> modelname;
   conf >> xicore;

   //number of saved draws and number of trees
   size_t nd;
   size_t m;
   size_t mh;

   conf >> nd;
   conf >> m;
   conf >> mh;

   //number of predictors
   size_t p;
   conf >> p;

   //xd==influence input setting x to get local hyperrectangle
   std::vector<double> xd(p);
   for(size_t i=0;i<p;i++)
      conf >> xd[i];

   //min and max of predictors
   std::vector<double> minx(p);
   std::vector<double> maxx(p);
   for(size_t i=0;i<p;i++)
      conf >> minx[i];
   for(size_t i=0;i<p;i++)
      conf >> maxx[i];

   //thread count
   int tc;
   conf >> tc;

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
      cout << "OpenBT influence CLI" << endl;
      cout << "Loading config file at " << folder << endl;
   }

/*   //--------------------------------------------------
   //read in xp.
   std::vector<double> xp;
   double xtemp;
   size_t np;
//   std::string folder("."+modelname+"/");
   std::stringstream xfss;
   std::string xfs;
   xfss << folder << xpcore << mpirank;
   xfs=xfss.str();
   std::ifstream xf(xfs);
   while(xf >> xtemp)
      xp.push_back(xtemp);
   np = xp.size()/p;
#ifndef SILENT
   cout << "node " << mpirank << " loaded " << np << " inputs of dimension " << p << " from " << xfs << endl;
#endif
*/
   //--------------------------------------------------
   //make xinfo
   xinfo xi;
   xi.resize(p);

   for(size_t i=0;i<p;i++) {
      std::vector<double> xivec;
      double xitemp;

      std::stringstream xifss;
      std::string xifs;
      xifss << folder << xicore << (i+1);
      xifs=xifss.str();
      std::ifstream xif(xifs);
      while(xif >> xitemp)
         xivec.push_back(xitemp);
      xi[i]=xivec;
   }
#ifndef SILENT
   cout << "&&& made xinfo\n";
#endif

   //summarize input variables:
#ifndef SILENT
   if(mpirank==0)
      for(size_t i=0;i<p;i++)
      {
         cout << "Variable " << i << " has numcuts=" << xi[i].size() << " : ";
         cout << xi[i][0] << " ... " << xi[i][xi[i].size()-1] << endl;
      }
#endif



   // set up ambrt object
   ambrt ambm(m);
   ambm.setxi(&xi); //set the cutpoints for this model object

   //setup psbrt object
   psbrt psbm(mh);
   psbm.setxi(&xi); //set the cutpoints for this model object



   //load from file
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
   if(m!=im) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
   if(mh!=imh) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
#else
   if(nd!=ind) { cout << "Error loading posterior trees" << endl; return 0; }
   if(m!=im) { cout << "Error loading posterior trees" << endl; return 0; }
   if(mh!=imh) { cout << "Error loading posterior trees" << endl; return 0; }
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



   // Calculate range of posterior samples to calculate xd's influence rectangles for MPI.
   int startnd=0,endnd=nd-1;
   size_t snd,end,rnd=nd;
#ifdef _OPENMPI
   calcbegend(nd,mpirank,tc,&startnd,&endnd);
   snd=(size_t)startnd;
   end=(size_t)endnd;
   rnd=end-snd;
#ifndef SILENT
   cout << "Node " << mpirank << " calculating influence rectangles for posterior draws " << startnd << " to " << endnd-1 << " (range="<< rnd << ")" << endl;
#endif
#endif

   //objects where we'll store the realizations
   // std::vector<std::vector<double> > tedraw(nd,std::vector<double>(np));
   // std::vector<std::vector<double> > tedrawh(nd,std::vector<double>(np));
   // std::vector<std::vector<double> > tedrawp(nd,std::vector<double>(np));
   std::vector<std::vector<double> > aset(rnd, std::vector<double>());
   std::vector<std::vector<double> > bset(rnd, std::vector<double>());
   std::vector<std::vector<double> > auset(rnd, std::vector<double>());
   std::vector<std::vector<double> > buset(rnd, std::vector<double>());

   // double *fp = new double[np];
   // dinfo dip;
   // dip.x = &xp[0]; dip.y=fp; dip.p = p; dip.n=np; dip.tc=1;

   // Temporary vectors used for loading one model realization at a time.
   std::vector<int> onn(m,1);
   std::vector<std::vector<int> > oid(m, std::vector<int>(1));
   std::vector<std::vector<int> > ov(m, std::vector<int>(1));
   std::vector<std::vector<int> > oc(m, std::vector<int>(1));
   std::vector<std::vector<double> > otheta(m, std::vector<double>(1));
   std::vector<int> snn(mh,1);
   std::vector<std::vector<int> > sid(mh, std::vector<int>(1));
   std::vector<std::vector<int> > sv(mh, std::vector<int>(1));
   std::vector<std::vector<int> > sc(mh, std::vector<int>(1));
   std::vector<std::vector<double> > stheta(mh, std::vector<double>(1));

   // Draw realizations of the posterior predictive.
   size_t curdx=0;
   size_t cumdx=0;
#ifdef _OPENMPI
   double tstart=0.0,tend=0.0;
   if(mpirank==0) tstart=MPI_Wtime();
#endif


   // Mean trees first
   if(mpirank==0) cout << "Calculating local influence hyperrectangle" << endl;
   std::vector<double> asol,bsol;
   size_t ii=0;

   for(size_t i=0;i<nd;i++) {
      curdx=0;
      for(size_t j=0;j<m;j++) {
         onn[j]=e_ots.at(i*m+j);
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

      ambm.loadtree(0,m,onn,oid,ov,oc,otheta);
      // draw realization
      // ambm.predict(&dip);
      // for(size_t j=0;j<np;j++) tedraw[i][j] = fp[j] + fmean;

      // find hyperrectangle corresponding to xd
      if(i>=snd && i<end) {
         ambm.inflrect(xd.data(),asol,bsol,minx,maxx,p);
         aset[ii]=asol;
         bset[ii]=bsol;
         asol.clear();
         bsol.clear();
         ambm.influnionrect(xd.data(),asol,bsol,minx,maxx,p);
         auset[ii]=asol;
         buset[ii]=bsol;
         asol.clear();
         bsol.clear();
         ii++;
      }
   }


/*   // Variance trees second
   if(mpirank==0) cout << "Drawing sd response from posterior predictive" << endl;
   cumdx=0;
   curdx=0;
   for(size_t i=0;i<nd;i++) {
      curdx=0;
      for(size_t j=0;j<mh;j++) {
         snn[j]=e_sts.at(i*mh+j);
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

      psbm.loadtree(0,mh,snn,sid,sv,sc,stheta);
      // draw realization
      psbm.predict(&dip);
      for(size_t j=0;j<np;j++) tedrawh[i][j] = fp[j];
   }

   // For probit models we'll also construct probabilities
   if(modeltype==MODEL_PROBIT || modeltype==MODEL_MODIFIEDPROBIT) {
      if(mpirank==0) cout << "Drawing posterior predictive probabilities" << endl;
      for(size_t i=0;i<nd;i++)
         for(size_t j=0;j<np;j++)
            tedrawp[i][j]=normal_01_cdf(tedraw[i][j]/tedrawh[i][j]);
   }
*/

#ifdef _OPENMPI
   if(mpirank==0) {
      tend=MPI_Wtime();
      cout << "Posterior influential hyperrectangle draw time was " << (tend-tstart)/60.0 << " minutes." << endl;
   }
#endif

   // Save the draws.
   if(mpirank==0) cout << "Saving influential hyperrectangle draws...";

   std::ofstream omf(folder + modelname + ".influence" + std::to_string(mpirank));
   for(size_t i=0;i<rnd;i++) {
      for(size_t j=0;j<p;j++)
            omf << std::scientific << aset[i][j] << " ";
      for(size_t j=0;j<p;j++)
            omf << std::scientific << bset[i][j] << " ";
      for(size_t j=0;j<p;j++)
            omf << std::scientific << auset[i][j] << " ";
      for(size_t j=0;j<p;j++)
            omf << std::scientific << buset[i][j] << " ";
      omf << endl;
   }
   omf.close();

   if(mpirank==0) cout << " done." << endl;

   //-------------------------------------------------- 
   // Cleanup.
#ifdef _OPENMPI
   MPI_Finalize();
#endif

   return 0;
}

