//     cli.cpp: Implement command-line model interface to OpenBT.
//     Copyright (C) 2012-2019 Matthew T. Pratola
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
   std::ifstream conf(folder+"config");

   // model type
   int modeltype;
   conf >> modeltype;

   // core filenames for x,y input
   std::string xcore,ycore;
   conf >> xcore;
   conf >> ycore;

   //offset -- used in probit, but not in bart for instance.
   double off;
   conf >> off;

   //number of trees
   size_t m;
   size_t mh;
   conf >> m;
   conf >> mh;

   //nd and burn
   size_t nd;
   size_t burn;
   size_t nadapt;
   size_t adaptevery;
   conf >> nd;
   conf >> burn;
   conf >> nadapt;
   conf >> adaptevery;

   //mu prior (tau, ambrt) and sigma prior (lambda,nu, psbrt)
   double tau;
   double overalllambda;
   double overallnu;
   conf >> tau;
   conf >> overalllambda;
   conf >> overallnu;

   //tree prior
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

   //sigma vector
   std::string score;
   conf >> score;

   //change variable
   std::string chgvcore;
   conf >> chgvcore;

   //control
   double pbd;
   double pb;
   double pbdh;
   double pbh;
   double stepwpert;
   double stepwperth;
   double probchv;
   double probchvh;
   size_t minnumbot;
   size_t minnumboth;
   size_t printevery;
   int tempshardepth;
   size_t shardepth=0;
   bool sharding=false;
   bool randshard=false;
   double shardpsplit;
   std::string xicore;
   std::string modelname;
   conf >> pbd;
   conf >> pb;
   conf >> pbdh;
   conf >> pbh;
   conf >> stepwpert;
   conf >> stepwperth;
   conf >> probchv;
   conf >> probchvh;
   conf >> minnumbot;
   conf >> minnumboth;
   conf >> printevery;
   conf >> xicore;
   conf >> tempshardepth;
   conf >> shardpsplit;
   conf >> randshard;
   conf >> modelname;

   if(tempshardepth>-1) {
      sharding=true;
      shardepth=(size_t)tempshardepth;
   }

   bool dopert=true;
   bool doperth=true;
   if(probchv<0) dopert=false;
   if(probchvh<0) doperth=false;


   //summary statistics yes/no
   bool summarystats;
   conf >> summarystats;
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
      cout << "OpenBT command-line interface (cli)" << endl;
      cout << "Loading config file at " << folder << endl;
      if(!sharding)
         cout << "Sharding disabled" << endl;
      else
         cout << "Sharded model enabled (sharding variable 0, depth " << shardepth << ")" << endl;
 
      if(randshard)
         cout << "Randomized Sharding Enabled" << endl;
      else
         cout << "Randomized Sharding Disabled" << endl;
   }

   //--------------------------------------------------
   //read in y 
   std::vector<double> y;
   double ytemp;
   size_t n=0;
#ifdef _OPENMPI
   if(mpirank>0) { //only load data on slaves
#endif
      std::stringstream yfss;
      std::string yfs;
      yfss << folder << ycore << mpirank;
      yfs=yfss.str();
      std::ifstream yf(yfs);
      while(yf >> ytemp)
         y.push_back(ytemp);
      n=y.size();
#ifndef SILENT
      cout << "node " << mpirank << " loaded " << n << " from " << yfs <<endl;
#endif
#ifdef _OPENMPI
   }
#endif

   //--------------------------------------------------
   //Initialize latent variable z.  Only used in probit, for example.
   std::vector<double> z;
   if(modeltype==MODEL_PROBIT || modeltype==MODEL_MODIFIEDPROBIT)
#ifdef _OPENMPI
      if(mpirank>0) { //only need latent z's on slaves
#endif
         for(size_t i=0;i<n;i++) {
            if(y[i]==1.0)
               z.push_back(2.0);//gen_right_trunc_normal(off,1.0,0.0,gen);//1.0; //std::max(gen.normal(),0.0+off);
            else 
               z.push_back(-2.0);//gen_left_trunc_normal(off,1.0,0.0,gen);//-1.0; //std::min(gen.normal(),0.0-off);
         }
#ifdef _OPENMPI
      }
#endif

   //--------------------------------------------------
   //Initialize vector of truncated observations.  Only used in merck_truncated model.
   std::vector<size_t> truncs;
   std::vector<double> truncvals;
   size_t trunctemp;
   size_t ntruncs=0;
   if(modeltype==MODEL_MERCK_TRUNCATED) {
#ifdef _OPENMPI
      if(mpirank>0) { //only need indices on slaves.
         std::stringstream trfss;
         std::string tfs;
         trfss << folder << "truncs" << mpirank;
         tfs=trfss.str();
         std::ifstream tf(tfs);
         while(tf >> trunctemp)
            truncs.push_back(trunctemp);
         ntruncs=truncs.size();

         truncvals.resize(ntruncs);
         for(size_t j=0;j<ntruncs;j++)
            truncvals[j]=y[truncs[j]];
#ifndef SILENT
         cout << "node " << mpirank << " loaded " << ntruncs << " from " << tfs << endl;
         if(ntruncs>0)
            cout << "node " << mpirank << " first trunc value is " << truncvals[0] << endl;
#endif
      }
#endif
   }

   //--------------------------------------------------
   //read in x 
   std::vector<double> x;
   double xtemp;
   size_t p=0;
#ifdef _OPENMPI
   if(mpirank>0) {
#endif
      std::stringstream xfss;
      std::string xfs;
      xfss << folder << xcore << mpirank;
      xfs=xfss.str();
      std::ifstream xf(xfs);
      while(xf >> xtemp)
         x.push_back(xtemp);
      p = x.size()/n;
#ifndef SILENT
      cout << "node " << mpirank << " loaded " << n << " inputs of dimension " << p << " from " << xfs << endl;
#endif
#ifdef _OPENMPI
   }
   int tempp = (unsigned int) p;
   MPI_Allreduce(MPI_IN_PLACE,&tempp,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
   if(mpirank>0 && p != ((size_t) tempp)) { cout << "PROBLEM LOADING DATA" << endl; MPI_Finalize(); return 0;}
   p=(size_t)tempp;
#endif


   //--------------------------------------------------
   //dinfo
   dinfo di;
   di.n=0;di.p=p,di.x = NULL;di.y=NULL;di.tc=tc;
#ifdef _OPENMPI
   if(mpirank>0) { 
#endif
      di.n=n; di.x = &x[0]; di.y = &y[0]; 
      if(modeltype==MODEL_PROBIT || modeltype==MODEL_MODIFIEDPROBIT)
         di.y = &z[0];
#ifdef _OPENMPI
   }
#endif

   //--------------------------------------------------
   //read in sigmav  -- same as above.
   std::vector<double> sigmav;
   double stemp;
   size_t nsig=0;
#ifdef _OPENMPI
   if(mpirank>0) { //only load data on slaves
#endif
      std::stringstream sfss;
      std::string sfs;
      sfss << folder << score << mpirank;
      sfs=sfss.str();
      std::ifstream sf(sfs);
      while(sf >> stemp)
         sigmav.push_back(stemp);
      nsig=sigmav.size();
#ifndef SILENT
      cout << "node " << mpirank << " loaded " << nsig << " from " << sfs <<endl;
#endif
#ifdef _OPENMPI
      if(n!=nsig) { cout << "PROBLEM LOADING SIGMAV" << endl; MPI_Finalize(); return 0; }
   }
#else
   if(n!=nsig) { cout << "PROBLEM LOADING SIGMAV" << endl; return 0; }
#endif

   double *sig=&sigmav[0];
   dinfo disig;
   disig.n=0; disig.p=p; disig.x=NULL; disig.y=NULL; disig.tc=tc;
#ifdef _OPENMPI
   if(mpirank>0) { 
#endif
      disig.n=n; disig.x=&x[0]; disig.y=sig; 
#ifdef _OPENMPI
   }
#endif

   //--------------------------------------------------
   // read in the initial change of variable rank correlation matrix
   std::vector<std::vector<double>> chgv;
   std::vector<double> cvvtemp;
   double cvtemp;
   std::ifstream chgvf(folder + chgvcore);
   for(size_t i=0;i<di.p;i++) {
      cvvtemp.clear();
      for(size_t j=0;j<di.p;j++) {
         chgvf >> cvtemp;
         cvvtemp.push_back(cvtemp);
      }
      chgv.push_back(cvvtemp);
   }
#ifndef SILENT
   cout << "mpirank=" << mpirank << ": change of variable rank correlation matrix loaded:" << endl;
#endif
   // if(mpirank==0) //print it out:
   //    for(size_t i=0;i<di.p;i++) {
   //       for(size_t j=0;j<di.p;j++)
   //          cout << "(" << i << "," << j << ")" << chgv[i][j] << "  ";
   //       cout << endl;
   //    }


   //--------------------------------------------------
   // decide what variables each slave node will update in change-of-variable proposals.
#ifdef _OPENMPI
   int* lwr=new int[tc];
   int* upr=new int[tc];
   lwr[0]=-1; upr[0]=-1;
   for(size_t i=1;i<(size_t)tc;i++) { 
      lwr[i]=-1; upr[i]=-1; 
      calcbegend(p,i-1,tc-1,&lwr[i],&upr[i]);
      if(p>1 && lwr[i]==0 && upr[i]==0) { lwr[i]=-1; upr[i]=-1; }
   }

#ifndef SILENT
   if(mpirank>0) cout << "Slave node " << mpirank << " will update variables " << lwr[mpirank] << " to " << upr[mpirank]-1 << endl;
#endif
#endif

   //--------------------------------------------------
   //print args
#ifndef SILENT
   cout << "**********************\n";
   cout << "n: " << n << endl;
   cout << "p: " << p << endl;
   if(mpirank>0) cout << "first row: " << x[0] << ", " << x[p-1] << endl;
   if(mpirank>0) cout << "second row: " << x[p] << ", " << x[2*p-1] << endl;
   if(mpirank>0) cout << "last row: " << x[(n-1)*p] << ", " << x[n*p-1] << endl;
   if(mpirank>0) cout << "first and last y: " << y[0] << ", " << y[n-1] << endl;
   cout << "number of trees mean: " << m << endl;
   cout << "number of trees stan dev: " << mh << endl;
   cout << "tau: " << tau << endl;
   cout << "overalllambda: " << overalllambda << endl;
   cout << "overallnu: " << overallnu << endl;
   cout << "burn (nskip): " << burn << endl;
   cout << "nd (ndpost): " << nd << endl;
   cout << "nadapt: " << nadapt << endl;
   cout << "adaptevery: " << adaptevery << endl;
   cout << "mean tree prior base: " << alpha << endl;
   cout << "mean tree prior power: " << mybeta << endl;
   cout << "variance tree prior base: " << alphah << endl;
   cout << "variance tree prior power: " << mybetah << endl;
   cout << "thread count: " << tc << endl;
   if(mpirank>0) cout << "first and last sigmav: " << sigmav[0] << ", " << sigmav[n-1] << endl;
   cout << "chgv first row: " << chgv[0][0] << ", " << chgv[0][p-1] << endl;
   cout << "chgv last row: " << chgv[p-1][0] << ", " << chgv[p-1][p-1] << endl;
   cout << "mean trees prob birth/death: " << pbd << endl;
   cout << "mean trees prob birth: " << pb << endl;
   cout << "variance trees prob birth/death: " << pbdh << endl;
   cout << "variance trees prob birth: " << pbh << endl;
   cout << "mean trees initial step width pert move: " << stepwpert << endl;
   cout << "variance trees initial step width pert move: " << stepwperth << endl;
   cout << "mean trees prob of a change var move : " << probchv << endl;
   cout << "variance trees prob of a change var move : " << probchvh << endl;
   cout << "mean trees min num obs in bottom node: " << minnumbot << endl;
   cout << "variance trees min num obs in bottom node: " << minnumboth << endl;
   cout << "*****printevery: " << printevery << endl;
#endif

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
   for(size_t i=0;i<p;i++)
   {
      cout << "Variable " << i << " has numcuts=" << xi[i].size() << " : ";
      cout << xi[i][0] << " ... " << xi[i][xi[i].size()-1] << endl;
   }
#endif

   // set up ambrt object
   ambrt ambm(m);

   //cutpoints
   ambm.setxi(&xi);    //set the cutpoints for this model object
   //data objects
   ambm.setdata(&di);  //set the data
   //thread count
   ambm.settc(tc-1);      //set the number of slaves when using MPI.
   //mpi rank
#ifdef _OPENMPI
   ambm.setmpirank(mpirank);  //set the rank when using MPI.
   ambm.setmpicvrange(lwr,upr); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
   //tree prior
   ambm.settp(alpha, //the alpha parameter in the tree depth penalty prior
         mybeta,     //the beta parameter in the tree depth penalty prior
         sharding,
         shardepth,
         shardpsplit,
         randshard
         );
   //MCMC info
   ambm.setmi(
         pbd,  //probability of birth/death
         pb,  //probability of birth
         minnumbot,    //minimum number of observations in a bottom node
         dopert, //do perturb/change variable proposal?
         stepwpert,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
         probchv,  //probability of doing a change of variable proposal.  perturb prob=1-this.
         &chgv  //initialize the change of variable correlation matrix.
         );
   ambm.setci(tau,sig);


   //--------------------------------------------------
   //setup psbrt object
   psbrt psbm(mh,overalllambda);

   //make di for psbrt object
   dinfo dips;
   dips.n=0; dips.p=p; dips.x=NULL; dips.y=NULL; dips.tc=tc;
   double *r=NULL;
#ifdef _OPENMPI
   if(mpirank>0) {
#endif
      r = new double[n];
      for(size_t i=0;i<n;i++) r[i]=sigmav[i];
      dips.x=&x[0]; dips.y=r; dips.n=n;
#ifdef _OPENMPI
   }
#endif

   double opm=1.0/((double)mh);
   double nu=2.0*pow(overallnu,opm)/(pow(overallnu,opm)-pow(overallnu-2.0,opm));
   double lambda=pow(overalllambda,opm);

   //cutpoints
   psbm.setxi(&xi);    //set the cutpoints for this model object
   //data objects
   psbm.setdata(&dips);  //set the data
   //thread count
   psbm.settc(tc-1); 
   //mpi rank
#ifdef _OPENMPI
   psbm.setmpirank(mpirank);  //set the rank when using MPI.
   psbm.setmpicvrange(lwr,upr); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
   //tree prior
   psbm.settp(alphah, //the alpha parameter in the tree depth penalty prior
         mybetah,     //the beta parameter in the tree depth penalty prior
         sharding,
         shardepth,
         shardpsplit,
         randshard
         );
   psbm.setmi(
         pbdh,  //probability of birth/death
         pbh,  //probability of birth
         minnumboth,    //minimum number of observations in a bottom node
         doperth, //do perturb/change variable proposal?
         stepwperth,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
         probchvh,  //probability of doing a change of variable proposal.  perturb prob=1-this.
         &chgv  //initialize the change of variable correlation matrix.
         );
   psbm.setci(nu,lambda);



   //--------------------------------------------------
   //run mcmc
   std::vector<int> onn(nd*m,1);
   std::vector<std::vector<int> > oid(nd*m, std::vector<int>(1));
   std::vector<std::vector<int> > ovar(nd*m, std::vector<int>(1));
   std::vector<std::vector<int> > oc(nd*m, std::vector<int>(1));
   std::vector<std::vector<double> > otheta(nd*m, std::vector<double>(1));
   std::vector<int> snn(nd*mh,1);
   std::vector<std::vector<int> > sid(nd*mh, std::vector<int>(1));
   std::vector<std::vector<int> > svar(nd*mh, std::vector<int>(1));
   std::vector<std::vector<int> > sc(nd*mh, std::vector<int>(1));
   std::vector<std::vector<double> > stheta(nd*mh, std::vector<double>(1));
   brtMethodWrapper fambm(&brt::f,ambm);
   brtMethodWrapper fpsbm(&brt::f,psbm);
   std::vector<double> Cookd_mean(di.n,0.0);
   std::vector<double> Cookd_mean_temp(di.n,0.0);
   std::vector<double> Cookd_max(di.n,0.0);
   std::vector<double> Cookd_max_temp(di.n,0.0);
   std::vector<double> KLinfl(di.n,0.0);
   std::vector<double> KLinfl_temp(di.n,0.0);


#ifdef _OPENMPI
   double tstart=0.0,tend=0.0;
   if(mpirank==0) tstart=MPI_Wtime();
   if(mpirank==0) cout << "Starting MCMC..." << endl;
#else
   cout << "Starting MCMC..." << endl;
#endif

   for(size_t i=0;i<nadapt;i++) { 
      if((i % printevery) ==0 && mpirank==0) cout << "Adapt iteration " << i << endl;
#ifdef _OPENMPI
      if(mpirank==0) ambm.draw(gen); else ambm.draw_mpislave(gen);
#else
      ambm.draw(gen);
#endif
      dips = di;
      dips -= fambm;
      if((i+1)%adaptevery==0 && mpirank==0) ambm.adapt();
      if(modeltype!=MODEL_PROBIT) {
#ifdef _OPENMPI
         if(mpirank==0) psbm.draw(gen); else psbm.draw_mpislave(gen);
#else
         psbm.draw(gen);
#endif
         disig = fpsbm;
      }
      if((i+1)%adaptevery==0 && mpirank==0) psbm.adapt();      
      if(modeltype==MODEL_PROBIT || modeltype==MODEL_MODIFIEDPROBIT) {
         for(size_t j=0;j<n;j++) {
            if(y[j]==1.0)
               z[j]=std::max(gen.normal()*fpsbm.callMethod(j)+fambm.callMethod(j)+off,0.0);
            else
               z[j]=std::min(gen.normal()*fpsbm.callMethod(j)+fambm.callMethod(j)+off,0.0);
         }
      }
      if(modeltype==MODEL_MERCK_TRUNCATED) {
         for(size_t j=0;j<ntruncs;j++) {
            double u=gen.uniform();
            double pv=normal_01_cdf((truncvals[j]-fambm.callMethod(truncs[j]))/fpsbm.callMethod(truncs[j]));
            di.y[truncs[j]]=normal_01_cdf_inv(u*pv)*fpsbm.callMethod(truncs[j])+fambm.callMethod(truncs[j]);
         }
      }
   }
   for(size_t i=0;i<burn;i++) {
      if((i % printevery) ==0 && mpirank==0) cout << "Burn iteration " << i << endl;
#ifdef _OPENMPI
      if(mpirank==0) ambm.draw(gen); else ambm.draw_mpislave(gen);
#else
      ambm.draw(gen);
#endif
      dips = di;
      dips -= fambm;      
      if(modeltype!=MODEL_PROBIT) {
#ifdef _OPENMPI
         if(mpirank==0) psbm.draw(gen); else psbm.draw_mpislave(gen);
#else
      psbm.draw(gen);
#endif
        disig = fpsbm;
      }
      if(modeltype==MODEL_PROBIT || modeltype==MODEL_MODIFIEDPROBIT) {
         for(size_t j=0;j<n;j++) {
            if(y[j]==1.0)
               z[j]=std::max(gen.normal()*fpsbm.callMethod(j)+fambm.callMethod(j)+off,0.0);
            else
               z[j]=std::min(gen.normal()*fpsbm.callMethod(j)+fambm.callMethod(j)+off,0.0);
         }
      }
      if(modeltype==MODEL_MERCK_TRUNCATED) {
         for(size_t j=0;j<ntruncs;j++) {
            double u=gen.uniform();
            double pv=normal_01_cdf((truncvals[j]-fambm.callMethod(truncs[j]))/fpsbm.callMethod(truncs[j]));
            di.y[truncs[j]]=normal_01_cdf_inv(u*pv)*fpsbm.callMethod(truncs[j])+fambm.callMethod(truncs[j]);
         }
      }
   }
   if(summarystats) {
      ambm.setstats(true);
      psbm.setstats(true);
   }
   for(size_t i=0;i<nd;i++) {
      if((i % printevery) ==0 && mpirank==0) cout << "Draw iteration " << i << endl;
#ifdef _OPENMPI
      if(mpirank==0) ambm.draw(gen); else ambm.draw_mpislave(gen);
#else
      ambm.draw(gen);
#endif
      dips = di;
      dips -= fambm;
      if(modeltype!=MODEL_PROBIT) {
#ifdef _OPENMPI
         if(mpirank==0) psbm.draw(gen); else psbm.draw_mpislave(gen);
#else
      psbm.draw(gen);
#endif
         disig = fpsbm;
      }
      if(modeltype==MODEL_PROBIT || modeltype==MODEL_MODIFIEDPROBIT) {
         for(size_t j=0;j<n;j++) {
            if(y[j]==1.0)
               z[j]=std::max(gen.normal()*fpsbm.callMethod(j)+fambm.callMethod(j)+off,0.0);
            else
               z[j]=std::min(gen.normal()*fpsbm.callMethod(j)+fambm.callMethod(j)+off,0.0);
         }
      }
      if(modeltype==MODEL_MERCK_TRUNCATED) {
         for(size_t j=0;j<ntruncs;j++) {
            double u=gen.uniform();
            double pv=normal_01_cdf((truncvals[j]-fambm.callMethod(truncs[j]))/fpsbm.callMethod(truncs[j]));
            di.y[truncs[j]]=normal_01_cdf_inv(u*pv)*fpsbm.callMethod(truncs[j])+fambm.callMethod(truncs[j]);
         }
      }

      //Calculate Cook's distance and KL divergence influence metric
      ambm.cookdinfl(Cookd_mean_temp,Cookd_max_temp,disig.y);
      ambm.kldivinfl(KLinfl_temp,disig.y);
      for(size_t j=0;j<di.n;j++) {
         Cookd_mean[j] += Cookd_mean_temp[j]/((double) nd);
         Cookd_max[j] += Cookd_max_temp[j]/((double) nd);

         if(KLinfl_temp[j] == std::numeric_limits<double>::infinity()) {
            KLinfl[j]=std::numeric_limits<double>::infinity();
         }
         else if(KLinfl[j] != std::numeric_limits<double>::infinity()) {
            KLinfl[j] += KLinfl_temp[j]/((double) nd);
         }
      }


      //save tree to vec format
      if(mpirank==0) {
         ambm.savetree(i,m,onn,oid,ovar,oc,otheta);
         psbm.savetree(i,mh,snn,sid,svar,sc,stheta);
      }
   }
#ifdef _OPENMPI
   if(mpirank==0) {
      tend=MPI_Wtime();
      cout << "Training time was " << (tend-tstart)/60.0 << " minutes." << endl;
   }
#endif

   // Finalize KLinfl and write out to files.
   if(mpirank>0) {
      for(size_t j=0;j<di.n;j++) KLinfl[j] = std::abs(KLinfl[j]);

      std::ofstream omf(folder + modelname + ".influence" + std::to_string(mpirank));
      for(size_t j=0;j<di.n;j++) 
         omf << std::scientific << Cookd_mean[j] << " " << std::scientific << Cookd_max[j] << " " << std::scientific << KLinfl[j] << endl;
      omf.close();
   }

   //Flatten posterior trees to a few (very long) vectors so we can just pass pointers
   //to these vectors back to R (which is much much faster than copying all the data back).
   if(mpirank==0) {
      cout << "Returning posterior, please wait...";
      std::vector<int>* e_ots=new std::vector<int>(nd*m);
      std::vector<int>* e_oid=new std::vector<int>;
      std::vector<int>* e_ovar=new std::vector<int>;
      std::vector<int>* e_oc=new std::vector<int>;
      std::vector<double>* e_otheta=new std::vector<double>;
      std::vector<int>* e_sts=new std::vector<int>(nd*mh);
      std::vector<int>* e_sid=new std::vector<int>;
      std::vector<int>* e_svar=new std::vector<int>;
      std::vector<int>* e_sc=new std::vector<int>;
      std::vector<double>* e_stheta=new std::vector<double>;
      for(size_t i=0;i<nd;i++)
         for(size_t j=0;j<m;j++) {
            e_ots->at(i*m+j)=static_cast<int>(oid[i*m+j].size());
            e_oid->insert(e_oid->end(),oid[i*m+j].begin(),oid[i*m+j].end());
            e_ovar->insert(e_ovar->end(),ovar[i*m+j].begin(),ovar[i*m+j].end());
            e_oc->insert(e_oc->end(),oc[i*m+j].begin(),oc[i*m+j].end());
            e_otheta->insert(e_otheta->end(),otheta[i*m+j].begin(),otheta[i*m+j].end());
         }
      for(size_t i=0;i<nd;i++)
         for(size_t j=0;j<mh;j++) {
            e_sts->at(i*mh+j)=static_cast<int>(sid[i*mh+j].size());
            e_sid->insert(e_sid->end(),sid[i*mh+j].begin(),sid[i*mh+j].end());
            e_svar->insert(e_svar->end(),svar[i*mh+j].begin(),svar[i*mh+j].end());
            e_sc->insert(e_sc->end(),sc[i*mh+j].begin(),sc[i*mh+j].end());
            e_stheta->insert(e_stheta->end(),stheta[i*mh+j].begin(),stheta[i*mh+j].end());
         }

   //write out to file
      std::ofstream omf(folder + modelname + ".fit");
      omf << nd << endl;
      omf << m << endl;
      omf << mh << endl;
      omf << e_ots->size() << endl;
      for(size_t i=0;i<e_ots->size();i++) omf << e_ots->at(i) << endl;
      omf << e_oid->size() << endl;
      for(size_t i=0;i<e_oid->size();i++) omf << e_oid->at(i) << endl;
      omf << e_ovar->size() << endl;
      for(size_t i=0;i<e_ovar->size();i++) omf << e_ovar->at(i) << endl;
      omf << e_oc->size() << endl;
      for(size_t i=0;i<e_oc->size();i++) omf << e_oc->at(i) << endl;
      omf << e_otheta->size() << endl;
      for(size_t i=0;i<e_otheta->size();i++) omf << std::scientific << e_otheta->at(i) << endl;
      omf << e_sts->size() << endl;
      for(size_t i=0;i<e_sts->size();i++) omf << e_sts->at(i) << endl;
      omf << e_sid->size() << endl;
      for(size_t i=0;i<e_sid->size();i++) omf << e_sid->at(i) << endl;
      omf << e_svar->size() << endl;
      for(size_t i=0;i<e_svar->size();i++) omf << e_svar->at(i) << endl;
      omf << e_sc->size() << endl;
      for(size_t i=0;i<e_sc->size();i++) omf << e_sc->at(i) << endl;
      omf << e_stheta->size() << endl;
      for(size_t i=0;i<e_stheta->size();i++) omf << std::scientific << e_stheta->at(i) << endl;
      omf.close();

      cout << " done." << endl;
   }


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


   //-------------------------------------------------- 
   // Cleanup.
#ifdef _OPENMPI
   delete[] lwr;
   delete[] upr;
   if(mpirank>0) delete[] r;
   MPI_Finalize();
#else
   delete[] r;
#endif

   return 0;
}
