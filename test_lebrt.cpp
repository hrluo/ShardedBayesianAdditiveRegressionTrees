//     test.cpp: Lebesgue tree BT model class testing/validation code.
//     Copyright (C) 2012-2016 Matthew T. Pratola
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
#include <fstream>

#include "crn.h"
#include "brt.h"
#include "brtfuns.h"
#include "dinfo.h"
#include "lebrt.h"

#ifdef _OPENMPI
#   include <mpi.h>
#endif

using std::cout;
using std::endl;

int main()
{

   cout << "\n*****into test for lebrt\n";
   cout << "\n\n";

   int mpirank=0;
   int tc=0;

#ifdef _OPENMPI
   MPI_Init(NULL,NULL);
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Comm_rank(MPI_COMM_WORLD,&mpirank);
   MPI_Comm_size(MPI_COMM_WORLD,&tc);
#ifndef SILENT
   cout << "\nMPI: node " << mpirank << " of " << tc << " processes." << endl;
#endif
   if(tc<=1) return 0; //need at least 2 processes!
#endif


   // Set dimension of u's for lebrt object.
   size_t p=1;  // lebrt objects only support 1-dimensional covariate
   size_t n=100; //100 fake data on each slave node
   crn gen;
   gen.set_seed(199);


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

   if(mpirank>0) cout << "Slave node " << mpirank << " will update variables " << lwr[mpirank] << " to " << upr[mpirank]-1 << endl;
#endif


   //--------------------------------------------------
   //make xinfo
   xinfo xi;
// if fixed shard depth
// a shard depth of 4 with a balanced tree suggests 2^4=16 shards and 2^4=15 cutpoints, giving
// on average 500/16=31 obs per shard.
   size_t nc=15;
// if not fixed shard depth set it to something largish and let the algorithm determine
// For eg, nc=100 with 500 observations suggests 500/100=5 observations per shard on average.
//   size_t nc=100;
   double xinc = (1.0-0.0)/(nc+1.0);
   xi.resize(p);

   for(size_t i=0;i<p;i++) {
      xi[i].resize(nc);
      for(size_t j=0;j<nc;j++) xi[i][j] = 0.0 + (j+1)*xinc;
   }

   cout << endl << "cutpoints initialized (on [0,1]):" << endl;
   for(size_t i=0;i<nc;i++)
      cout << xi[0][i] << "|";
   cout << endl << endl;;



   //--------------------------------------------------
   // read in the initial change of variable rank correlation matrix
   std::vector<std::vector<double> > chgv;

   chgv.resize(1);
   chgv[0].resize(1);
   chgv[0][0]=1.0;

   cout << "change of variable rank correlation matrix loaded:" << endl;
   for(size_t i=0;i<p;i++) {
      for(size_t j=0;j<p;j++)
         cout << "(" << i << "," << j << ")" << chgv[i][j] << "  ";
      cout << endl;
   }



   //--------------------------------------------------
   //dinfo
   dinfo di;
   di.n=0;di.p=p,di.x = NULL;di.y=NULL;di.tc=tc;
#ifdef _OPENMPI
   if(mpirank>0) { 
#endif
      std::vector<double> y(n);
      for(size_t i=0;i<n;i++)
         y[i]=1.0;

      std::vector<double> x(n);
      // for(size_t i=0;i<n;i++)
      //    x[i]=gen.uniform();

      di.n=n; di.x = &x[0]; di.y = &y[0]; 
#ifdef _OPENMPI
   }
#endif

#ifdef _OPENMPI
if(mpirank>0) {
      cout << "1>di.n=" << di.n << endl;
      cout << "1>di.x=" << di.x[0] << ", " << *(di.x+1) << ", " << *(di.x+2) << "..." << endl;
      cout << "1>di.y=" << di.y[0] << ", " << *(di.y+1) << ", " << *(di.y+2) << "..." << endl;
}
#endif




   //--------------------------------------------------
   // run simpler tests just on the root node
#ifdef _OPENMPI
if(mpirank==0) {
#endif

   tree t;
   t.pr();


   //--------------------------------------------------
   // check lesinfo objects

   cout  << "##################################################\n";
   cout << "*****Checking out lesinfo objects !!!!\n";

   lesinfo lei;
   cout << "lei:\n";
   cout << lei.n << ", " << lei.w << endl;
   lei.n=10;lei.w=0.5;
   lesinfo lei2(lei);
   cout << "lei2:\n";
   cout << lei2.n << ", " << lei2.w << endl;

   lesinfo lei3(lei2);
   lei3 += lei2;
   cout << "lei3:\n";
   cout << lei3.n << ", " << lei3.w << endl;

   lei3=lei2;
   cout << "lei3 (after =lei2):\n";
   cout << lei3.n << ", " << lei3.w << endl;

   // lesinfo lei4 = lei3+lei2;
   lesinfo lei4;
   lei4=lei3+lei2;
   cout << "lei4:\n";
   cout << lei4.n << ", " << lei4.w << endl;


#ifdef _OPENMPI
}
#endif


#ifdef _OPENMPI
if(mpirank>0) {
      cout << "2>di.n=" << di.n << endl;
      cout << "2>di.x=" << *(di.x) << ", " << *(di.x+1) << ", " << *(di.x+2) << "..." << endl;
      cout << "2>di.y=" << *(di.y) << ", " << *(di.y+1) << ", " << *(di.y+2) << "..." << endl;
}
#endif

   //--------------------------------------------------
   // try instantiating a lebrt
   lebrt lebm;

   cout  << "##################################################\n";
   cout << "*****Print out first lebrt object !!!!\n";
//   lebm.pr();



   //cutpoints
   lebm.setxi(&xi);    //set the cutpoints for this model object
   //data objects
   lebm.setdata(&di);  //set the data
   lebm.reshard(gen);

// #ifdef _OPENMPI
//    if(mpirank>0) { 
//       for(size_t i=0;i<n;i++)
//          di.y[i]=1.0;

//       for(size_t i=0;i<n;i++)
//          di.x[i]=gen.uniform();
//    }
// #endif


if(mpirank>0) {
      cout << "3>di.n=" << di.n << endl;
      cout << "3>di.x=" << *(di.x) << ", " << *(di.x+1) << ", " << *(di.x+2) << "..." << endl;
      cout << "3>di.y=" << *(di.y) << ", " << *(di.y+1) << ", " << *(di.y+2) << "..." << endl;
}


   //thread count
   lebm.settc(tc-1);      //set the number of slaves when using MPI, etc.
#ifdef _OPENMPI
   lebm.setmpirank(mpirank);  //set the rank when using MPI.
//   lebm.setmpicvrange(lwr,upr); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
   //tree prior
   lebm.settp(1.0, //the alpha parameter in the tree depth penalty prior
         0.0,     //the beta parameter in the tree depth penalty prior
         false,
         0,
         0.0,
         false
         );
   //MCMC info
   lebm.setmi(
         0.5,  //probability of birth/death
         0.5,  //probability of birth
         1,    //minimum number of observations in a bottom node
         true, //do perturb/change variable proposal?
         0.01,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
         0.01,  //probability of doing a change of variable proposal.  perturb prob=1-this.
         &chgv  //initialize the change of variable correlation matrix.
         );
   lebm.setci();

   cout << "\n*****after init:\n";
   lebm.pr();

   //    cout << "\n*****after 1 draw:\n";
   #ifdef _OPENMPI
         if(mpirank==0) lebm.draw(gen); else lebm.draw_mpislave(gen);
   #else
         lebm.draw(gen);
   #endif

   if(mpirank==0) lebm.pr();




   cout << "\n*****after 10 more draws:\n";

if(mpirank>0) {
      cout << "4>di.n=" << di.n << endl;
      cout << "4>di.x=" << *(di.x) << ", " << *(di.x+1) << ", " << *(di.x+2) << "..." << endl;
      cout << "4>di.y=" << *(di.y) << ", " << *(di.y+1) << ", " << *(di.y+2) << "..." << endl;
}

   #ifdef _OPENMPI
         for(size_t i=0;i<10;i++)
            if(mpirank==0) lebm.draw(gen); else lebm.draw_mpislave(gen);
   #else
         for(size_t i=0;i<10;i++)
            lebm.draw(gen);
   #endif

   if(mpirank==0) lebm.pr();



   #ifdef _OPENMPI
      MPI_Finalize();
   #endif

   return 0;
}
