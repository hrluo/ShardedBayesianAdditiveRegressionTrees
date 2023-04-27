//     mopareto.cpp: Implement Pareto-front multiobjective optimization using OpenBT.
//     Copyright (C) 2020 Matthew T. Pratola, Akira Horiguchi
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
//     Akira Horiguchi: horiguchi.6@osu.edu


#include <iostream>
#include <string>
#include <ctime>
#include <sstream>
#include <algorithm>  // std::set_union, std::set_intersection, std::sort
#include <unordered_map>  

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

using std::cout;
using std::endl;



// Calculate Pareto Front and Pareto Set given 2 trained BART models.
int main(int argc, char* argv[])
{
   std::string folder("");
   std::string folder2("");
   std::string folder3("");

   if(argc>1)
   {
      //argument on the command line is path to config file.
      folder=std::string(argv[1]);
      folder=folder+"/";
   }

   //--------------------------------------------------
   //process args
   std::ifstream conf(folder+"config.mopareto");

   //model name, number of saved draws and number of trees
   std::string modelname;
   std::string modelname2;
   std::string modelname3;
   std::string xicore;

   //model name and xi
   conf >> modelname;
   conf >> modelname2;
   conf >> modelname3;
   conf >> xicore;

   //location of the second fitted model
   conf >> folder2;
   folder2=folder2+"/";
   conf >> folder3;
   folder3=folder3+"/";

   //number of saved draws and number of trees
   size_t nd;
   size_t m1;
   size_t mh1;
   size_t m2;
   size_t mh2;
   size_t m3;
   size_t mh3;

   conf >> nd;
   conf >> m1;
   conf >> mh1;
   conf >> m2;
   conf >> mh2;
   conf >> m3;
   conf >> mh3;

   //number of predictors
   size_t p;
   conf >> p;

   //min and max of predictors
   std::vector<double> minx(p);
   std::vector<double> maxx(p);
   for(size_t i=0;i<p;i++)
      conf >> minx[i];
   for(size_t i=0;i<p;i++)
      conf >> maxx[i];

   //global means of each response
   double fmean1, fmean2, fmean3;
   conf >> fmean1;
   conf >> fmean2;
   conf >> fmean3;

   //thread count
   int tc;
   conf >> tc;
   conf.close();

   //simple flag to tell us if we are doing 2-output or 3-output Pareto front
   bool threeresponse=false;
   if(m3>0) threeresponse=true;
   size_t d=2;  // number of responses
   if (threeresponse) d=3;

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
      cout << "OpenBT Multiobjective Optimization using Pareto Front/Set CLI" << endl;
      cout << "Loading config file at " << folder << endl;
      cout << "Loading config file at " << folder2 << endl;
   }


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



   // set up ambrt objects
   ambrt ambm1(m1);
   ambm1.setxi(&xi); //set the cutpoints for this model object
   ambrt ambm2(m2);
   ambm2.setxi(&xi); //set the cutpoints for this model object
   ambrt ambm3(m3);
   if(threeresponse) ambm3.setxi(&xi);

   //setup psbrt objects
   psbrt psbm1(mh1);
   psbm1.setxi(&xi); //set the cutpoints for this model object
   psbrt psbm2(mh1);
   psbm2.setxi(&xi); //set the cutpoints for this model object
   psbrt psbm3(mh3);
   if(threeresponse) psbm3.setxi(&xi);


   //load first fitted model from file
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
   if(m1!=im) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
   if(mh1!=imh) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
#else
   if(nd!=ind) { cout << "Error loading posterior trees" << endl; return 0; }
   if(m1!=im) { cout << "Error loading posterior trees" << endl; return 0; }
   if(mh1!=imh) { cout << "Error loading posterior trees" << endl; return 0; }
#endif

   size_t temp=0;
   imf >> temp;
   std::vector<int> e1_ots(temp);
   for(size_t i=0;i<temp;i++) imf >> e1_ots.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e1_oid(temp);
   for(size_t i=0;i<temp;i++) imf >> e1_oid.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e1_ovar(temp);
   for(size_t i=0;i<temp;i++) imf >> e1_ovar.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e1_oc(temp);
   for(size_t i=0;i<temp;i++) imf >> e1_oc.at(i);

   temp=0;
   imf >> temp;
   std::vector<double> e1_otheta(temp);
   for(size_t i=0;i<temp;i++) imf >> std::scientific >> e1_otheta.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e1_sts(temp);
   for(size_t i=0;i<temp;i++) imf >> e1_sts.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e1_sid(temp);
   for(size_t i=0;i<temp;i++) imf >> e1_sid.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e1_svar(temp);
   for(size_t i=0;i<temp;i++) imf >> e1_svar.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e1_sc(temp);
   for(size_t i=0;i<temp;i++) imf >> e1_sc.at(i);

   temp=0;
   imf >> temp;
   std::vector<double> e1_stheta(temp);
   for(size_t i=0;i<temp;i++) imf >> std::scientific >> e1_stheta.at(i);

   imf.close();




   //load second fitted model from file
#ifndef SILENT
   if(mpirank==0) cout << "Loading saved posterior tree draws" << endl;
#endif
   imf.open(folder2 + modelname2 + ".fit");
   imf >> ind;
   imf >> im;
   imf >> imh;
#ifdef _OPENMPI
   if(nd!=ind) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
   if(m2!=im) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
   if(mh2!=imh) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
#else
   if(nd!=ind) { cout << "Error loading posterior trees" << endl; return 0; }
   if(m2!=im) { cout << "Error loading posterior trees" << endl; return 0; }
   if(mh2!=imh) { cout << "Error loading posterior trees" << endl; return 0; }
#endif

   temp=0;
   imf >> temp;
   std::vector<int> e2_ots(temp);
   for(size_t i=0;i<temp;i++) imf >> e2_ots.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e2_oid(temp);
   for(size_t i=0;i<temp;i++) imf >> e2_oid.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e2_ovar(temp);
   for(size_t i=0;i<temp;i++) imf >> e2_ovar.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e2_oc(temp);
   for(size_t i=0;i<temp;i++) imf >> e2_oc.at(i);

   temp=0;
   imf >> temp;
   std::vector<double> e2_otheta(temp);
   for(size_t i=0;i<temp;i++) imf >> std::scientific >> e2_otheta.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e2_sts(temp);
   for(size_t i=0;i<temp;i++) imf >> e2_sts.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e2_sid(temp);
   for(size_t i=0;i<temp;i++) imf >> e2_sid.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e2_svar(temp);
   for(size_t i=0;i<temp;i++) imf >> e2_svar.at(i);

   temp=0;
   imf >> temp;
   std::vector<int> e2_sc(temp);
   for(size_t i=0;i<temp;i++) imf >> e2_sc.at(i);

   temp=0;
   imf >> temp;
   std::vector<double> e2_stheta(temp);
   for(size_t i=0;i<temp;i++) imf >> std::scientific >> e2_stheta.at(i);

   imf.close();




   //load third fitted model from file
   if(threeresponse) {
#ifndef SILENT
      if(mpirank==0) cout << "Loading saved posterior tree draws" << endl;
#endif
      imf.open(folder3 + modelname3 + ".fit");
      imf >> ind;
      imf >> im;
      imf >> imh;
#ifdef _OPENMPI
      if(nd!=ind) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
      if(m3!=im) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
      if(mh3!=imh) { cout << "Error loading posterior trees" << endl; MPI_Finalize(); return 0; }
#else
      if(nd!=ind) { cout << "Error loading posterior trees" << endl; return 0; }
      if(m3!=im) { cout << "Error loading posterior trees" << endl; return 0; }
      if(mh3!=imh) { cout << "Error loading posterior trees" << endl; return 0; }
#endif
   }

   temp=0;
   if(threeresponse) imf >> temp;
   std::vector<int> e3_ots(temp);
   if(threeresponse) for(size_t i=0;i<temp;i++) imf >> e3_ots.at(i);

   temp=0;
   if(threeresponse) imf >> temp;
   std::vector<int> e3_oid(temp);
   if(threeresponse) for(size_t i=0;i<temp;i++) imf >> e3_oid.at(i);

   temp=0;
   if(threeresponse) imf >> temp;
   std::vector<int> e3_ovar(temp);
   if(threeresponse) for(size_t i=0;i<temp;i++) imf >> e3_ovar.at(i);

   temp=0;
   if(threeresponse) imf >> temp;
   std::vector<int> e3_oc(temp);
   if(threeresponse) for(size_t i=0;i<temp;i++) imf >> e3_oc.at(i);

   temp=0;
   if(threeresponse) imf >> temp;
   std::vector<double> e3_otheta(temp);
   if(threeresponse) for(size_t i=0;i<temp;i++) imf >> std::scientific >> e3_otheta.at(i);

   temp=0;
   if(threeresponse) imf >> temp;
   std::vector<int> e3_sts(temp);
   if(threeresponse) for(size_t i=0;i<temp;i++) imf >> e3_sts.at(i);

   temp=0;
   if(threeresponse) imf >> temp;
   std::vector<int> e3_sid(temp);
   if(threeresponse) for(size_t i=0;i<temp;i++) imf >> e3_sid.at(i);

   temp=0;
   if(threeresponse) imf >> temp;
   std::vector<int> e3_svar(temp);
   if(threeresponse) for(size_t i=0;i<temp;i++) imf >> e3_svar.at(i);

   temp=0;
   if(threeresponse) imf >> temp;
   std::vector<int> e3_sc(temp);
   if(threeresponse) for(size_t i=0;i<temp;i++) imf >> e3_sc.at(i);

   temp=0;
   if(threeresponse) imf >> temp;
   std::vector<double> e3_stheta(temp);
   if(threeresponse) for(size_t i=0;i<temp;i++) imf >> std::scientific >> e3_stheta.at(i);

   if(threeresponse) imf.close();




   // Calculate range of posterior samples to do Pareto front/set on for MPI.
   int startnd=0,endnd=nd-1;
   size_t snd,end,rnd=nd;
#ifdef _OPENMPI
   calcbegend(nd,mpirank,tc,&startnd,&endnd);
   snd=(size_t)startnd;
   end=(size_t)endnd;
   rnd=end-snd;
#ifndef SILENT
   cout << "Node " << mpirank << " calculating Pareto front and set for posterior draws " << startnd << " to " << endnd-1 << " (range="<< rnd << ")" << endl;
#endif
#endif

   //objects where we'll store the realizations
   std::vector<std::vector<double> > asol;
   std::vector<std::vector<double> > bsol;
   // std::vector<double> thetasol;
   std::list<std::vector<double> > thetasol;

   // Temporary vectors used for loading one model realization at a time.
   std::vector<int> onn1(m1,1);
   std::vector<std::vector<int> > oid1(m1, std::vector<int>(1));
   std::vector<std::vector<int> > ov1(m1, std::vector<int>(1));
   std::vector<std::vector<int> > oc1(m1, std::vector<int>(1));
   std::vector<std::vector<double> > otheta1(m1, std::vector<double>(1));
   std::vector<int> snn1(mh1,1);
   std::vector<std::vector<int> > sid1(mh1, std::vector<int>(1));
   std::vector<std::vector<int> > sv1(mh1, std::vector<int>(1));
   std::vector<std::vector<int> > sc1(mh1, std::vector<int>(1));
   std::vector<std::vector<double> > stheta1(mh1, std::vector<double>(1));

   std::vector<int> onn2(m2,1);
   std::vector<std::vector<int> > oid2(m2, std::vector<int>(1));
   std::vector<std::vector<int> > ov2(m2, std::vector<int>(1));
   std::vector<std::vector<int> > oc2(m2, std::vector<int>(1));
   std::vector<std::vector<double> > otheta2(m2, std::vector<double>(1));
   std::vector<int> snn2(mh2,1);
   std::vector<std::vector<int> > sid2(mh2, std::vector<int>(1));
   std::vector<std::vector<int> > sv2(mh2, std::vector<int>(1));
   std::vector<std::vector<int> > sc2(mh2, std::vector<int>(1));
   std::vector<std::vector<double> > stheta2(mh2, std::vector<double>(1));

   std::vector<int> onn3(m3,1);
   std::vector<std::vector<int> > oid3(m3, std::vector<int>(1));
   std::vector<std::vector<int> > ov3(m3, std::vector<int>(1));
   std::vector<std::vector<int> > oc3(m3, std::vector<int>(1));
   std::vector<std::vector<double> > otheta3(m3, std::vector<double>(1));
   std::vector<int> snn3(mh3,1);
   std::vector<std::vector<int> > sid3(mh3, std::vector<int>(1));
   std::vector<std::vector<int> > sv3(mh3, std::vector<int>(1));
   std::vector<std::vector<int> > sc3(mh3, std::vector<int>(1));
   std::vector<std::vector<double> > stheta3(mh3, std::vector<double>(1));

   // Draw realizations of the posterior predictive.
   size_t curdx1=0;
   size_t cumdx1=0;
   size_t curdx2=0;
   size_t cumdx2=0;
   size_t curdx3=0;
   size_t cumdx3=0;
   std::vector<std::vector<double> > a1,a2,a3,b1,b2,b3;
   std::vector<double> theta1,theta2,theta3;
#ifdef _OPENMPI
   double tstart=0.0,tend=0.0;
   if(mpirank==0) tstart=MPI_Wtime();
#endif


   // Mean trees first
   if(mpirank==0) cout << "Calculating Pareto front and set for mean trees" << endl;

   size_t ii=0;
   std::vector<std::vector<std::vector<double> > > aset(nd, std::vector<std::vector<double> >(0, std::vector<double>(0)));
   std::vector<std::vector<std::vector<double> > > bset(nd, std::vector<std::vector<double> >(0, std::vector<double>(0)));
   std::vector<std::vector<std::vector<double> > > front(nd, std::vector<std::vector<double> >(0, std::vector<double>(0)));

   // 0. Create hashmap to get xi indices more easily.
   // Should also work even if xi=xicuts. 
   std::vector<std::unordered_map<double, size_t> > ximaps;
   ximaps.reserve(p);
   for (size_t w=0;w<p;w++) {
      std::unordered_map<double, size_t> ximap;
      for (size_t c=0;c<xi[w].size();c++) 
         ximap[xi[w][c]] = c;
      ximaps.push_back(ximap);
   }



   for(size_t i=0;i<nd;i++) {

      // Load a realization from model 1
      curdx1=0;
      for(size_t j=0;j<m1;j++) {
         onn1[j]=e1_ots.at(i*m1+j);
         oid1[j].resize(onn1[j]);
         ov1[j].resize(onn1[j]);
         oc1[j].resize(onn1[j]);
         otheta1[j].resize(onn1[j]);
         for(size_t k=0;k< (size_t)onn1[j];k++) {
            oid1[j][k]=e1_oid.at(cumdx1+curdx1+k);
            ov1[j][k]=e1_ovar.at(cumdx1+curdx1+k);
            oc1[j][k]=e1_oc.at(cumdx1+curdx1+k);
            otheta1[j][k]=e1_otheta.at(cumdx1+curdx1+k);
         }
         curdx1+=(size_t)onn1[j];
      }
      cumdx1+=curdx1;
      ambm1.loadtree(0,m1,onn1,oid1,ov1,oc1,otheta1);

      // Load a realization from model 2
      curdx2=0;
      for(size_t j=0;j<m2;j++) {
         onn2[j]=e2_ots.at(i*m2+j);
         oid2[j].resize(onn2[j]);
         ov2[j].resize(onn2[j]);
         oc2[j].resize(onn2[j]);
         otheta2[j].resize(onn2[j]);
         for(size_t k=0;k< (size_t)onn2[j];k++) {
            oid2[j][k]=e2_oid.at(cumdx2+curdx2+k);
            ov2[j][k]=e2_ovar.at(cumdx2+curdx2+k);
            oc2[j][k]=e2_oc.at(cumdx2+curdx2+k);
            otheta2[j][k]=e2_otheta.at(cumdx2+curdx2+k);
         }
         curdx2+=(size_t)onn2[j];
      }
      cumdx2+=curdx2;
      ambm2.loadtree(0,m2,onn2,oid2,ov2,oc2,otheta2);

      // Load a realization from model 3
      if(threeresponse) {
         curdx3=0;
         for(size_t j=0;j<m3;j++) {
            onn3[j]=e3_ots.at(i*m3+j);
            oid3[j].resize(onn3[j]);
            ov3[j].resize(onn3[j]);
            oc3[j].resize(onn3[j]);
            otheta3[j].resize(onn3[j]);
            for(size_t k=0;k< (size_t)onn3[j];k++) {
               oid3[j][k]=e3_oid.at(cumdx3+curdx3+k);
               ov3[j][k]=e3_ovar.at(cumdx3+curdx3+k);
               oc3[j][k]=e3_oc.at(cumdx3+curdx3+k);
               otheta3[j][k]=e3_otheta.at(cumdx3+curdx3+k);
            }
            curdx3+=(size_t)onn3[j];
         }
         cumdx3+=curdx3;
         ambm3.loadtree(0,m3,onn3,oid3,ov3,oc3,otheta3);
      }

      // Calculate Pareto front and set
      // std::vector<double> aout(p),bout(p);
      if(i>=snd && i<end) {
         // convert ensembles to hyperrectangle format
         ambm1.ens2rects(a1, b1, theta1, minx, maxx, p);
         ambm2.ens2rects(a2, b2, theta2, minx, maxx, p);

         std::list<std::vector<double> > ltheta1;
         for (double t1: theta1) ltheta1.push_back({t1+fmean1});
         combine_ensembles(p, xi, ximaps, minx, maxx, a1, b1, ltheta1, a2, b2, theta2, fmean2, asol, bsol, thetasol, true);

         if(threeresponse) {
            // Possible to avoid copying these three containers? 
            a1 = asol;  
            b1 = bsol;
            ltheta1 = thetasol;
            asol.clear();
            bsol.clear();
            thetasol.clear();
            
            ambm3.ens2rects(a3,b3,theta3,minx,maxx,p); 
            combine_ensembles(p, xi, ximaps, minx, maxx, a1, b1, ltheta1, a3, b3, theta3, fmean3, asol, bsol, thetasol, true);
         }

         // then we can get the front,set, and then clear these vectors
         thetasol.sort();

         std::vector<size_t> frontdx;
         frontdx=find_pareto_front(1,thetasol.size(),thetasol);
         // Save the Pareto front and set
         // Note frontdx has indices in 1..sizeof(front) so we need to -1 to get correct vector entry.
         // Also note we have to remap to original index to get the correct corresponding VHRs in the Pareto Set.
         aset[ii].resize(frontdx.size());
         bset[ii].resize(frontdx.size());
         front[ii].resize(frontdx.size());
         for(size_t k=0;k<frontdx.size();k++) {
               aset[ii][k].resize(p);
               bset[ii][k].resize(p);
               std::list<std::vector<double> >::iterator it = std::next(thetasol.begin(),frontdx[k]-1);
               for(size_t j=0;j<p;j++) {
                  // aset[ii][k][j]=asol[frontdx[k]-1][j];
                  // bset[ii][k][j]=bsol[frontdx[k]-1][j];
                  aset[ii][k][j]=asol[(size_t)((*it).at(d))-1][j];
                  bset[ii][k][j]=bsol[(size_t)((*it).at(d))-1][j];
               }
               front[ii][k].resize(d);
               for (size_t dd=0;dd<d;dd++) front[ii][k][dd]=(*it).at(dd);
         }

         // if(mpirank==0) cout << "thetasol=" << thetasol.size() << " frontsize=" << frontdx.size() << endl;
         asol.clear();
         bsol.clear();
         thetasol.clear();
         frontdx.clear();

         ii++;
      }
   }



/* Variances trees Pareto Front currently not implemented.

   // Variance trees second
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
      cout << "Pareto front and set draw time was " << (tend-tstart)/60.0 << " minutes." << endl;
   }
#endif




   // Save the draws.
   if(mpirank==0) cout << "Saving Pareto front and set...";

   std::ofstream omf(folder + modelname + ".mopareto" + std::to_string(mpirank));
   for(size_t i=0;i<rnd;i++) {
      omf << front[i].size() << " ";
      for(size_t dd=0;dd<d;dd++) 
         for(size_t j=0;j<front[i].size();j++)
            omf << std::scientific << front[i][j][dd] << " ";
      // for(size_t j=0;j<front[i].size();j++)
      //    omf << std::scientific << front[i][j][0] << " ";
      // for(size_t j=0;j<front[i].size();j++)
      //    omf << std::scientific << front[i][j][1] << " ";
      for(size_t j=0;j<p;j++)
         for(size_t k=0;k<front[i].size();k++)
            omf << std::scientific << aset[i][k][j] << " ";
      for(size_t j=0;j<p;j++)
         for(size_t k=0;k<front[i].size();k++)
            omf << std::scientific << bset[i][k][j] << " ";
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

