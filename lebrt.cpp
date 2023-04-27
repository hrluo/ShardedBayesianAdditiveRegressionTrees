//     lebrt.cpp: Lebesgue tree BT model class methods.
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


#include "lebrt.h"
//#include "brtfuns.h"
#include <iostream>
#include <map>
#include <vector>

using std::cout;
using std::endl;


//--------------------------------------------------
//a single iteration of the MCMC for lebrt model
void lebrt::draw(rn& gen)
{
   //All the usual steps
   brt::draw(gen);

   // Update the in-sample predicted vector
//   setf();

   // Update the in-sample residual vector
   setr();

}
//--------------------------------------------------
//slave controller for draw when using MPI
void lebrt::draw_mpislave(rn& gen)
{
   //All the usual steps
   brt::draw_mpislave(gen);

   // Update the in-sample predicted vector
 //  setf();

   // Update the in-sample residual vector
   setr();
}
//--------------------------------------------------
//draw theta for a single bottom node for the lebrt model
double lebrt::drawnodetheta(sinfo& si, rn& gen)
{
   lesinfo& lesi=static_cast<lesinfo&>(si);

   return lesi.w;
}
//--------------------------------------------------
//lm: log of integrated likelihood, depends on prior and suff stats
double lebrt::lm(sinfo& si)
{
   lesinfo& lesi=static_cast<lesinfo&>(si);

   return log(lesi.n>0);
}
//--------------------------------------------------
//Add in an observation, this has to be changed for every model.
//Note that this may well depend on information in brt with our leading example
//being double *sigma in cinfo for the case of e~N(0,sigma_i^2).
// Note that we are using the training data and the brt object knows the training data
//     so all we need to specify is the row of the data (argument size_t i).
void lebrt::add_observation_to_suff(diterator& diter, sinfo& si)
{
   lesinfo& lesi=static_cast<lesinfo&>(si);

   lesi.n+=1;
}
//--------------------------------------------------
//getsuff used for birth.  Overides the brt::local_getsuff method.
void lebrt::local_getsuff(diterator& diter, tree::tree_p nx, size_t v, size_t c, sinfo& sil, sinfo& sir)    
{
   double *xx;//current x
   lesinfo& lesil=static_cast<lesinfo&>(sil);
   lesinfo& lesir=static_cast<lesinfo&>(sir);
   sil.n=0; sir.n=0;

   // calculate the update weight parameters for this split at c.
   int L,U;
   size_t p=1; //p=1 for lebrt objects because there is only one splitting variable, namely what we call the "u".
   double a;
   double b;
   double z;
   L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();
   nx->rgi(p-1,&L,&U);

   // Now we have the interval endpoints, put corresponding values in a,b matrices.
   if(L!=std::numeric_limits<int>::min()) a=(*xi)[p-1][L];
   else a=0.0;  // the u's are *assumed* to be in [0,1].  If not, big problem here!
   if(U!=std::numeric_limits<int>::max()) b=(*xi)[p-1][U];
   else b=1.0;  // the u's are *assumed* to be in [0,1].  If not, big problem here!

   // so current weight at node nx is (b-a)/(1.0-0.0).  Now split it according to c.
   z=(*xi)[p-1][c];
   lesil.w=(z-a)/(1.0-0.0);
   lesir.w=(b-z)/(1.0-0.0);
   for(;diter<diter.until();diter++)
   {
      xx = diter.getxp();
      if(nx==t.bn(diter.getxp(),*xi)) { //does the bottom node = xx's bottom node
         if(xx[v] < (*xi)[v][c]) {
//cout << "<looping through diter: v=" << v << " c=" << c << " xx[v]=" << xx[0] << "  cut=" << (*xi)[v][c] << endl;

               //sil.n +=1;
               add_observation_to_suff(diter,sil);
          } else {
//cout << ">looping through diter: v=" << v << " c=" << c << " xx[v]=" << xx[0] << endl;

               //sir.n +=1;
               add_observation_to_suff(diter,sir);
          }
      }
   }
}

//--------------------------------------------------
// MPI virtualized part for sending/receiving left,right suffs
void lebrt::local_mpi_sr_suffs(sinfo& sil, sinfo& sir)
{
#ifdef _OPENMPI
   lesinfo& lesil=static_cast<lesinfo&>(sil);
   lesinfo& lesir=static_cast<lesinfo&>(sir);
   if(rank==0) { // MPI receive all the answers from the slaves

      MPI_Status status;
      char buffer[SIZE_UINT6];
      int position=0;
      unsigned int ln,rn;
      double lw,rw;
      for(size_t i=1; i<=(size_t)tc; i++) {
         position=0;
         MPI_Recv(buffer,SIZE_UINT6,MPI_PACKED,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&ln,1,MPI_UNSIGNED,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&rn,1,MPI_UNSIGNED,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&lw,1,MPI_DOUBLE,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT6,&position,&rw,1,MPI_DOUBLE,MPI_COMM_WORLD);

         lesil.n=lesil.n+(size_t)ln;
         lesir.n=lesir.n+(size_t)rn;
         lesil.w=std::max(lesil.w,lw);
         lesir.w=std::max(lesir.w,rw);
      }
   }
   else // MPI send all the answers to root
   {
      char buffer[SIZE_UINT6];
      int position=0;  
      unsigned int ln,rn;
      ln=(unsigned int)sil.n;
      rn=(unsigned int)sir.n;

      MPI_Pack(&ln,1,MPI_UNSIGNED,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      MPI_Pack(&rn,1,MPI_UNSIGNED,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      MPI_Pack(&lesil.w,1,MPI_DOUBLE,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      MPI_Pack(&lesir.w,1,MPI_DOUBLE,buffer,SIZE_UINT6,&position,MPI_COMM_WORLD);
      MPI_Send(buffer,SIZE_UINT6,MPI_PACKED,0,0,MPI_COMM_WORLD);
   }
#endif   
}
//--------------------------------------------------
//allsuff(2) -- the MPI communication part of local_mpiallsuff.  This is model-specific.
void lebrt::local_mpi_reduce_allsuff(std::vector<sinfo*>& siv)
{
#ifdef _OPENMPI
   unsigned int nvec[siv.size()];
   double wvec[siv.size()];

   for(size_t i=0;i<siv.size();i++) { // on root node, these should be 0 because of newsinfo().
      lesinfo* lesi=static_cast<lesinfo*>(siv[i]);
      nvec[i]=(unsigned int)lesi->n;    // cast to int
      wvec[i]=lesi->w;
   }
// cout << "pre:" << siv[0]->n << " " << siv[1]->n << endl;

   // MPI sum
   // MPI_Allreduce(MPI_IN_PLACE,&nvec,siv.size(),MPI_UNSIGNED,MPI_SUM,MPI_COMM_WORLD);
   // MPI_Allreduce(MPI_IN_PLACE,&sumwvec,siv.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
   // MPI_Allreduce(MPI_IN_PLACE,&sumwyvec,siv.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

   if(rank==0) {
      MPI_Status status;
      unsigned int tempnvec[siv.size()];
      tree::npv bots;
      t.getbots(bots);
      size_t B=bots.size();
      size_t p=1; //p=1 for lebrt objects because there is only one splitting variable, namely what we call the "u".
      double a;
      double b;

      // initialize a,b
      for(size_t k=0;k<B;k++) {
         int L,U;
         L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();
         bots[k]->rgi(p-1,&L,&U);

         // Now we have the interval endpoints, put corresponding values in a,b matrices.
         if(L!=std::numeric_limits<int>::min()) a=(*xi)[p-1][L];
         else a=0.0;  // the u's are *assumed* to be in [0,1].  If not, big problem here!
         if(U!=std::numeric_limits<int>::max()) b=(*xi)[p-1][U];
         else b=1.0;  // the u's are *assumed* to be in [0,1].  If not, big problem here!
         wvec[k]=(b-a)/(1.0-0.0); //the u's are *assumed* to be in [0,1].  If not, big problem here!
      }

      // receive nvec, wvec, update and send back.
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Recv(&tempnvec,siv.size(),MPI_UNSIGNED,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
         for(size_t j=0;j<siv.size();j++)
            nvec[j]+=tempnvec[j];
      }

      MPI_Request *request=new MPI_Request[tc];
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(&nvec,siv.size(),MPI_UNSIGNED,i,0,MPI_COMM_WORLD,&request[i-1]);
      }

      // cast back to lesi
      for(size_t i=0;i<siv.size();i++) {
         lesinfo* lesi=static_cast<lesinfo*>(siv[i]);
         lesi->n=(size_t)nvec[i];    // cast back to size_t
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;

      // send wvec.
      request=new MPI_Request[tc];
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(&wvec,siv.size(),MPI_DOUBLE,i,0,MPI_COMM_WORLD,&request[i-1]);
      }

      // cast back to lesi
      for(size_t i=0;i<siv.size();i++) {
         lesinfo* lesi=static_cast<lesinfo*>(siv[i]);
         lesi->w=wvec[i];
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
   }
   else {
      MPI_Request *request=new MPI_Request;
      MPI_Status status;

      // send/recv nvec      
      MPI_Isend(&nvec,siv.size(),MPI_UNSIGNED,0,0,MPI_COMM_WORLD,request);
      MPI_Wait(request,MPI_STATUSES_IGNORE);
      delete request;

      MPI_Recv(&nvec,siv.size(),MPI_UNSIGNED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

      // receive wvec
      MPI_Recv(&wvec,siv.size(),MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

      // cast back to msi
      for(size_t i=0;i<siv.size();i++) {
         lesinfo* lesi=static_cast<lesinfo*>(siv[i]);
         lesi->w=wvec[i];
      }
   }

#endif
}

void lebrt::local_setr(diterator& diter)
{
//   tree::tree_p bn;

   for(;diter<diter.until();diter++) {
//      bn = t.bn(diter.getxp(),*xi);
      resid[*diter] = 0.0;
   }
}

void lebrt::local_setf(diterator& diter)
{
   tree::tree_p bn;
cout << rank << ": setf: updating yhat entry " << *diter << endl;

   for(;diter<diter.until();diter++) {
      bn = t.bn(diter.getxp(),*xi);
      yhat[*diter] = bn->gettheta();
   }
}

//--------------------------------------------------
// Influence metrics.  See Pratola, George and McCulloch (2021)

// Cook's distance
void lebrt::cookdinfl(std::vector<double>& cdinfl, double* sigma)
{
   brt::cookdinfl(cdinfl);
   for(size_t i=0;i<di->n;i++) {
      double stdres = resid[i]/sigma[i];
      double stdres2 = stdres*stdres;
      cdinfl[i] *= stdres2;
   }
}

//KL-divergence based influence metric
void lebrt::kldivinfl(std::vector<double>& klinfl, double* sigma)
{
   brt::kldivinfl(klinfl);
   for(size_t i=0;i<di->n;i++)
      if(klinfl[i]!=std::numeric_limits<double>::infinity()) {
         double stdres = resid[i]/sigma[i];
         double stdres2 = stdres*stdres;
         klinfl[i]=-0.5*std::log(2*3.14159)-std::log(sigma[i])-0.5*stdres2;
      }
}


//--------------------------------------------------
//pr for brt
void lebrt::pr()
{
   std::cout << "***** lebrt object:\n";
   cout << "Conditioning info: (none)" << endl;
   brt::pr();
}

/*
//--------------------------------------------------
//--------------------------------------------------
//bd: birth/death
void lebrt::bd(rn& gen)
{
//   cout << "--------------->>into bd" << endl;
   tree::npv goodbots;  //nodes we could birth at (split on)
   double PBx = getpb(t,*xi,mi.pb,goodbots); //prob of a birth at x

   if(gen.uniform() < PBx) { //do birth or death
      mi.bproposal++;
      //--------------------------------------------------
      //draw proposal
      tree::tree_p nx; //bottom node
      size_t v,c; //variable and cutpoint
      double pr; //part of metropolis ratio from proposal and prior
      bprop(t,*xi,tp,mi.pb,goodbots,PBx,nx,v,c,pr,gen);

      //--------------------------------------------------
      //compute sufficient statistics
      sinfo& sil = *newsinfo();
      sinfo& sir = *newsinfo();
      sinfo& sit = *newsinfo();

      getsuff(nx,v,c,sil,sir);
      // sit = sil + sir; NO! The + operator cannot be overloaded, so instead we do this:
      sit += sil;
      sit += sir;

      //--------------------------------------------------
      //compute alpha
      bool hardreject=true;
      double lalpha=0.0;
      double lml, lmr, lmt;  // lm is the log marginal left,right,total
      if((sil.n>=mi.minperbot) && (sir.n>=mi.minperbot)) { 
         lml=lm(sil); lmr=lm(sir); lmt=lm(sit);
         hardreject=false;
         lalpha = log(pr) + (lml+lmr-lmt);
         lalpha = std::min(0.0,lalpha);
      }
      //--------------------------------------------------
      //try metrop
      double thetal,thetar; //parameters for new bottom nodes, left and right
      double uu = gen.uniform();
#ifdef _OPENMPI
      MPI_Request *request = new MPI_Request[tc];
#endif
      if( !hardreject && (log(uu) < lalpha) ) {
         thetal = 0.0;//drawnodetheta(sil,gen);
         thetar = 0.0;//drawnodetheta(sir,gen);
         t.birthp(nx,v,c,thetal,thetar);
         mi.baccept++;
#ifdef _OPENMPI
//        cout << "accept birth " << lalpha << endl;
         const int tag=MPI_TAG_BD_BIRTH_VC_ACCEPT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
      else { //transmit reject over MPI
//        cout << "reject birth " << lalpha << endl;
         const int tag=MPI_TAG_BD_BIRTH_VC_REJECT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
#else
      }
#endif
      delete &sil;
      delete &sir;
      delete &sit;
#ifdef _OPENMPI
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
#endif
   } else {
      mi.dproposal++;
      //--------------------------------------------------
      //draw proposal
      double pr;  //part of metropolis ratio from proposal and prior
      tree::tree_p nx; //nog node to death at
      dprop(t,*xi,tp,mi.pb,goodbots,PBx,nx,pr,gen);

      //--------------------------------------------------
      //compute sufficient statistics
      //sinfo sil,sir,sit;
      sinfo& sil = *newsinfo();
      sinfo& sir = *newsinfo();
      sinfo& sit = *newsinfo();
      getsuff(nx->getl(),nx->getr(),sil,sir);
      // sit = sil + sir; NO! The + operator cannot be overloaded, so instead we do this:
      sit += sil;
      sit += sir;

      //--------------------------------------------------
      //compute alpha
      double lml, lmr, lmt;  // lm is the log marginal left,right,total
      lml=lm(sil); lmr=lm(sir); lmt=lm(sit);
      double lalpha = log(pr) + (lmt - lml - lmr);
      lalpha = std::min(0.0,lalpha);

      //--------------------------------------------------
      //try metrop
      double theta;
#ifdef _OPENMPI
      MPI_Request *request = new MPI_Request[tc];
#endif
      if(log(gen.uniform()) < lalpha) {
         theta = 0.0;//drawnodetheta(sit,gen);
         t.deathp(nx,theta);
         mi.daccept++;
#ifdef _OPENMPI
//        cout << "accept death " << lalpha << endl;
         const int tag=MPI_TAG_BD_DEATH_LR_ACCEPT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
      else { //transmit reject over MPI
//        cout << "reject death " << lalpha << endl;
         const int tag=MPI_TAG_BD_DEATH_LR_REJECT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
#else
      }
#endif
      delete &sil;
      delete &sir;
      delete &sit;
#ifdef _OPENMPI
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
#endif
   }
}*/