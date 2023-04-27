//     lebrt.h: Lebesgue tree BT model class definition.
//     Copyright (C) 2012-2022 Matthew T. Pratola
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


#ifndef GUARD_lebrt_h
#define GUARD_lebrt_h

#include "tree.h"
#include "treefuns.h"
#include "dinfo.h"
#include "brt.h"


class lesinfo : public sinfo { //sufficient statistics (will depend on end node model)
public:
   lesinfo():sinfo(),w(0.0) {}
   lesinfo(const lesinfo& is):sinfo(is),w(is.w) {}
   virtual ~lesinfo() {}  //need this so memory is properly freed in derived classes.
   double w;
   // compound addition operator needed when adding suff stats
   virtual sinfo& operator+=(const sinfo& rhs) {
      sinfo::operator+=(rhs);
      const lesinfo& lerhs=static_cast<const lesinfo&>(rhs);
//      w=std::max(w,lerhs.w);
      this->w+=lerhs.w;
      return *this;
   }
   // assignment operator for suff stats
   virtual sinfo& operator=(const sinfo& rhs)
   {
      if(&rhs != this) {
         sinfo::operator=(rhs);
         const lesinfo& lerhs=static_cast<const lesinfo&>(rhs);
         this->w = lerhs.w;
      }
      return *this;
   }
   // addition opertor is defined in terms of compound addition
   const lesinfo operator+(const lesinfo& other) const {
      lesinfo result = *this; //copy of myself.
      result += other;
      return result;
   }
   // a union-like operator (logical or on weights, additive on sample sizes)
   virtual sinfo& operator|=(const sinfo& rhs)
   {
      sinfo::operator+=(rhs);
      const lesinfo& lerhs=static_cast<const lesinfo&>(rhs);
      this->w =std::max(this->w,lerhs.w);
      return *this;
   }
};

class lebrt : public brt 
{
public:
   //--------------------
   //classes
   // tprior and mcmcinfo are same as in brt
   class cinfo { //parameters for end node model prior
   public:
      cinfo() {}
   };
   //--------------------
   //constructors/destructors
   lebrt():brt() {}
   //--------------------
   //methods
   void draw(rn& gen);
   void draw_mpislave(rn& gen);
//   void setdata(size_t n) { dinfo mydi; u.resize(n); mydi.n=n; mydi.p=1; mydi.x=&u[0]; this->di=&mydi; resid.resize(di->n); yhat.resize(di->n); setf(); setr(); }
   void reshard(rn& gen) { for(size_t i=0;i<di->n;i++) di->x[i]=gen.uniform(); }
   void setci() { t.settheta(1.0); }   //not exactly conditioning info, but convenient place to
                                       //initialize root tree theta to be 1.0 instead of default 0.0
   virtual double drawnodetheta(sinfo& si, rn& gen);
   virtual double lm(sinfo& si);
   virtual void add_observation_to_suff(diterator& diter, sinfo& si);
   virtual sinfo* newsinfo() { return new lesinfo; }
   virtual std::vector<sinfo*>& newsinfovec() { std::vector<sinfo*>* si= new std::vector<sinfo*>; return *si; }
   virtual std::vector<sinfo*>& newsinfovec(size_t dim) { std::vector<sinfo*>* si = new std::vector<sinfo*>; si->resize(dim); for(size_t i=0;i<dim;i++) si->push_back(new lesinfo); return *si; }
   virtual void local_mpi_reduce_allsuff(std::vector<sinfo*>& siv);
   virtual void local_mpi_sr_suffs(sinfo& sil, sinfo& sir);
   virtual void local_getsuff(diterator& diter, tree::tree_p nx, size_t v, size_t c, sinfo& sil, sinfo& sir);
   virtual void local_setr(diterator& diter);
   virtual void local_setf(diterator& diter);
   void pr();
   void cookdinfl(std::vector<double>& cdinfl, double* sigma); //Cook's distance
   void kldivinfl(std::vector<double>& klinfl, double* sigma); //KL-divergence based influence metric

   //--------------------
   //data
//   std::vector<double> u;
   //--------------------------------------------------
   //stuff that maybe should be protected
protected:
   //--------------------
   //model information
   cinfo ci; //conditioning info (e.g. other parameters and prior and end node models)
   //--------------------
   //data
   //--------------------
   //mcmc info
   //--------------------
   //methods
};


#endif
