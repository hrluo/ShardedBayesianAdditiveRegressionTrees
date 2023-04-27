//     brtfuns.cpp: Base BT model class helper functios.
//     Copyright (C) 2012-2016 Matthew T. Pratola, Robert E. McCulloch and Hugh A. Chipman
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
//     Robert E. McCulloch: robert.e.mculloch@gmail.com
//     Hugh A. Chipman: hughchipman@gmail.com
//
//     Contributing Authors
//     Akira Horiguchi: horiguchi.6@osu.edu (find_pareto_front, not_dominated, combine_ensembles)


#include "brtfuns.h"

//--------------------------------------------------
//make xinfo = cutpoints
void makexinfo(size_t p, size_t n, double *x, xinfo& xi, size_t nc)
{
   double xinc;

   //compute min and max for each x
   std::vector<double> minx(p,INFINITY);
   std::vector<double> maxx(p,-INFINITY);
   double xx;

   #ifdef _OPENMPI
   int mpirank=0;
   MPI_Comm_rank(MPI_COMM_WORLD,&mpirank);
   if(mpirank>0) {
   #endif
   for(size_t i=0;i<p;i++) {
      for(size_t j=0;j<n;j++) {
         xx = *(x+p*j+i);
         if(xx < minx[i]) minx[i]=xx;
         if(xx > maxx[i]) maxx[i]=xx;
      }
   }

   //if MPI codepath, aggregate the min/max values across slaves.
   #ifdef _OPENMPI
   }
   for(size_t i=0;i<p;i++) {
      MPI_Allreduce(MPI_IN_PLACE,&minx[i],1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&maxx[i],1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      cout << "mpirank=" << mpirank << ": minx[" << i << "]=" << minx[i] << " maxx[" << i << "]=" << maxx[i] << endl;
   }
   #endif

   //make grid of nc cutpoints between min and max for each x.
   xi.resize(p);
   for(size_t i=0;i<p;i++) {
      xinc = (maxx[i]-minx[i])/(nc+1.0);
      xi[i].resize(nc);
      for(size_t j=0;j<nc;j++) xi[i][j] = minx[i] + (j+1)*xinc;
   }
}

//--------------------------------------------------
//compute prob of a birth, goodbots will contain all the good bottom nodes
double getpb(tree& t, xinfo& xi, double pipb, tree::npv& goodbots)
{
   double pb;  //prob of birth to be returned
   tree::npv bnv; //all the bottom nodes
   t.getbots(bnv);
   for(size_t i=0;i!=bnv.size();i++)
      if(cansplit(bnv[i],xi)) goodbots.push_back(bnv[i]);
   if(goodbots.size()==0) { //are there any bottom nodes you can split on?
      pb=0.0;
   } else {
      if(t.treesize()==1) pb=1.0; //is there just one node?
      else pb=pipb;
   }
   return pb;
}
//--------------------------------------------------
//bprop: function to generate birth proposal
void bprop(tree& x, xinfo& xi, brt::tprior& tp, double pb, tree::npv& goodbots, double& PBx, tree::tree_p& nx, size_t& v, size_t& c, double& pr, rn& gen)
{

      //draw bottom node, choose node index ni from list in goodbots
      size_t ni = floor(gen.uniform()*goodbots.size());
      nx = goodbots[ni]; //the bottom node we might birth at

      //draw v,  the variable
      std::vector<size_t> goodvars; //variables nx can split on
      if(tp.sharding && nx->depth()<=tp.shardepth) { //if sharding and less than shardepth then we only split on shard var
         v=0;
      }
      else { //else if sharding but deeper than shardepth we choose from vars 1..p
         if(tp.sharding) getgoodvars_sharded(nx,xi,goodvars);
         else getgoodvars(nx,xi,goodvars); //else we are not sharding so default to choose vars from 0..p.
         size_t vi = floor(gen.uniform()*goodvars.size()); //index of chosen split variable
         v = goodvars[vi];
      }

      //draw c, the cutpoint
      int L,U;
      L=0; U = xi[v].size()-1;
      nx->rg(v,&L,&U);
      c = L + floor(gen.uniform()*(U-L+1)); //U-L+1 is number of available split points

      //--------------------------------------------------
      //compute things needed for metropolis ratio

      double Pbotx = 1.0/goodbots.size(); //proposal prob of choosing nx
      size_t dnx = nx->depth();

      if(tp.sharding) dnx=std::max(dnx-tp.shardepth-1.0,0.0); //if we are sharding then
                                                          //don't count sharding splits as contributing 
                                                          //to the depth calculation for actual variables.

      double PGnx;
      // if(tp.sharding && nx->depth()<=tp.shardepth)
      //    PGnx = tp.shardpsplit;  //if sharding and we haven't used all the shard splits, we split according to shardpsplit (default is 0.5)
      // else
         PGnx = tp.alpha/pow(1.0 + dnx,tp.beta); //prior prob of growing at nx


      double PGly, PGry; //prior probs of growing at new children (l and r) of proposal
      // if(tp.sharding && nx->depth()<tp.shardepth)
      // {
      //    PGly=tp.shardpsplit; PGry=tp.shardpsplit;  //if sharding and we haven't used all the shard splits, children would shard split again at rate shardpsplit
      // }
      // else {
         if(goodvars.size()>1) { //know there are variables we could split l and r on
            PGly = tp.alpha/pow(1.0 + dnx+1.0,tp.beta); //depth of new nodes would be one more
            PGry = PGly;
         } else { //only had v to work with, if it is exhausted at either child need PG=0
            if((int)(c-1)<L) { //v exhausted in new left child l, new upper limit would be c-1
               PGly = 0.0;
            } else {
               PGly = tp.alpha/pow(1.0 + dnx+1.0,tp.beta);
            }
            if(U < (int)(c+1)) { //v exhausted in new right child r, new lower limit would be c+1
               PGry = 0.0;
            } else {
               PGry = tp.alpha/pow(1.0 + dnx+1.0,tp.beta);
            }
         }
      // }

      double PDy; //prob of proposing death at y
      if(goodbots.size()>1) { //can birth at y because splittable nodes left
         PDy = 1.0 - pb;
      } else { //nx was the only node you could split on
         if((PGry==0) && (PGly==0)) { //cannot birth at y
            PDy=1.0;
         } else { //y can birth at either l or r
            PDy = 1.0 - pb;
         }
      }

      double Pnogy; //death prob of choosing the nog node at y
      size_t nnogs = x.nnogs();
      tree::tree_p nxp = nx->getp();
      if(nxp==0) { //no parent, nx is the top and only node
         Pnogy=1.0;
      } else {
         if(nxp->ntype() == 'n') { //if parent is a nog, number of nogs same at x and y
            Pnogy = 1.0/nnogs;
         } else { //if parent is not a nog, y has one more nog.
           Pnogy = 1.0/(nnogs+1.0);
         }
      }

      if(tp.sharding && nx->depth()<tp.shardepth)  //if we're sharding and still have room, split again.
         pr = 1.0;
      else
         pr = (PGnx*(1.0-PGly)*(1.0-PGry)*PDy*Pnogy)/((1.0-PGnx)*Pbotx*PBx);
}
//--------------------------------------------------
// death proposal
void dprop(tree& x, xinfo& xi, brt::tprior& tp, double pb, tree::npv& goodbots, double& PBx, tree::tree_p& nx, double& pr, rn& gen)
{
      //draw nog node, any nog node is a possibility
      tree::npv nognds; //nog nodes
      x.getnogs(nognds);
      size_t ni = floor(gen.uniform()*nognds.size());
      nx = nognds[ni]; //the nog node we might kill children at

      //--------------------------------------------------
      //compute things needed for metropolis ratio

      double PGny; //prob the nog node grows
      size_t dny = nx->depth();

       if(tp.sharding) dny=std::max(dny-tp.shardepth-1.0,0.0); //if we are sharding then
                                                           //don't count sharding splits as contributing 
                                                           //to the depth calculation for actual variables.

      // if(tp.sharding && nx->depth()<=tp.shardepth)
      //    PGny = tp.shardpsplit;  //if sharding and we haven't used all the shard splits, we split according to shardpsplit (default is 0.5)
      // else
         PGny = tp.alpha/pow(1.0+dny,tp.beta);

      //better way to code these two?
      double PGlx;
      double PGrx;
      // if(tp.sharding && nx->depth()<tp.shardepth)
      // {
      //    PGlx = tp.shardpsplit;//if sharding and we haven't used all the shard splits, children would shard split again at rate shardpsplit
      //    PGrx = tp.shardpsplit;
      // }
      // else
      // {
         PGlx = pgrow(nx->getl(),xi,tp);
         PGrx = pgrow(nx->getr(),xi,tp);
      // }   

      double PBy;  //prob of birth move at y
      if(nx->ntype()=='t') { //is the nog node nx the top node
         PBy = 1.0;
      } else {
         PBy = pb;
      }

      double Pboty;  //prob of choosing the nog as bot to split on when y
      int ngood = goodbots.size();
      if(cansplit(nx->getl(),xi)) --ngood; //if can split at left child, lose this one
      if(cansplit(nx->getr(),xi)) --ngood; //if can split at right child, lose this one
      ++ngood;  //know you can split at nx
      Pboty=1.0/ngood;

      double PDx = 1.0-PBx; //prob of a death step at x
      double Pnogx = 1.0/nognds.size();

      if(tp.sharding && nx->depth()<tp.shardepth)  //if we're sharding and nog node is less than sharddepth, no death.
         pr = 0.0;
      else
         pr =  ((1.0-PGny)*PBy*Pboty)/(PGny*(1.0-PGlx)*(1.0-PGrx)*PDx*Pnogx);
}
//--------------------------------------------------
//get prob a node grows, 0 if no good vars, else alpha/(1+d)^beta
double pgrow(tree::tree_p n, xinfo& xi, brt::tprior& tp)
{
   size_t dn = n->depth();

    if(tp.sharding) dn=std::max(dn-tp.shardepth-1.0,0.0); //if we are sharding then
                                                          //don't count sharding splits as contributing 
                                                          //to the depth calculation for actual variables.

   if(cansplit(n,xi)) {
      return tp.alpha/pow(1.0+dn,tp.beta);
   } else {
      return 0.0;
   }
}
//--------------------------------------------------
//calculate beginning and end points of data vector to be accessed in parallel computations
void calcbegend(int n, int my_rank, int thread_count, int* beg, int* end)
{
   if(n>=thread_count) {
      int h = n/thread_count;
      *beg = my_rank*h;
      *end = *beg+h;
      if(my_rank==(thread_count-1)) *end=n;
   }
   else // n < thread_count
   {
      *beg=my_rank;
      *end=my_rank+1;
      if(my_rank>=n) {
         *beg=0;
         *end=0;
      }
   }
}
//--------------------------------------------------
// Functions to support change-of-variable proposal
//--------------------------------------------------
// update the correlation matrix for chgv move taking into account that not all
// variables may be eligible at pertnode.
void updatecormat(tree::tree_p pertnode, xinfo& xi, std::vector<std::vector<double> >& chgv)
{
   int Ln,Un; //L,U for the ``new'' variable
   size_t oldv=pertnode->getv();
   size_t p=chgv.size();

   for(size_t i=0;i<p;i++) {
      if(i!=oldv && std::abs(chgv[oldv][i])>0.0) {
         if(chgv[oldv][i]<0.0)  //swap left,right branches
            pertnode->swaplr();
         getvarLU(pertnode,i,xi,&Ln,&Un);
         if(chgv[oldv][i]<0.0)  //undo the swap
            pertnode->swaplr();
         if(Un<Ln) //we can't transition to variable i here according to the tree structure
            chgv[oldv][i]=0.0;
      }
   }
}

void mpi_update_norm_cormat(size_t rank, size_t tc, tree::tree_p pertnode, xinfo& xi, std::vector<double>& chgvrow, int* chv_lwr, int* chv_upr)
{
#ifdef _OPENMPI
   size_t oldv=pertnode->getv();

   if(rank==0) {
      MPI_Status status;
      double cumsum=0.0;

      MPI_Allreduce(MPI_IN_PLACE,&cumsum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

      for(size_t i=1;i<=tc;i++) {
         size_t slavenumvars=(size_t) (chv_upr[i]-chv_lwr[i]);
         if(slavenumvars>0) {
            MPI_Recv(&chgvrow[chv_lwr[i]],slavenumvars,MPI_DOUBLE,i,MPI_TAG_PERTCHGV_MATRIX_UPDATE,MPI_COMM_WORLD,&status);
         }
      }
   }
   else {
      MPI_Request request;
      double cumsum=0.0;
      size_t slavenumvars=(size_t) (chv_upr[rank]-chv_lwr[rank]);
      double* temp_oldv_var;
      if(slavenumvars>0) temp_oldv_var=new double[slavenumvars];


      if(slavenumvars>0) {
         for(size_t i=0;i<slavenumvars;i++) {
            int Ln,Un;
            temp_oldv_var[i]=chgvrow[chv_lwr[rank]+i];
            if((chv_lwr[rank]+i) != oldv && std::abs(temp_oldv_var[i])>0.0) {
               if(chgvrow[chv_lwr[rank]+i]<0.0)  //swap left,right branches
                  pertnode->swaplr();
               getvarLU(pertnode,chv_lwr[rank]+i,xi,&Ln,&Un);
               if(chgvrow[chv_lwr[rank]+i]<0.0)  //undo the swap
                  pertnode->swaplr();
               if(Un<Ln) //we can't transition to variable var here according to the tree structure
                  temp_oldv_var[i]=0.0;
            }
            cumsum+=std::abs(temp_oldv_var[i]);
         }
      }

      //parallel update of cumulative sums for denominator in matrix row update
      MPI_Allreduce(MPI_IN_PLACE,&cumsum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

      if(slavenumvars>0) {
         //update our part of the vector
         for(size_t i=0;i<slavenumvars;i++)
            temp_oldv_var[i]/=cumsum;

         //send our part of the vector back to root
         MPI_Isend(temp_oldv_var,slavenumvars,MPI_DOUBLE,0,MPI_TAG_PERTCHGV_MATRIX_UPDATE,MPI_COMM_WORLD,&request);
         MPI_Wait(&request,MPI_STATUSES_IGNORE);
         delete[] temp_oldv_var;
      }
   }
#endif
}

//--------------------------------------------------
// renormalize the correlation matrix so that the probability of row sums to 1.
void normchgvrow(size_t row, std::vector<std::vector<double> >& chgv)
{
   double tmp=0.0;
   size_t p=chgv.size();

   for(size_t i=0;i<p;i++)
      tmp+=std::abs(chgv[row][i]);
   for(size_t i=0;i<p;i++)
      chgv[row][i]/=tmp;
}
//--------------------------------------------------
// randomly choose a new variable to transition to from oldv
size_t getchgv(size_t oldv, std::vector<std::vector<double> >& chgv, rn& gen)
{
   double cp=gen.uniform();
   size_t p=chgv.size();
   size_t newv=oldv;
   std::vector<double> cumprob;

   cumprob=chgv[oldv];
   cumprob[1]=std::abs(cumprob[1]);
   for(size_t i=1;i<p;i++)
      cumprob[i]=std::abs(cumprob[i])+cumprob[i-1];

   for(size_t i=0;i<p;i++) {
      if(cumprob[i] >= cp) { //propose transition to this variable
         newv=i;
         i=p;  //break out of loop early
      }        
   }
   return newv;
}
//--------------------------------------------------
// randomly choose a new variable to transition to from oldv
size_t getchgvfromrow(size_t oldv, std::vector<double>& chgvrow, rn& gen)
{
   double cp=gen.uniform();
   size_t p=chgvrow.size();
   size_t newv=oldv;
   std::vector<double> cumprob;

   cumprob=chgvrow;
   cumprob[1]=std::abs(cumprob[1]);
   for(size_t i=1;i<p;i++)
      cumprob[i]=std::abs(cumprob[i])+cumprob[i-1];

   for(size_t i=0;i<p;i++) {
      if(cumprob[i] >= cp) { //propose transition to this variable
         newv=i;
         i=p;  //break out of loop early
      }        
   }
   return newv;
}

//--------------------------------------------------
// Functions to support rotate proposal
//--------------------------------------------------
// Rotate a given rotatable node in the tree.
// node n must be the left child of n->p and also not a root or terminal leaf node.
void rotright(tree::tree_p n)
{
   tree::tree_cp ctstar;
   tree::tree_p tstar;
   tree::tree_p newt = new tree;
   size_t vprime;
   size_t cprime;

   tstar=n->p->r;
   ctstar=n->p->r;
   tree::tree_p newnr = new tree(*ctstar);
   tstar->p=0;

   vprime=n->v;
   cprime=n->c;   
   n->v=n->p->v;
   n->c=n->p->c;
   n->p->v=vprime;
   n->p->c=cprime;
   newt->v=n->v;
   newt->c=n->c;

   newt->p=n->p;
   newt->l=n->r;
   newt->l->p=newt;
   newt->r=tstar;
   tstar->p=newt;
   n->p->r=newt;
   
   n->r=newnr;
   newnr->p=n;
}
//--------------------------------------------------
// Rotate a given rotatable node in the tree.
// node n must be the right child of n->p and also not a root or terminal leaf node.
void rotleft(tree::tree_p n)
{
   tree::tree_cp ctstar;
   tree::tree_p tstar;
   tree::tree_p newt = new tree;
   size_t vprime;
   size_t cprime;

   tstar=n->p->l;
   ctstar=n->p->l;
   tree::tree_p newnl = new tree(*ctstar);
   tstar->p=0;

   vprime=n->v;
   cprime=n->c;
   n->v=n->p->v;
   n->c=n->p->c;
   n->p->v=vprime;
   n->p->c=cprime;
   newt->v=n->v;
   newt->c=n->c;

   newt->p=n->p;
   newt->r=n->l;
   newt->r->p=newt;
   newt->l=tstar;
   tstar->p=newt;
   n->p->l=newt;

   n->l=newnl;
   newnl->p=n;
}
//--------------------------------------------------
// reduce the left sub-tree of the node that was rotated to the top
void reduceleft(tree::tree_p n, size_t v, size_t c)
{
   tree::tree_p temp;

   if(n->r->l && n->r->v==v) //right is not terminal and splits on v
      if(n->r->c >= c) { //then only keep left branch
         delete n->r->r;
         temp=n->r;
         n->r=temp->l;
         temp->l->p=n;
         temp->r=0;
         temp->l=0;
         temp->p=0;
         delete temp;
      }
   if(n->l->l && n->l->v==v) //left is not terminal and splits on v
      if(n->l->c >= c) { // then only keep left branch
         delete n->l->r;
         temp=n->l;
         n->l=temp->l;
         temp->l->p=n;
         temp->r=0;
         temp->l=0;
         temp->p=0;
         delete temp;
      }
}
//--------------------------------------------------
// reduce the right sub-tree of the node that was rotated to the top
void reduceright(tree::tree_p n, size_t v, size_t c)
{
   tree::tree_p temp;

   if(n->r->v==v && n->r->l) //right is not terminal and splits on v
      if(n->r->c <= c) { //then only keep right branch
         delete n->r->l;
         temp=n->r;
         n->r=temp->r;
         temp->r->p=n;
         temp->r=0;
         temp->l=0;
         temp->p=0;
         delete temp;
      }
   if(n->l->v==v && n->l->l) //left is not terminal and splits on v
      if(n->l->c <= c) { // then only keep right branch
         delete n->l->l;
         temp=n->l;
         n->l=temp->r;
         temp->r->p=n;
         temp->r=0;
         temp->l=0;
         temp->p=0;
         delete temp;
      }
}

//--------------------------------------------------
//Collapse tprime into terminal node t of base tree.
//t must be a terminal node of base tree under which we want to collapse
//an entire tree tprime into.  
//tprime must be the root node of the tree to be collapsed under the base tree at node t.
void collapsetree(tree& st, tree::tree_p t, tree::tree_p tprime)
{
      tree::npv tlefts, trights, tbots;
      tree::tree_cp tempt;

      double theta=t->gettheta();

      //simple case, tprime is terminal, t is (always) terminal
      if(!tprime->l) {
         t->settheta(tprime->gettheta()+theta);
      }
      else if(!t->p)  //simple case 2: t is a terminal root node
      {
         st.tonull();
         st=(*tprime); //copy
         st.getbots(tbots);// all terminal nodes below t.
         for(size_t j=0;j<tbots.size();j++)
            tbots[j]->settheta(tbots[j]->gettheta()+theta);
      }
      else { //general case, t is (always) terminal, tprime is not.
         tempt=tprime;
         tree::tree_p tpar=t->p;
         if(t->isleft()) {
            t->p=0;
            delete t;
            tpar->l=new tree(*tempt);
            tpar->l->p=tpar;
            tpar->l->getpathtorootlr(tlefts,trights);
            //collapse redundancies in tprime
            splitall(tpar->l,tlefts,trights);
            tpar->l->getbots(tbots);// all terminal nodes below t.
         }
         else { //isright
            t->p=0;
            delete t;
            tpar->r=new tree(*tempt);
            tpar->r->p=tpar;
            tpar->r->getpathtorootlr(tlefts,trights);
            //collapse redundancies in tprime
            splitall(tpar->r,tlefts,trights);
            tpar->r->getbots(tbots);// all terminal nodes below t.
         }

         for(size_t j=0;j<tbots.size();j++)
            tbots[j]->settheta(tbots[j]->gettheta()+theta);
      }
}
//--------------------------------------------------
//split tree along a sequence of variable, cutpoint pairs
//retains only the part of the tree that remains.
//Note this generates both the left and right subtree branches
//below the terminal node of the current tree.
void splitall(tree::tree_p t, tree::npv& tlefts, tree::npv& trights)
{
   //if path is null, term is allready top parent node
   for(size_t i=0;i<tlefts.size();i++) {
      splitleft(t,tlefts[i]->v,tlefts[i]->c);
   }
   for(size_t i=0;i<trights.size();i++) {
      splitright(t,trights[i]->v,trights[i]->c);
   }
}

//--------------------------------------------------
//split tree along variable v at cutpoint c retaining only 
//part of the tree that is ``left'' of this v,c rule
void splitleft(tree::tree_p t, size_t v, size_t c)
{
   tree::tree_p temp;

   if(t->l) //not terminal node
   {
      if(t->v==v && t->c >= c)
      {
         temp=t->l;
         if(t->isleft())
         {
            t->p->l=temp;
            temp->p=t->p;
         }
         else //isright
         {
            t->p->r=temp;
            temp->p=t->p;
         }
         delete t->r;
         t->p=0;
         t->r=0;
         t->l=0;
         delete t;
         t=temp;
         splitleft(t,v,c);
      }
      else
      {
         splitleft(t->l,v,c);
         splitleft(t->r,v,c);
      }
   }
}
//--------------------------------------------------
//split tree along variable v at cutpoint c retaining only 
//part of the tree that is ``right'' of this v,c rule
void splitright(tree::tree_p t, size_t v, size_t c)
{
   tree::tree_p temp;

   if(t->l) //not terminal node
   {
      if(t->v==v && t->c <= c)
      {
         temp=t->r;
         if(t->isleft())
         {
            t->p->l=temp;
            temp->p=t->p;
         }
         else //isright
         {
            t->p->r=temp;
            temp->p=t->p;
         }
         delete t->l;
         t->p=0;
         t->l=0;
         t->r=0;
         delete t;
         t=temp;
         splitright(t,v,c);
      }
      else
      {
         splitright(t->l,v,c);
         splitright(t->r,v,c);
      }
   }
}
//--------------------------------------------------
//does an actual merge (randomly chosen) 
bool merge(tree::tree_p tl, tree::tree_p tr, tree::tree_p t, size_t v, size_t c, rn& gen)
{
   bool m1,m2;
   tree::tree_cp temptl,temptr;
   int tnwl=0,tnwr=0;
   double u;

   u=gen.uniform();

   if(arenodesleafs(tl,tr)) {  //merging type 3
      if(u<0.5) {
         t->v=tl->v;
         t->c=tl->c;
         t->theta=tl->theta;  //doesn't matter actually it will be overwritten.
         t->l=0;
         t->r=0;
      }
      else 
      {
         temptl=tl;
         temptr=tr;
         t->v=v;
         t->c=c;
         t->l=new tree(*temptl);
         t->r=new tree(*temptr);
         t->l->p=t;
         t->r->p=t;
      }
      return true;
   }
   else if(arenodesequal(tl,tr) && !splitsonv(tl,tr,v)) {  //merging type 4
      m1=mergecount(tl->l,tr->l,v,c,&tnwl);
      m2=mergecount(tl->r,tr->r,v,c,&tnwr);
      if(u < (1.0/(tnwl+tnwr+1.0)) )
      {
         temptl=tl;
         temptr=tr;
         t->v=v;
         t->c=c;
         t->l=new tree(*temptl);
         t->r=new tree(*temptr);
         t->l->p=t;
         t->r->p=t;
      }
      else
      {
         t->v=tl->v;
         t->c=tl->c;
         t->l=new tree;
         t->r=new tree;
         t->l->p=t;
         t->r->p=t;
         tnwl=0;
         tnwr=0;
         m1=merge(tl->l,tr->l,t->l,v,c,gen);
         m2=merge(tl->r,tr->r,t->r,v,c,gen);
      }
      return (m1 & m2);
   }
   else if(splitsonv(tl,tr,v)) {  //merging type 7
      m1=mergecount(tl->r,tr,v,c,&tnwr);
      m2=mergecount(tl,tr->l,v,c,&tnwl);
      if(u < (1.0/(tnwr+tnwl+1.0)) )
      {
         temptl=tl;
         temptr=tr;
         t->v=v;
         t->c=c;
         t->l=new tree(*temptl);
         t->r=new tree(*temptr);
         t->l->p=t;
         t->r->p=t;
      }
      else if(u < ((1.0+tnwr)/(1.0+tnwr+tnwl)) )
      {
         temptl=tl->l;
         t->v=tl->v;
         t->c=tl->c;
         t->l=new tree(*temptl);
         t->l->p=t;
         t->r=new tree;
         t->r->p=t;
         m2=merge(tl->r,tr,t->r,v,c,gen);
      }
      else
      {
         temptr=tr->r;
         t->v=tr->v;
         t->c=tr->c;
         t->r=new tree(*temptr);
         t->r->p=t;
         t->l=new tree;
         t->l->p=t;
         m1=merge(tl,tr->l,t->l,v,c,gen);
      }
      if(!m1) cout << "doh7a" << endl;
      if(!m2) cout << "doh7b" << endl; 
      return (m1 & m2);
   }
   else if(splitsonv(tl,v) && isleaf(tr)) //merging type 1
   {
      m1=mergecount(tl->r,tr,v,c,&tnwr);
      if(u < (1.0/(tnwr + 1.0)) )
      {
         temptl=tl;
         temptr=tr;
         t->v=v;
         t->c=c;
         t->l=new tree(*temptl);
         t->r=new tree(*temptr);
         t->l->p=t;
         t->r->p=t;
      }
      else
      {
         temptl=tl->l;
         t->v=tl->v;
         t->c=tl->c;
         t->l=new tree(*temptl);
         t->l->p=t;
         t->r=new tree;
         t->r->p=t;
         m1=merge(tl->r,tr,t->r,v,c,gen);
      }
      if(!m1) cout << "doh1(m1)" << endl;
      return m1;
   }
   else if(splitsonv(tr,v) && isleaf(tl)) //merging type 2
   {
      m2=mergecount(tl,tr->l,v,c,&tnwl);
      if(u < (1.0/(tnwl+1.0)) )
      {
         temptl=tl;
         temptr=tr;
         t->v=v;
         t->c=c;
         t->l=new tree(*temptl);
         t->r=new tree(*temptr);
         t->l->p=t;
         t->r->p=t;        
      }
      else
      {
         temptr=tr->r;
         t->v=tr->v;
         t->c=tr->c;
         t->r=new tree(*temptr);
         t->r->p=t;
         t->l=new tree;
         t->l->p=t;
         m2=merge(tl,tr->l,t->l,v,c,gen);
      }
      if(!m2) cout << "doh2(m2)" << endl;
      return m2;
   }
   else if(!isleaf(tl) && !isleaf(tr) && splitsonv(tr,v)) { //merge type 6(i)
      m1=mergecount(tl,tr->l,v,c,&tnwr);
      if(u < (1.0/(tnwr + 1.0)) )
      {
         temptl=tl;
         temptr=tr;
         t->v=v;
         t->c=c;
         t->l=new tree(*temptl);
         t->r=new tree(*temptr);
         t->l->p=t;
         t->r->p=t;        
      }
      else
      {
         temptr=tr->r;
         t->v=tr->v;
         t->c=tr->c;
         t->r=new tree(*temptr);
         t->r->p=t;
         t->l=new tree;
         t->l->p=t;
         m1=merge(tl,tr->l,t->l,v,c,gen);
      }
      if(!m1) cout << "doh6i(m1)" << endl;
      return m1;
   }
   else if(!isleaf(tl) && !isleaf(tr) && splitsonv(tl,v)) { //merge type 6(ii)
      m2=mergecount(tl->r,tr,v,c,&tnwl);
      if(u < (1.0/(tnwl + 1.0)) )
      {
         temptl=tl;
         temptr=tr;
         t->v=v;
         t->c=c;
         t->l=new tree(*temptl);
         t->r=new tree(*temptr);
         t->l->p=t;
         t->r->p=t;
      }
      else
      {
         temptl=tl->l;
         t->v=tl->v;
         t->c=tl->c;
         t->l=new tree(*temptl);
         t->l->p=t;
         t->r=new tree;
         t->r->p=t;
         m2=merge(tl->r,tr,t->r,v,c,gen);
      }
      if(!m2) cout << "doh6ii(m2)" << endl;
      return m2;
   }
   else if(!splitsonv(tl,v) && isleaf(tr)) { //merge type 5(i)
      temptl=tl;
      temptr=tr;
      t->v=v;
      t->c=c;
      t->l=new tree(*temptl);
      t->r=new tree(*temptr);
      t->l->p=t;
      t->r->p=t;
      return true;
   }
   else if(!splitsonv(tr,v) && isleaf(tl)) { //merge type 5(ii)
      temptl=tl;
      temptr=tr;
      t->v=v;
      t->c=c;
      t->l=new tree(*temptl);
      t->r=new tree(*temptr);
      t->l->p=t;
      t->r->p=t;
      return true;
   }
   else // default type aka type 8
   {
      temptl=tl;
      temptr=tr;
      t->v=v;
      t->c=c;
      t->l=new tree(*temptl);
      t->r=new tree(*temptr);
      t->l->p=t;
      t->r->p=t;
      return true;
   }

   return false;
}
//--------------------------------------------------
// only to get nways, not to actually do the merge.
bool mergecount(tree::tree_p tl, tree::tree_p tr, size_t v, size_t c, int* nways)
{
   bool m1,m2;
   int tnwl=0,tnwr=0;

   if(arenodesleafs(tl,tr)) {  //merging type 3
      *nways += 2;
      return true;
   }
   else if(arenodesequal(tl,tr) && !splitsonv(tl,tr,v)) {  //merging type 4
      *nways += 1;
      m1=mergecount(tl->l,tr->l,v,c,&tnwl);
      m2=mergecount(tl->r,tr->r,v,c,&tnwr);
      *nways += (tnwr*tnwl);
      return (m1 & m2);
   }
   else if(splitsonv(tl,tr,v)) {  //merging type 7
      *nways += 1;  //for this one
      m1=mergecount(tl->r,tr,v,c,&tnwr);
      m2=mergecount(tl,tr->l,v,c,&tnwl);
      *nways+= (tnwr+tnwl);
      if(!m1) cout << "doh7a" << endl;
      if(!m2) cout << "doh7b" << endl; 
      return (m1 & m2);
   }
   else if(splitsonv(tl,v) && isleaf(tr)) //merging type 1
   {
      *nways += 1; //for this one
      m1=mergecount(tl->r,tr,v,c,&tnwr);
      *nways += tnwr;
      if(!m1) cout << "doh1(m1)" << endl;
      return m1;
   }
   else if(splitsonv(tr,v) && isleaf(tl)) //merging type 2
   {
      *nways += 1; //for this one
      m2=mergecount(tl,tr->l,v,c,&tnwl);
      *nways += tnwl;
      if(!m2) cout << "doh2(m2)" << endl;
      return m2;
   }
   else if(!isleaf(tl) && !isleaf(tr) && splitsonv(tr,v)) { //merge type 6(i)
      *nways += 1; //for this one
      m1=mergecount(tl,tr->l,v,c,&tnwr);
      *nways += tnwr;
      if(!m1) cout << "doh6i(m1)" << endl;
      return m1;
   }
   else if(!isleaf(tl) && !isleaf(tr) && splitsonv(tl,v)) { //merge type 6(ii)
      *nways +=1 ; //for this one
      m2=mergecount(tl->r,tr,v,c,&tnwl);
      *nways += tnwl;
      if(!m2) cout << "doh6ii(m2)" << endl;
      return m2;
   }
   else if(!splitsonv(tl,v) && isleaf(tr)) { //merge type 5(i)
      *nways += 1; //for this one
      return true;
   }
   else if(!splitsonv(tr,v) && isleaf(tl)) { //merge type 5(ii)
      *nways += 1; //for this one
      return true;
   }
   else // default type aka type 8
   {
      *nways += 1; //for this one
      return true;
   }

   return false;
}



//--------------------------------------------------
// Functions to support calculation of Sobol indices for BART
// Based on Horiguchi, Pratola and Santner (2020).
//--------------------------------------------------
double probxnoti_termk(size_t i, size_t k, std::vector<std::vector<double> >& a, 
   std::vector<std::vector<double> >& b, std::vector<double>& minx, std::vector<double>& maxx)
{
  double prob=1.0;
  size_t p=minx.size();

  for(size_t j=0;j<p;j++)
    if(j!=i) {
      prob *= (b[j][k]-a[j][k])/(maxx[j]-minx[j]);
    }

  return prob;
}
double probxi_termk(size_t i, size_t k, std::vector<std::vector<double> >& a, 
   std::vector<std::vector<double> >& b, std::vector<double>& minx, std::vector<double>& maxx)
{
  double prob;

  prob=(b[i][k]-a[i][k])/(maxx[i]-minx[i]);

  return prob;
}
double probxij_termk(size_t i, size_t j, size_t k, std::vector<std::vector<double> >& a, 
   std::vector<std::vector<double> >& b, std::vector<double>& minx, std::vector<double>& maxx)
{
  double prob;

  prob=(b[i][k]-a[i][k])/(maxx[i]-minx[i])*(b[j][k]-a[j][k])/(maxx[j]-minx[j]);

  return prob;
}
//intersection product
double probxnoti_termkl(size_t i, size_t k, size_t l, std::vector<std::vector<double> >& a, 
   std::vector<std::vector<double> >& b, std::vector<double>& minx, std::vector<double>& maxx)
{
  double prob=1.0;
  double aa,bb;
  size_t p=minx.size();

  for(size_t j=0;j<p;j++)
    if(j!=i) {
      aa=std::max(a[j][k],a[j][l]);
      bb=std::min(b[j][k],b[j][l]);
      prob *= (std::max(bb-aa,0.0))/(maxx[j]-minx[j]);  //std::max needed because maybe they don't intersect
    } 

  return prob;
}
double probxi_termkl(size_t i, size_t k, size_t l, std::vector<std::vector<double> >& a, 
   std::vector<std::vector<double> >& b, std::vector<double>& minx, std::vector<double>& maxx)
{
  double aa,bb;
  double prob;

  aa=std::max(a[i][k],a[i][l]);
  bb=std::min(b[i][k],b[i][l]);
  prob=(std::max(bb-aa,0.0))/(maxx[i]-minx[i]); //std::max needed because maybe they don't intersect

  return prob;
}
double probxij_termkl(size_t i, size_t j, size_t k, size_t l, std::vector<std::vector<double> >& a, 
   std::vector<std::vector<double> >& b, std::vector<double>& minx, std::vector<double>& maxx)
{
  double aai,bbi,aaj,bbj;
  double prob;

  aai=std::max(a[i][k],a[i][l]);
  aaj=std::max(a[j][k],a[j][l]);
  bbi=std::min(b[i][k],b[i][l]);
  bbj=std::min(b[j][k],b[j][l]);

  prob=(std::max(bbi-aai,0.0))/(maxx[i]-minx[i])*(std::max(bbj-aaj,0.0))/(maxx[j]-minx[j]);
  //std::max needed because maybe they don't intersect

  return prob;
}
double probxall_termkl(size_t k, size_t l, std::vector<std::vector<double> >& a, 
   std::vector<std::vector<double> >& b, std::vector<double>& minx, std::vector<double>& maxx)
{
   double aa,bb;
   double prob=1.0;
   size_t p=minx.size();

   for(size_t j=0;j<p;j++)
   {
      aa=std::max(a[j][k],a[j][l]);
      bb=std::min(b[j][k],b[j][l]);

      prob *= (std::max(bb-aa,0.0))/(maxx[j]-minx[j]);
      //std::max needed because maybe they don't intersect
   }

  return prob;
}

//--------------------------------------------------
// This function only used for determining Pareto front/set.
// Based on Horiguchi, Santner, Sun and Pratola (2020).
// Note here the a0,b0,a1,b1 are of dimension rows=#nodes and columns=# variables (p)
// which is the opposite of all the Sobol functions above.
// Function returns measure of the intersection and the intersection rectangle in aout,bout.
//--------------------------------------------------
double probxall_termkl_rect(size_t k, size_t l, std::vector<std::vector<double> >& a0, 
   std::vector<std::vector<double> >& b0, std::vector<std::vector<double> >& a1, 
   std::vector<std::vector<double> >& b1, std::vector<double>& minx, std::vector<double>& maxx, std::vector<double>& aout, std::vector<double>& bout)
{
   double prob=1.0;
   size_t p=minx.size();

   for(size_t j=0;j<p;j++)
   {
      aout[j]=std::max(a0[k][j],a1[l][j]);
      bout[j]=std::min(b0[k][j],b1[l][j]);

      prob *= (std::max(bout[j]-aout[j],0.0))/(maxx[j]-minx[j]);
      //std::max needed because maybe they don't intersect
   }

  return prob;
}

//--------------------------------------------------
// Similar to probxall_termkl_rect() function except
// single hyperrectangles are passed in rather than a vector of them
// so no need for k,l indices.
// Currently only used in the ambrt::inflrect() method for reweighting
// predictions for influential observations.
// See Pratola, George and McCulloch (2020).
//--------------------------------------------------
double probxall_term_rect(std::vector<double>& a0, 
   std::vector<double>& b0, std::vector<double>& a1, 
   std::vector<double>& b1, std::vector<double>& minx, std::vector<double>& maxx, std::vector<double>& aout, std::vector<double>& bout)
{
   double prob=1.0;
   size_t p=minx.size();

   for(size_t j=0;j<p;j++)
   {
      aout[j]=std::max(a0[j],a1[j]);
      bout[j]=std::min(b0[j],b1[j]);

      prob *= (std::max(bout[j]-aout[j],0.0))/(maxx[j]-minx[j]);
      //std::max needed because maybe they don't intersect
   }

  return prob;
}
//--------------------------------------------------
// Similar to probxall_term_rect() function except
// we extract the union rectangle rather than the intersection.
// Currently only used in the ambrt::influnionrect() method for reweghting
// predictions for influential observations.
// See Pratola, George and McCulloch (2020).
// Note: this function assumes a0,b0 and a1,b1 are not disjoint.
//--------------------------------------------------
void unionxall_term_rect(std::vector<double>& a0, 
   std::vector<double>& b0, std::vector<double>& a1, 
   std::vector<double>& b1, std::vector<double>& minx, std::vector<double>& maxx, std::vector<double>& aout, std::vector<double>& bout)
{
   size_t p=minx.size();

   for(size_t j=0;j<p;j++)
   {
      aout[j]=std::min(a0[j],a1[j]);
      bout[j]=std::max(b0[j],b1[j]);
   }

  return;
}


// theta needs to be sorted in increasing order of its first coordinate, and in
// case of ties, increasing in its second coordinate.
std::vector<size_t> find_pareto_front(size_t start, size_t end, std::list<std::vector<double> > theta)
{
   std::vector<size_t> R,S,T;

   if(start==end) {
      R.push_back(start);
//      return R;
   }
   else {
      R=find_pareto_front(start,(size_t)((end-start)/2+start),theta);
      S=find_pareto_front((size_t)((end-start)/2+start)+1,end,theta);

      for(size_t s: S) 
         if(not_dominated(s,R,theta))
            T.push_back(s);

      R.insert(R.end(),T.begin(),T.end());  // R union T
   }
   // cout << "R=";
   // for(size_t i=0;i<R.size();i++) cout << " " << R[i];
   // cout << endl;
   return R;
}

// Is theta[index] not dominated by any theta's in R?
// We check this by checking each vector in R to see if it dominates
// the vector theta[index].  If none of the vectors in R dominate theta[index],
// then theta[index] is not dominated so return true.  Otherwise at least one vector
// in R dominate theta[index], so return false.
// Supports theta of any (finite) dimension.
bool not_dominated(size_t index, std::vector<size_t> R, std::list<std::vector<double> > theta)
{
   size_t d = theta.front().size() - 1;  // this assumes theta still has the index record as the last element of each vector.
   // note the -1's because we keep track of 1..sizeof(V) but the vectors are indexed by 0..sizeof(V)-1.
   std::list<std::vector<double> >::iterator it = std::next(theta.begin(),index-1);
   // for(size_t i=0;i<R.size();i++)
   for (size_t i: R) {
      std::list<std::vector<double> >::iterator itR = std::next(theta.begin(),i-1);
      // if R_ij <= v_j for all j then R_i dominates v, return false
//      if(theta[R[i]-1][0]<=theta[index-1][0] && theta[R[i]-1][1]<=theta[index-1][1])
      bool is_dominated = true;
      for (size_t j=0;j<d;j++) is_dominated && ((*itR).at(j) <= (*it).at(j));
      if(is_dominated) return false;
      // if((*itR).at(0) <= (*it).at(0) && (*itR).at(1) <= (*it).at(1))
      //    return false;
   }
   return true;
}


// combines a dd-dim output ensemble with a 1-dim output ensemble to make a (dd+1)-dim output ensemble
// assume a2/b2/theta2 is a 1-dim output ens
// but a1/b1/ltheta1 is a dd-dim output ens (where dd>=1)
// function's output is asol/bsol/thetasol
void combine_ensembles(size_t p, xinfo& xi, std::vector<std::unordered_map<double, size_t> >& ximaps, std::vector<double>& minx, std::vector<double>& maxx, 
   std::vector<std::vector<double> >& a1,   std::vector<std::vector<double> >& b1, std::list<std::vector<double> >& ltheta1, 
   std::vector<std::vector<double> >& a2,   std::vector<std::vector<double> >& b2, std::vector<double>& theta2, double fmean2, 
   std::vector<std::vector<double> >& asol, std::vector<std::vector<double> >& bsol, std::list<std::vector<double> >& thetasol, 
   bool to_clear_old_ens)
{

   std::vector<double> aout(p),bout(p);

   // 1. Assign rectangle indices in ens2 to the 3-d jagged tensor ie2
   // Time: O(theta2.size() * p * numcut)
   std::vector<std::vector<std::vector<size_t> > > ie2(p);  // (p,ncutsi+1,varying) jagged tensor -- to contain indices of ens2
   for (size_t w=0;w<p;w++) ie2[w].resize(xi[w].size());
   for (size_t j=0;j<theta2.size();j++) {
      for (size_t w=0;w<p;w++) {
         size_t lc = ximaps[w][a2[j][w]]; 
         size_t rc = ximaps[w][b2[j][w]]; 
         for (size_t c=lc;c<=rc;c++)  ie2[w][c].push_back(j);  
      }
   }

   // 2. Get rectangle intersections between ens1 and ens2
   // Time: O(theta1.size() * p * numcut), which is faster than O(theta1.size() * theta2.size())
   std::list<std::vector<double> >::iterator it1 = ltheta1.begin();
   for (size_t j=0;j<a1.size();j++) {
      // find all rectangles k in ambm2 that intersect with rectangle j from ambm1
      std::vector<size_t> valids;  // to contain all rectangles k in ambm2 that intersect with rectangle j from ambm1
      size_t v_size = 0;
      for (size_t w=0;w<p;w++) {  
         size_t lc = ximaps[w][a1[j][w]];  // ximaps[w][asol[j][w]]; 
         size_t rc = ximaps[w][b1[j][w]];  // ximaps[w][bsol[j][w]]; 
         size_t nvalidw = 0;  
         for (size_t c=lc;c<=rc;c++) nvalidw += ie2[w][c].size();
         std::vector<size_t> validsw(nvalidw);
         std::vector<size_t>::iterator it;
         nvalidw = 0;
         for (size_t c=lc;c<=rc;c++) {
            it=std::set_union(validsw.begin(), validsw.begin()+nvalidw, ie2[w][c].begin(), ie2[w][c].end(), validsw.begin());
            nvalidw = it-validsw.begin();  
         }
         if (w==0) { 
            valids.reserve(nvalidw);
            valids.assign(validsw.begin(), validsw.begin()+nvalidw);
            v_size = nvalidw;
         }
         else {  // intersect valids and validsw; store in valids
            it=std::set_intersection(valids.begin(), valids.begin()+v_size, validsw.begin(), validsw.begin()+nvalidw, valids.begin());
            v_size = it-valids.begin();
         }
      }

      for (size_t kind=0;kind<v_size;kind++) {  // v_size should be much smaller than theta2.size(), by a factor of 10^p or so
         size_t k = valids[kind];  // allows me to avoid resizing valids
         // if the rectangles defined by a1[j],b1[j] and a2[k],b2[k] intersect
         if(probxall_termkl_rect(j,k,a1,b1,a2,b2,minx,maxx,aout,bout)>0.0) {  // which should almost always be true, I believe
            asol.push_back(aout);
            bsol.push_back(bout);
            //The asol.size() is to record the index in the unsorted list so we can back
            //out the correct VHR for the theta's on the PF later on(line 497-498).
            //It would be cleaner if we had a list of vector, size_t tuple but this works as well.
            std::vector<double> th = *it1;  // ltheta1[j];
            if (th.size() > 1) th.pop_back();  // this is to remove the index record keeping in ltheta1
            th.push_back(theta2[k]+fmean2);
            th.push_back((double)(asol.size()));
            thetasol.push_back(th);
         }
      }
      it1++;
   }

   if (to_clear_old_ens) {  // clear the hyperrectangles for this realization
      a1.clear();
      b1.clear();
      ltheta1.clear();
      a2.clear();
      b2.clear();
      theta2.clear();
   }
}
