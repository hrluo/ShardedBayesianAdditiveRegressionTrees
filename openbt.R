##     openbt.R: R script wrapper functions for OpenBT.
##     Copyright (C) 2012-2019 Matthew T. Pratola
##
##     This file is part of OpenBT.
##
##     OpenBT is free software: you can redistribute it and/or modify
##     it under the terms of the GNU Affero General Public License as published by
##     the Free Software Foundation, either version 3 of the License, or
##     (at your option) any later version.
##
##     OpenBT is distributed in the hope that it will be useful,
##     but WITHOUT ANY WARRANTY; without even the implied warranty of
##     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##     GNU Affero General Public License for more details.
##
##     You should have received a copy of the GNU Affero General Public License
##     along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
##     Author contact information
##     Matthew T. Pratola: mpratola@gmail.com
##
##     Contributing Authors
##     Akira Horiguchi: horiguchi.6@osu.edu (mopareto)

# Load/Install required packages
required <- c("zip","data.table","Hmisc")
tbi <- required[!(required %in% installed.packages()[,"Package"])]
if(length(tbi)) {
   cat("***Installing OpenBT package dependencies***\n")
   install.packages(tbi,repos="https://cloud.r-project.org",quiet=TRUE)
}
library(zip,quietly=TRUE,warn.conflicts=FALSE)
library(data.table,quietly=TRUE,warn.conflicts=FALSE)
library(Hmisc,quietly=TRUE,warn.conflicts=FALSE) #for weighted mean/var/quantile used in repredict.*()


openbt = function(
x.train,
y.train,
#x.test=matrix(0.0,0,0),
ntree=NULL,
ntreeh=NULL,
ndpost=1000, nskip=100,
k=NULL,
power=2.0, base=.95,
tc=2,
sigmav=rep(1,length(y.train)),
fmean=mean(y.train),
overallsd = NULL,
overallnu= NULL,
chv = cor(x.train,method="spearman"),
pbd=.7,
pb=.5,
stepwpert=.1,
probchv=.1,
minnumbot=5,
printevery=100,
numcut=100,
xicuts=NULL,
nadapt=1000,
adaptevery=100,
summarystats=FALSE,
truncateds=NULL,
shardepth=-1,
shardpsplit=0.5,
randshard=FALSE,
shardcutslikedepth=FALSE,
model=NULL,
modelname="model"
)
{
#--------------------------------------------------
# model type definitions
modeltype=0 # undefined
MODEL_BT=1
MODEL_BINOMIAL=2
MODEL_POISSON=3
MODEL_BART=4
MODEL_HBART=5
MODEL_PROBIT=6
MODEL_MODIFIEDPROBIT=7
MODEL_MERCK_TRUNCATED=8
if(is.null(model))
{ 
   cat("Model type not specified.\n")
   cat("Available options are:\n")
   cat("model='bt'\n")
   cat("model='binomial'\n")
   cat("model='poisson'\n")
   cat("model='bart'\n")
   cat("model='hbart'\n")
   cat("model='probit'\n")
   cat("model='modifiedprobit'\n")
   cat("model='merck_truncated'\n")

   stop("missing model type.\n")
}
if(model=="bart")
{
   modeltype=MODEL_BART
   if(is.null(ntree)) ntree=200
   if(is.null(ntreeh)) ntreeh=1
   if(is.null(k)) k=2
   if(is.null(overallsd)) overallsd=sd(y.train)
   if(is.null(overallnu)) overallnu=10
   pbd=c(pbd,0.0)
}
if(model=="hbart")
{
   modeltype=MODEL_HBART
   if(is.null(ntree)) ntree=200
   if(is.null(ntreeh)) ntreeh=40
   if(is.null(k)) k=5
   if(is.null(overallsd)) overallsd=sd(y.train)
   if(is.null(overallnu)) overallnu=10
}

if(model=="probit")
{
   modeltype=MODEL_PROBIT
   if(is.null(ntree)) ntree=200
   if(is.null(ntreeh)) ntreeh=1
   if(is.null(k)) k=1
   if(is.null(overallsd)) overallsd=1
   if(is.null(overallnu)) overallnu=-1
   if(length(pbd)==1) pbd=c(pbd,0.0)
}
if(model=="modified-probit") 
{
   modeltype=MODEL_MODIFIEDPROBIT
   if(is.null(ntree)) ntree=200
   if(is.null(ntreeh)) ntreeh=40
   if(is.null(k)) k=1
   if(is.null(overallsd)) overallsd=1
   if(is.null(overallnu)) overallnu=-1
}
if(model=="merck_truncated")
{
   modeltype=MODEL_MERCK_TRUNCATED
   if(is.null(ntree)) ntree=200
   if(is.null(ntreeh)) ntreeh=1
   if(is.null(k)) k=2
   if(is.null(overallsd)) overallsd=sd(y.train)
   if(is.null(overallnu)) overallnu=10
   if(is.null(truncateds)) {
      miny=min(y.train)[1]
      truncateds=(y.train==miny)
   }
}
if(shardepth>-1)
{
   x.train=as.matrix(cbind(runif(nrow(x.train)),x.train))
   # update the change-of-variable proposal matrix.
   # note we never allow shard var to be transitioned to another var, or vice-versa.
   tempchv=matrix(0,ncol=ncol(chv)+1,nrow=nrow(chv)+1)
   tempchv[1,1]=1
   tempchv[2:(nrow(chv)+1),2:(ncol(chv)+1)]=chv
   chv=tempchv
}
#--------------------------------------------------
nd = ndpost
burn = nskip
m = ntree
mh = ntreeh
#--------------------------------------------------
#data
n = length(y.train)
p = ncol(x.train)
#np = nrow(x.test)
x = t(x.train)
#xp = t(x.test)
if(modeltype==MODEL_BART || modeltype==MODEL_HBART || modeltype==MODEL_MERCK_TRUNCATED)
{
   y.train=y.train-fmean
   fmean.out=paste(0.0)
}
if(modeltype==MODEL_PROBIT || modeltype==MODEL_MODIFIEDPROBIT)
{
   fmean.out=paste(qnorm(fmean))
   uniqy=sort(unique(y.train))
   if(length(uniqy)>2) stop("Invalid y.train: Probit requires dichotomous response coded 0/1")
   if(uniqy[1]!=0 || uniqy[2]!=1) stop("Invalid y.train: Probit requires dichotomous response coded 0/1")
}
#--------------------------------------------------
#cutpoints
if(!is.null(xicuts)) # use xicuts
{
   xi=xicuts
}
else # default to equal numcut per dimension
{
   xi=vector("list",p)
   minx=floor(apply(x,1,min))
   maxx=ceiling(apply(x,1,max))
   for(i in 1:p)
   {
      xinc=(maxx[i]-minx[i])/(numcut+1)
      xi[[i]]=(1:numcut)*xinc+minx[i]
   }
   if(shardcutslikedepth) {
      xinc=(maxx[1]-minx[1])/(2^(shardepth+1))
      xi[[1]]=1:(2^(shardepth+1)-1)*xinc+minx[1]
   }
}
#--------------------------------------------------
if(modeltype==MODEL_BART || modeltype==MODEL_HBART || modeltype==MODEL_MERCK_TRUNCATED)
{
   rgy = range(y.train)
}
if(modeltype==MODEL_PROBIT || modeltype==MODEL_MODIFIEDPROBIT)
{
   rgy = c(-2,2)
}

tau =  (rgy[2]-rgy[1])/(2*sqrt(m)*k)

#--------------------------------------------------
overalllambda = overallsd^2
#--------------------------------------------------
powerh=power
baseh=base
if(length(power)>1) {
   powerh=power[2]
   power=power[1]
}
if(length(base)>1) {
   baseh=base[2]
   base=base[1]
}
#--------------------------------------------------
pbdh=pbd
pbh=pb
if(length(pbd)>1) {
   pbdh=pbdh[2]
   pbd=pbd[1]
}
if(length(pb)>1) {
   pbh=pb[2]
   pb=pb[1]
}
#--------------------------------------------------
if(modeltype==MODEL_BART)
{
   cat("Model: Bayesian Additive Regression Trees model (BART)\n")
}
#--------------------------------------------------
if(modeltype==MODEL_HBART)
{
   cat("Model: Heteroscedastic Bayesian Additive Regression Trees model (HBART)\n")
}
#--------------------------------------------------
if(modeltype==MODEL_PROBIT)
{
   cat("Model: Dichotomous outcome model: Albert & Chib Probit (fixed)\n")
#   overallnu=-1
   if(ntreeh>1)
    stop("method probit requires ntreeh=1")
   if(pbdh>0.0)
    stop("method probit requires pbd[2]=0.0")
}
#--------------------------------------------------
if(modeltype==MODEL_MODIFIEDPROBIT)
{
   cat("Model: Dichotomous outcome model: Modified Albert & Chib Probit\n")
}
#--------------------------------------------------
if(modeltype==MODEL_MERCK_TRUNCATED)
{
   cat("Model: Truncated BART model\n")
}
#--------------------------------------------------
stepwperth=stepwpert
if(length(stepwpert)>1) {
   stepwperth=stepwpert[2]
   stepwpert=stepwpert[1]
}
#--------------------------------------------------
probchvh=probchv
if(length(probchv)>1) {
   probchvh=probchv[2]
   probchv=probchv[1]
}
#--------------------------------------------------
minnumboth=minnumbot
if(length(minnumbot)>1) {
   minnumboth=minnumbot[2]
   minnumbot=minnumbot[1]
}

#--------------------------------------------------
#write out config file
xroot="x"
yroot="y"
sroot="s"
chgvroot="chgv"
xiroot="xi"
folder=tempdir(check=TRUE)
if(!dir.exists(folder)) dir.create(folder)
tmpsubfolder=tempfile(tmpdir="")
tmpsubfolder=substr(tmpsubfolder,5,nchar(tmpsubfolder))
tmpsubfolder=paste("openbt",tmpsubfolder,sep="")
folder=paste(folder,"/",tmpsubfolder,sep="")
if(!dir.exists(folder)) dir.create(folder)
fout=file(paste(folder,"/config",sep=""),"w")
writeLines(c(paste(modeltype),xroot,yroot,fmean.out,paste(m),paste(mh),paste(nd),paste(burn),
            paste(nadapt),paste(adaptevery),paste(tau),paste(overalllambda),
            paste(overallnu),paste(base),paste(power),paste(baseh),paste(powerh),
            paste(tc),paste(sroot),paste(chgvroot),paste(pbd),paste(pb),
            paste(pbdh),paste(pbh),paste(stepwpert),paste(stepwperth),
            paste(probchv),paste(probchvh),paste(minnumbot),paste(minnumboth),
            paste(printevery),paste(xiroot),paste(shardepth),paste(shardpsplit),paste(as.integer(randshard)),paste(modelname),paste(summarystats)),fout)
close(fout)
# folder=paste(".",modelname,"/",sep="")
# system(paste("rm -rf ",folder,sep=""))
# system(paste("mkdir ",folder,sep=""))
# system(paste("cp config ",folder,"config",sep=""))


#--------------------------------------------------
#write out data subsets
nslv=tc-1
ylist=split(y.train,(seq(n)-1) %/% (n/nslv))
for(i in 1:nslv) write(ylist[[i]],file=paste(folder,"/",yroot,i,sep=""))
xlist=split(as.data.frame(x.train),(seq(n)-1) %/% (n/nslv))
for(i in 1:nslv) write(t(xlist[[i]]),file=paste(folder,"/",xroot,i,sep=""))
slist=split(sigmav,(seq(n)-1) %/% (n/nslv))
for(i in 1:nslv) write(slist[[i]],file=paste(folder,"/",sroot,i,sep=""))
chv[is.na(chv)]=0 # if a var as 0 levels it will have a cor of NA so we'll just set those to 0.
write(chv,file=paste(folder,"/",chgvroot,sep=""))
for(i in 1:p) write(xi[[i]],file=paste(folder,"/",xiroot,i,sep=""))
rm(chv)

if(modeltype==MODEL_MERCK_TRUNCATED)
{
   tlist=split(truncateds,(seq(n)-1) %/% (n/nslv))
   for(i in 1:nslv) {
      truncs=which(tlist[[i]]==TRUE)-1 #-1 for correct indexing in c/c++
      ftrun=file(paste(folder,"/","truncs",i,sep=""),"w")
      write(truncs,ftrun)
      close(ftrun)
   }
}
#--------------------------------------------------
#run program
cmdopt=100 #default to serial/OpenMP
runlocal=FALSE
cmd="openbtcli --conf"
if(Sys.which("openbtcli")[[1]]=="") # not installed in a global locaiton, so assume current directory
   runlocal=TRUE

if(runlocal && Sys.which("./openbtcli")[[1]]=="") # not installed globally, not installed locally.
   stop("OpenBT library installation is missing.  Please see the OpenBT install instructions at http://www.bitbucket.org/mpratola/openbt/wiki/Home.\n")

if(runlocal) cmd="./openbtcli --conf"

cmdopt=system(cmd)

if(cmdopt==101) # MPI
{
   cmd=paste("mpirun -np ",tc," openbtcli ",folder,sep="")
}

if(cmdopt==100)  # serial/OpenMP
{ 
   if(runlocal)
      cmd=paste("./openbtcli ",folder,sep="")
   else
      cmd=paste("openbtcli ",folder,sep="")
}
#
#cat(cmd)
system(cmd)
system(paste("rm -f ",folder,"/config",sep=""))
#system(paste("mv ",folder,"fit ",folder,modelname,".fit",sep=""))

res=list()

# Read in the influence metrics.
res$influence=NULL
for(i in 1:nslv)
   res$influence=rbind(res$influence,as.matrix(read.table(paste(folder,"/",modelname,".influence",i,sep=""))))
colnames(res$influence)=c("MeanCookD", "MaxCookD", "KLDiv")
res$modeltype=modeltype
res$model=model
res$xroot=xroot; res$yroot=yroot; res$m=m; res$mh=mh; res$nd=nd; res$burn=burn
res$nadapt=nadapt; res$adaptevery=adaptevery; res$tau=tau; res$overalllambda=overalllambda
res$overallnu=overallnu; res$k=k; res$base=base; res$power=power; res$baseh=baseh; res$powerh=powerh
res$tc=tc; res$sroot=sroot; res$chgvroot=chgvroot; res$pbd=pbd; res$pb=pb
res$pbdh=pbdh; res$pbh=pbh; res$stepwpert=stepwpert; res$stepwperth=stepwperth
res$probchv=probchv; res$probchvh=probchvh; res$minnumbot=minnumbot; res$minnumboth=minnumboth
res$printevery=printevery; res$xiroot=xiroot; res$minx=minx; res$maxx=maxx;
res$summarystats=summarystats; res$modelname=modelname; res$shardepth=shardepth; res$shardx=NULL
if(shardepth!=-1) res$shardx=x.train[,1]
class(xi)="OpenBT_cutinfo"
res$xicuts=xi
res$fmean=fmean
res$folder=folder
class(res)="OpenBT_posterior"

return(res)
}






predict.openbt <- function(
fit=NULL,
x.test=NULL,
tc=2,
fmean=fit$fmean,
q.lower=0.025,
q.upper=0.975
)
{

# model type definitions
MODEL_BT=1
MODEL_BINOMIAL=2
MODEL_POISSON=3
MODEL_BART=4
MODEL_HBART=5
MODEL_PROBIT=6
MODEL_MODIFIEDPROBIT=7
MODEL_MERCK_TRUNCATED=8

#--------------------------------------------------
# params
if(is.null(fit)) stop("No fitted model specified!\n")
if(is.null(x.test)) stop("No prediction points specified!\n")
if(fit$shardepth>-1)
{
   x.test=as.matrix(cbind(runif(nrow(x.test)),x.test))
}

nslv=tc
x.test=as.matrix(x.test)
p=ncol(x.test)
n=nrow(x.test)
xproot="xp"

if(fit$modeltype==MODEL_BART || fit$modeltype==MODEL_HBART || fit$modeltype==MODEL_MERCK_TRUNCATED)
{
   fmean.out=paste(fmean)
}
if(fit$modeltype==MODEL_PROBIT || fit$modeltype==MODEL_MODIFIEDPROBIT)
{
   fmean.out=paste(qnorm(fmean))
}

#--------------------------------------------------
#write out config file
fout=file(paste(fit$folder,"/config.pred",sep=""),"w")
writeLines(c(fit$modelname,fit$modeltype,fit$xiroot,xproot,
            paste(fit$nd),paste(fit$m),
            paste(fit$mh),paste(p),paste(tc),
            fmean.out), fout)
close(fout)

#--------------------------------------------------
#write out data subsets
#folder=paste(".",fit$modelname,"/",sep="")
xlist=split(as.data.frame(x.test),(seq(n)-1) %/% (n/nslv))
for(i in 1:nslv) write(t(xlist[[i]]),file=paste(fit$folder,"/",xproot,i-1,sep=""))
for(i in 1:p) write(fit$xicuts[[i]],file=paste(fit$folder,"/",fit$xiroot,i,sep=""))


#--------------------------------------------------
#run prediction program
cmdopt=100 #default to serial/OpenMP
runlocal=FALSE
cmd="openbtcli --conf"
if(Sys.which("openbtcli")[[1]]=="") # not installed in a global locaiton, so assume current directory
   runlocal=TRUE

if(runlocal) cmd="./openbtcli --conf"

cmdopt=system(cmd)

if(cmdopt==101) # MPI
{
   cmd=paste("mpirun -np ",tc," openbtpred ",fit$folder,sep="")
}

if(cmdopt==100)  # serial/OpenMP
{ 
   if(runlocal)
      cmd=paste("./openbtpred ",fit$folder,sep="")
   else
      cmd=paste("openbtpred ",fit$folder,sep="")
}

#cmd=paste("mpirun -np ",tc," openbtpred",sep="")
#cat(cmd)
system(cmd)
system(paste("rm -f ",fit$folder,"/config.pred",sep=""))


#--------------------------------------------------
#format and return
res=list()

# Old, original code for reading in the posterior predictive draws.
# res$mdraws=read.table(paste(fit$folder,"/",fit$modelname,".mdraws",0,sep=""))
# res$sdraws=read.table(paste(fit$folder,"/",fit$modelname,".sdraws",0,sep=""))
# for(i in 2:nslv)
# {
#    res$mdraws=cbind(res$mdraws,read.table(paste(fit$folder,"/",fit$modelname,".mdraws",i-1,sep="")))
#    res$sdraws=cbind(res$sdraws,read.table(paste(fit$folder,"/",fit$modelname,".sdraws",i-1,sep="")))
# }
# res$mdraws=as.matrix(res$mdraws)
# res$sdraws=as.matrix(res$sdraws)

# Faster using data.table's fread than the built-in read.table.
# However, it does strangely introduce some small rounding error on the order of 8.9e-16.
fnames=list.files(fit$folder,pattern=paste(fit$modelname,".mdraws*",sep=""),full.names=TRUE)
res$mdraws=as.matrix(do.call(cbind,sapply(fnames,data.table::fread)))
fnames=list.files(fit$folder,pattern=paste(fit$modelname,".sdraws*",sep=""),full.names=TRUE)
res$sdraws=as.matrix(do.call(cbind,sapply(fnames,data.table::fread)))

system(paste("rm -f ",fit$folder,"/",fit$modelname,".mdraws*",sep=""))
system(paste("rm -f ",fit$folder,"/",fit$modelname,".sdraws*",sep=""))

res$mmean=apply(res$mdraws,2,mean)
res$smean=apply(res$sdraws,2,mean)
res$msd=apply(res$mdraws,2,sd)
res$ssd=apply(res$sdraws,2,sd)
res$m.5=apply(res$mdraws,2,quantile,0.5)
res$m.lower=apply(res$mdraws,2,quantile,q.lower)
res$m.upper=apply(res$mdraws,2,quantile,q.upper)
res$s.5=apply(res$sdraws,2,quantile,0.5)
res$s.lower=apply(res$sdraws,2,quantile,q.lower)
res$s.upper=apply(res$sdraws,2,quantile,q.upper)
res$pdraws=NULL
res$pmean=NULL
res$psd=NULL
res$p.5=NULL
res$p.lower=NULL
res$p.upper=NULL

if(fit$modeltype==MODEL_PROBIT || fit$modeltype==MODEL_MODIFIEDPROBIT)
{
   res$pdraws=read.table(paste(fit$folder,"/",fit$modelname,".pdraws",0,sep=""))
   for(i in 2:nslv)
   {
      res$pdraws=cbind(res$pdraws,read.table(paste(fit$folder,"/",fit$modelname,".pdraws",i-1,sep="")))
   }

   system(paste("rm -f ",fit$folder,"/",fit$modelname,".pdraws*",sep=""))

   res$pdraws=as.matrix(res$pdraws)
   res$pmean=apply(res$pdraws,2,mean)
   res$psd=apply(res$pdraws,2,sd)
   res$p.5=apply(res$pdraws,2,quantile,0.5)
   res$p.lower=apply(res$pdraws,2,quantile,q.lower)
   res$p.upper=apply(res$pdraws,2,quantile,q.upper)
}

res$q.lower=q.lower
res$q.upper=q.upper
res$x.test=x.test
res$modeltype=fit$modeltype

class(res)="OpenBT_predict"

return(res)
}




vartivity.openbt = function(
fit=NULL,
q.lower=0.025,
q.upper=0.975
)
{

#--------------------------------------------------
# params
if(is.null(fit)) stop("No fitted model specified!\n")
p=length(fit$xicuts)
m=fit$m
mh=fit$mh
nd=fit$nd
modelname=fit$modelname


#--------------------------------------------------
#write out config file
fout=file(paste(fit$folder,"/config.vartivity",sep=""),"w")
writeLines(c(fit$modelname,
            paste(nd),paste(m),
            paste(mh),paste(p)) ,fout)
close(fout)


#--------------------------------------------------
#run vartivity program  -- it's not actually parallel so no call to mpirun.
runlocal=FALSE
if(Sys.which("openbtcli")[[1]]=="") # not installed in a global locaiton, so assume current directory
   runlocal=TRUE

if(runlocal)
   cmd=paste("./openbtvartivity ",fit$folder,sep="")
else
   cmd=paste("openbtvartivity ",fit$folder,sep="")

#cmd=paste("./openbtvartivity",sep="")
system(cmd)
system(paste("rm -f ",fit$folder,"/config.vartivity",sep=""))


#--------------------------------------------------
#read in result
res=list()
res$vdraws=read.table(paste(fit$folder,"/",fit$modelname,".vdraws",sep=""))
res$vdrawsh=read.table(paste(fit$folder,"/",fit$modelname,".vdrawsh",sep=""))
res$vdraws=as.matrix(res$vdraws)
res$vdrawsh=as.matrix(res$vdrawsh)

# normalize counts
colnorm=apply(res$vdraws,1,sum)
ix=which(colnorm>0)
res$vdraws[ix,]=res$vdraws[ix,]/colnorm[ix]
colnorm=apply(res$vdrawsh,1,sum)
ix=which(colnorm>0)
res$vdrawsh[ix,]=res$vdrawsh[ix,]/colnorm[ix]

res$mvdraws=apply(res$vdraws,2,mean)
res$mvdrawsh=apply(res$vdrawsh,2,mean)
res$vdraws.sd=apply(res$vdraws,2,sd)
res$vdrawsh.sd=apply(res$vdrawsh,2,sd)
res$vdraws.5=apply(res$vdraws,2,quantile,0.5)
res$vdrawsh.5=apply(res$vdrawsh,2,quantile,0.5)
res$vdraws.lower=apply(res$vdraws,2,quantile,q.lower)
res$vdraws.upper=apply(res$vdraws,2,quantile,q.upper)
res$vdrawsh.lower=apply(res$vdrawsh,2,quantile,q.lower)
res$vdrawsh.upper=apply(res$vdrawsh,2,quantile,q.upper)
res$q.lower=q.lower
res$q.upper=q.upper
res$modeltype=fit$modeltype

class(res)="OpenBT_vartivity"

return(res)
}


sobol.openbt = function(
fit=NULL,
q.lower=0.025,
q.upper=0.975,
tc=2
)
{

#--------------------------------------------------
# params
if(is.null(fit)) stop("No fitted model specified!\n")
p=length(fit$xicuts)
m=fit$m
mh=fit$mh
nd=fit$nd
modelname=fit$modelname


#--------------------------------------------------
#write out config file
fout=file(paste(fit$folder,"/config.sobol",sep=""),"w")
writeLines(c(fit$modelname,fit$xiroot,
            paste(nd),paste(m),
            paste(mh),paste(p),paste(fit$minx),
            paste(fit$maxx),paste(tc)) ,fout)
close(fout)


#--------------------------------------------------
#run Sobol program
cmdopt=100 #default to serial/OpenMP
runlocal=FALSE
cmd="openbtcli --conf"
if(Sys.which("openbtcli")[[1]]=="") # not installed in a global locaiton, so assume current directory
   runlocal=TRUE

if(runlocal) cmd="./openbtcli --conf"

cmdopt=system(cmd)

if(cmdopt==101) # MPI
{
   cmd=paste("mpirun -np ",tc," openbtsobol ",fit$folder,sep="")
}

if(cmdopt==100)  # serial/OpenMP
{ 
   if(runlocal)
      cmd=paste("./openbtsobol ",fit$folder,sep="")
   else
      cmd=paste("openbtsobol ",fit$folder,sep="")
}

system(cmd)
system(paste("rm -f ",fit$folder,"/config.sobol",sep=""))


#--------------------------------------------------
#read in result
res=list()
draws=read.table(paste(fit$folder,"/",fit$modelname,".sobol",0,sep=""))
for(i in 2:tc)
    draws=rbind(draws,read.table(paste(fit$folder,"/",fit$modelname,".sobol",i-1,sep="")))
draws=as.matrix(draws)


labs=gsub("\\s+",",",apply(combn(1:p,2),2,function(zz) Reduce(paste,zz)))
res$vidraws=draws[,1:p]
res$vijdraws=draws[,(p+1):(p+p*(p-1)/2)]
res$tvidraws=draws[,(ncol(draws)-p):(ncol(draws)-1)]
res$vdraws=draws[,ncol(draws)]
res$sidraws=res$vidraws/res$vdraws
res$sijdraws=res$vijdraws/res$vdraws
res$tsidraws=res$tvidraws/res$vdraws
res$vidraws=as.matrix(res$vidraws)
colnames(res$vidraws)=paste("V",1:p,sep="")
res$vijdraws=as.matrix(res$vijdraws)
colnames(res$vijdraws)=paste("V",labs,sep="")
res$tvidraws=as.matrix(res$tvidraws)
colnames(res$tvidraws)=paste("TV",1:p,sep="")
res$vdraws=as.matrix(res$vdraws)
colnames(res$vdraws)="V"
res$sidraws=as.matrix(res$sidraws)
colnames(res$sidraws)=paste("S",1:p,sep="")
res$sijdraws=as.matrix(res$sijdraws)
colnames(res$sijdraws)=paste("S",labs,sep="")
res$tsidraws=as.matrix(res$tsidraws)
colnames(res$tsidraws)=paste("TS",1:p,sep="")
rm(draws)

# summaries
res$msi=apply(res$sidraws,2,mean)
res$msi.sd=apply(res$sidraws,2,sd)
res$si.5=apply(res$sidraws,2,quantile,0.5)
res$si.lower=apply(res$sidraws,2,quantile,q.lower)
res$si.upper=apply(res$sidraws,2,quantile,q.upper)
names(res$msi)=paste("S",1:p,sep="")
names(res$msi.sd)=paste("S",1:p,sep="")
names(res$si.5)=paste("S",1:p,sep="")
names(res$si.lower)=paste("S",1:p,sep="")
names(res$si.upper)=paste("S",1:p,sep="")

res$msij=apply(res$sijdraws,2,mean)
res$sij.sd=apply(res$sijdraws,2,sd)
res$sij.5=apply(res$sijdraws,2,quantile,0.5)
res$sij.lower=apply(res$sijdraws,2,quantile,q.lower)
res$sij.upper=apply(res$sijdraws,2,quantile,q.upper)
names(res$msij)=paste("S",labs,sep="")
names(res$sij.sd)=paste("S",labs,sep="")
names(res$sij.5)=paste("S",labs,sep="")
names(res$sij.lower)=paste("S",labs,sep="")
names(res$sij.upper)=paste("S",labs,sep="")

res$mtsi=apply(res$tsidraws,2,mean)
res$tsi.sd=apply(res$tsidraws,2,sd)
res$tsi.5=apply(res$tsidraws,2,quantile,0.5)
res$tsi.lower=apply(res$tsidraws,2,quantile,q.lower)
res$tsi.upper=apply(res$tsidraws,2,quantile,q.upper)
names(res$mtsi)=paste("TS",1:p,sep="")
names(res$tsi.sd)=paste("TS",1:p,sep="")
names(res$tsi.5)=paste("TS",1:p,sep="")
names(res$tsi.lower)=paste("TS",1:p,sep="")
names(res$tsi.upper)=paste("TS",1:p,sep="")


res$q.lower=q.lower
res$q.upper=q.upper
res$modeltype=fit$modeltype

class(res)="OpenBT_sobol"

return(res)
}



# Pareto Front Multiobjective Optimization using 2 fitted BART models
mopareto.openbt = function(
fit1=NULL,
fit2=NULL,
fit3=NULL,
q.lower=0.025,
q.upper=0.975,
tc=2
)
{

#--------------------------------------------------
# params
if(is.null(fit1) || is.null(fit2)) stop("No fitted models specified!\n")
#if(fit1$xicuts != fit2$xicuts) stop("Models not compatible\n")
if(fit1$nd != fit2$nd) stop("Models have different number of posterior samples\n")
p=length(fit1$xicuts)
d=2+(!is.null(fit3))
print(paste0("d=",d))
m1=fit1$m
m2=fit2$m
mh1=fit1$mh
mh2=fit2$mh
m3=0
mh3=0
if(!is.null(fit3)) {
   if(fit3$nd != fit2$nd) stop("Models have different number of posterior samples\n")
   m3=fit3$m
   mh3=fit3$mh
}
# if(is.null(fit3)) {
#    fit3=list()
#    fit3$modelname="null"
#    fit3$folder="null"
#    fit3$fmean=0.0
# }

nd=fit1$nd
modelname=fit1$modelname


#--------------------------------------------------
#write out config file
fout=file(paste(fit1$folder,"/config.mopareto",sep=""),"w")
if(!is.null(fit3)) {
writeLines(c(fit1$modelname,fit2$modelname,fit3$modelname,fit1$xiroot,
            fit2$folder,fit3$folder,
            paste(nd),paste(m1),
            paste(mh1),paste(m2),paste(mh2),
            paste(m3),paste(mh3),
            paste(p),paste(fit1$minx),
            paste(fit1$maxx),paste(fit1$fmean),paste(fit2$fmean),
            paste(fit3$fmean),paste(tc)) ,fout)
}
if(is.null(fit3)) {
writeLines(c(fit1$modelname,fit2$modelname,paste("null"),fit1$xiroot,
            fit2$folder,paste("null"),
            paste(nd),paste(m1),
            paste(mh1),paste(m2),paste(mh2),
            paste(m3),paste(mh3),
            paste(p),paste(fit1$minx),
            paste(fit1$maxx),paste(fit1$fmean),paste(fit2$fmean),
            paste(0.0),paste(tc)) ,fout)  
}
close(fout)


#--------------------------------------------------
#run Pareto Front program
cmdopt=100 #default to serial/OpenMP
runlocal=FALSE
cmd="openbtcli --conf"
if(Sys.which("openbtcli")[[1]]=="") # not installed in a global locaiton, so assume current directory
   runlocal=TRUE

if(runlocal) cmd="./openbtcli --conf"

cmdopt=system(cmd)

if(cmdopt==101) # MPI
{
   cmd=paste("mpirun -np ",tc," openbtmopareto ",fit1$folder,sep="")
}

if(cmdopt==100)  # serial/OpenMP
{ 
   if(runlocal)
      cmd=paste("./openbtmopareto ",fit1$folder,sep="")
   else
      cmd=paste("openbtmopareto ",fit1$folder,sep="")
}

system(cmd)
system(paste("rm -f ",fit1$folder,"/config.mopareto",sep=""))


#--------------------------------------------------
#read in result
res=list()
ii=1
u=0  # to modify temp indexing if fit3 exists
for(i in 1:tc) 
{
   con=file(paste(fit1$folder,"/",fit1$modelname,".mopareto",i-1,sep=""))
   open(con)
   s=readLines(con,1)
   while(length(s)>0) {
      temp=as.numeric(unlist(strsplit(s," ")))
      k=as.integer(temp[1])
      theta=matrix(0,ncol=k,nrow=d)
      theta[1,]=temp[2:(2+k-1)]
      theta[2,]=temp[(2+k):(2+2*k-1)]
      if(!is.null(fit3)) { 
         theta[3,]=temp[(2+2*k):(2+3*k-1)] 
         u=k
      }
      a=matrix(0,nrow=p,ncol=k)
      for(i in 1:p) a[i,]=temp[(2+(2+i-1)*k):(2+(2+i)*k-1)+u]
      b=matrix(0,nrow=p,ncol=k)
      for(i in 1:p) b[i,]=temp[(2+(2+p+i-1)*k):(2+(2+p+i)*k-1)+u]
      entry=list()
      entry[["theta"]]=theta
      entry[["a"]]=a
      entry[["b"]]=b
      res[[ii]]=entry
      ii=ii+1
      s=readLines(con,1)
   }
   close(con)
}


class(res)="OpenBT_mopareto"

return(res)
}



# Reweight predictions using output of influence.openbt().
# For method IS_GLOBAL: infl (influence obect) and pred (prediction object to be reweighted)
#                       Also, idx may be specified to apply weights only to a subset of observations.
# For method IS_LOCAL_*: pass infl (influence object), xpred (training x's from fit) and fit (fitted OpenBT object)
repredict.openbt<-function(pred=NULL,infl=NULL,fit=NULL,tc=2)
{
   if(is.null(pred)) stop("Model prediction object required.\n")
   if(is.null(infl)) stop("Model influence object required.\n")
   if(infl$method=="is_global" && class(pred)!="OpenBT_predict") stop("Model prediction object not recognized.\n")
   if(class(infl)!="OpenBT_influence") stop("Model influence object not recognized.\n")
   if(class(pred)!="OpenBT_predict") stop("Model pred object not recognized.\n")
   if(infl$method!="is_global" && infl$method!="is_local_int" && infl$method!="is_local_uint" && infl$method!="is_local_union" && infl$method!="is_local_l1" && infl$method!="is_local_l2") stop("Cannot reweight predictions with influence method ",infl$method,".\n")

   dropid=infl$infl.obs
   ninfl=length(dropid)
   res=list()
   cat("Applying weights\n")

   if(infl$method=="is_global")
   {
      res$mdraws=pred$mdraws
      res$wmat=apply(infl$w,1,prod)
      #re-weight and return the predictions
      # for(k in 1:ninfl)
         # for(i in 1:fit$nd) {
         #    # res$mdraws[i,]=res$mdraws[i,]*infl$w[i,k]
         #    res$mdraws[i,]=res$mdraws[i,]*res$wmat[i]
         # }

      # wbar=apply(infl$w,2,mean)#mean(infl$w)
      wbar=mean(res$wmat)#mean(infl$w)
      # res$mdraws=res$mdraws/wbar
      # res$wmat=infl$w
      res$wbar=wbar
      res$mmean=rep(0,nrow(pred$x.test))
      res$msd=rep(0,nrow(pred$x.test))
      res$m.5=rep(0,nrow(pred$x.test))
      res$m.lower=rep(0,nrow(pred$x.test))
      res$m.upper=rep(0,nrow(pred$x.test))
      for(i in 1:nrow(pred$x.test)) {
         res$mmean[i]=wtd.mean(res$mdraws[,i],res$wmat,normwt=TRUE)
         res$msd[i]=sqrt(wtd.var(res$mdraws[,i],res$wmat,normwt=TRUE))
         res$m.5[i]=wtd.quantile(res$mdraws[,i],res$wmat,probs=c(0.5),normwt=TRUE)
         res$m.lower[i]=wtd.quantile(res$mdraws[,i],res$wmat,probs=c(pred$q.lower),normwt=TRUE)
         res$m.upper[i]=wtd.quantile(res$mdraws[,i],res$wmat,probs=c(pred$q.upper),normwt=TRUE)

      }
   }

   if(infl$method=="is_local_int" || infl$method=="is_local_uint" || infl$method=="is_local_union" || infl$method=="is_local_l1")
   {

      if(is.null(fit)) stop("Method is_local_* requires fitted OpenBT_posterior in object fit\n")
      if(length(fit$xicuts)!=ncol(pred$x.test)) stop("xpred's are not same dimension as trained model\n")

      in_rect<-function(xin,a,b,p)
      {
         for(i in 1:length(xin)) {
            if(xin[i]<a[i] || xin[i]>b[i]) {
               return(FALSE)
            }
         }
         return(TRUE)
      }

      # get predictions
      p=length(fit$xicuts)
      res$mdraws=pred$mdraws

      # now for each posterior draw, identify all predictions in the hyperrectangle of influence
      wmat=matrix(1,nrow=fit$nd,ncol=nrow(pred$x.test)) # our final weight matrix

      for(i in 1:fit$nd) {
         for(j in 1:nrow(pred$x.test)) { 
            for(k in 1:ninfl) {
               if(in_rect(pred$x.test[j,],infl$hyperrects[[k]]$a[i,],infl$hyperrects[[k]]$b[i,],p)) {
                  # if(wmat[i,j]==1) 
                     # wmat[i,j]=infl$w[i,k,drop=FALSE]
                  # else 
                  wmat[i,j]=wmat[i,j]*infl$w[i,k,drop=FALSE]
               }
            }
         }
      }

      wbar=apply(wmat,2,mean)
      # res$mdraws=res$mdraws*wmat
      # res$mdraws=t(t(res$mdraws)/wbar)
      res$wmat=wmat
      res$wbar=wbar
      res$mmean=rep(0,nrow(pred$x.test))
      res$msd=rep(0,nrow(pred$x.test))
      res$m.5=rep(0,nrow(pred$x.test))
      res$m.lower=rep(0,nrow(pred$x.test))
      res$m.upper=rep(0,nrow(pred$x.test))
      for(i in 1:nrow(pred$x.test)) {
         res$mmean[i]=wtd.mean(res$mdraws[,i],res$wmat[,i],normwt=TRUE)
         res$msd[i]=sqrt(wtd.var(res$mdraws[,i],res$wmat[,i],normwt=TRUE))
         res$m.5[i]=wtd.quantile(res$mdraws[,i],res$wmat[,i],probs=c(0.5),normwt=TRUE)
         res$m.lower[i]=wtd.quantile(res$mdraws[,i],res$wmat[,i],probs=c(pred$q.lower),normwt=TRUE)
         res$m.upper[i]=wtd.quantile(res$mdraws[,i],res$wmat[,i],probs=c(pred$q.upper),normwt=TRUE)

      }

   }

   if(infl$method=="is_local_l2")
   {
      if(is.null(fit)) stop("Method is_local_* requires fitted OpenBT_posterior in object fit\n")

      res$mdraws=pred$mdraws
      wmat=matrix(1/fit$nd,nrow=fit$nd,ncol=nrow(pred$x.test))

      for(j in 1:ninfl) {
         dvec=rep(0,nrow(pred$x.test))
         for(k in 1:nrow(pred$x.test)) 
            dvec[k]=dist(rbind(pred$x.test[k,],x[dropid[[j]],]))
         idx=which(dvec<0.2)
         for(i in 1:fit$nd)
            for(k in idx)
               if(wmat[i,k]==1/fit$nd) wmat[i,k]=infl$w[i,j,drop=FALSE]
               else wmat[i,k]=wmat[i,k]*infl$w[i,j,drop=FALSE]
      }

      wbar=apply(wmat,2,mean)
      # res$mdraws=res$mdraws*wmat
      # res$mdraws=t(t(res$mdraws)/wbar)
      res$wmat=wmat
      res$wbar=wbar
      res$mmean=rep(0,nrow(pred$x.test))
      res$msd=rep(0,nrow(pred$x.test))
      res$m.5=rep(0,nrow(pred$x.test))
      res$m.lower=rep(0,nrow(pred$x.test))
      res$m.upper=rep(0,nrow(pred$x.test))
      for(i in 1:nrow(pred$x.test)) {
         res$mmean[i]=wtd.mean(res$mdraws[,i],res$wmat[,i],normwt=TRUE)
         res$msd[i]=sqrt(wtd.var(res$mdraws[,i],res$wmat[,i],normwt=TRUE))
         res$m.5[i]=wtd.quantile(res$mdraws[,i],res$wmat[,i],probs=c(0.5),normwt=TRUE)
         res$m.lower[i]=wtd.quantile(res$mdraws[,i],res$wmat[,i],probs=c(pred$q.lower),normwt=TRUE)
         res$m.upper[i]=wtd.quantile(res$mdraws[,i],res$wmat[,i],probs=c(pred$q.upper),normwt=TRUE)

      }
   }



   res$sdraws=pred$sdraws
   # res$mmean=apply(res$mdraws,2,mean)
   res$smean=pred$smean
   # res$msd=apply(res$mdraws,2,sd)
   res$ssd=pred$ssd
   # res$m.5=apply(res$mdraws,2,quantile,0.5)
   # res$m.lower=apply(res$mdraws,2,quantile,pred$q.lower)
   # res$m.upper=apply(res$mdraws,2,quantile,pred$q.upper)
   res$s.5=pred$s.5
   res$s.lower=pred$s.lower
   res$s.upper=pred$s.upper
   res$pdraws=pred$pdraws
   res$pmean=pred$pmean
   res$psd=pred$psd
   res$p.5=pred$p.5
   res$p.lower=pred$p.lower
   res$p.upper=pred$p.upper
   res$q.lower=pred$q.lower
   res$q.upper=pred$q.upper
   res$modeltype=pred$modeltype

   class(res)="OpenBT_predict"
   return(res)
}


# Calculate various metrics of influence for a regression tree model.
influence.openbt<-function(x.infl,y.infl,fit=NULL,infl.obs=NULL,dvec=NULL,tc=2,method="is_local_int")
{
   if(is.null(fit)) stop("Model fit object required.\n")
   if(class(fit)!="OpenBT_posterior") stop("Model fit object not recognized.\n")
   if(is.null(infl.obs)) stop("No hold-out drop id specified for reweighting (infl.obs=NULL).\n")
   if(min(infl.obs)<1) stop("Invalid drop id infl.obs=",infl.obs,".\n")
   if(max(infl.obs)>nrow(x.infl)) stop("Invalid drop id infl.obs=",infl.obs,".\n")
   if(method=="is_local_l1" && is.null(dvec)) stop("Invalid dvec for method is_local_l1.\n")

   nd=fit$nd
   # np=nrow(x.infl)
   ninfl=length(infl.obs)
   w=matrix(0,nrow=nd,ncol=ninfl)
   wsum=rep(1,ninfl)

   pp=predict.openbt(fit=fit,x.test=x.infl,tc=tc)

   if(method=="is_global") {
      # w is an nd by np matrix where the j_th column contains the weights for each
      # nd posterior samples if (x_j,y_j) were removed from the dataset.
      for(kk in 1:ninfl)
         for(i in 1:nd) w[i,kk]=(2*pi)^(1/2)*pp$sdraws[i,infl.obs[kk]]*exp(1/(2*pp$sdraws[i,infl.obs[kk]]^2)*(y.infl[infl.obs[kk]]-pp$mdraws[i,infl.obs[kk]])^2)
#      w=apply(w,1,prod) # iid BART likelihood => we just combine the different terms productwise.
#      wsum=sum(w)
   } 
   # end IS_GLOBAL

   if(method=="is_local_int" || method=="is_local_union" || method=="is_local_uint" || method=="is_local_l1")
   {
      p=length(fit$xicuts)
      m=fit$m
      mh=fit$mh
      nd=fit$nd
      modelname=fit$modelname
      hrects.int=list()
      hrects.uint=list()
      hrects.union=list()
      hrects.l1=list()
#      w=matrix(0,nrow=nd,ncol=ninfl)
#      wsum=rep(1,ninfl)

      for(kk in 1:ninfl) {
         xd=x.infl[infl.obs[kk],]
         if(!is.vector(xd)) stop("xd is not a px1 vector in x.infl.\n")

         #--------------------------------------------------
         #write out config file
         fout=file(paste(fit$folder,"/config.influence",sep=""),"w")
         writeLines(c(fit$modelname,fit$xiroot,
                     paste(nd),paste(m),
                     paste(mh),paste(p),paste(xd),paste(fit$minx),
                     paste(fit$maxx),paste(tc)) ,fout)
         close(fout)


         #--------------------------------------------------
         #run influential hyperrectangle program
         cmdopt=100 #default to serial/OpenMP
         runlocal=FALSE
         cmd="openbtcli --conf"
         if(Sys.which("openbtcli")[[1]]=="") # not installed in a global locaiton, so assume current directory
            runlocal=TRUE

         if(runlocal) cmd="./openbtcli --conf"

         cmdopt=system(cmd)

         if(cmdopt==101) # MPI
         {
            cmd=paste("mpirun -np ",tc," openbtinfl ",fit$folder,sep="")
         }

         if(cmdopt==100)  # serial/OpenMP
         { 
            if(runlocal)
               cmd=paste("./openbtinfl ",fit$folder,sep="")
            else
               cmd=paste("openbtinfl ",fit$folder,sep="")
         }

         system(cmd)
         system(paste("rm -f ",fit$folder,"/config.influence",sep=""))

         #--------------------------------------------------
         #read in result
         ii=1
         a=matrix(0,nrow=nd,ncol=p)
         b=matrix(0,nrow=nd,ncol=p)
         au=matrix(0,nrow=nd,ncol=p)
         bu=matrix(0,nrow=nd,ncol=p)
         for(i in 1:tc) 
         {
            con=file(paste(fit$folder,"/",fit$modelname,".influence",i-1,sep=""))
            open(con)
            s=readLines(con,1)
            while(length(s)>0) {
               temp=as.numeric(unlist(strsplit(s," ")))
               a[ii,]=temp[1:p]
               b[ii,]=temp[(p+1):(2*p)]
               au[ii,]=temp[(2*p+1):(3*p)]
               bu[ii,]=temp[(3*p+1):(4*p)]
               ii=ii+1
               s=readLines(con,1)
            }
            close(con)
         }

         system(paste("rm -f ",fit$folder,"/",fit$modelname,".influence*",sep=""))

         hrects.int[[kk]]=list()
         hrects.uint[[kk]]=list()
         hrects.union[[kk]]=list()
         hrects.l1[[kk]]=list()
         hrects.int[[kk]][["a"]]=a
         hrects.int[[kk]][["b"]]=b
         hrects.uint[[kk]][["a"]]=matrix(apply(a,2,min),nrow=nd,ncol=p,byrow=TRUE)
         hrects.uint[[kk]][["b"]]=matrix(apply(b,2,max),nrow=nd,ncol=p,byrow=TRUE)
         hrects.union[[kk]][["a"]]=au
         hrects.union[[kk]][["b"]]=bu
         hrects.l1[[kk]][["a"]]=matrix(xd-dvec,nrow=nd,ncol=p,byrow=TRUE)
         hrects.l1[[kk]][["b"]]=matrix(xd+dvec,nrow=nd,ncol=p,byrow=TRUE)

         # w is an nd by 1 matrix where containing the weights for each
         # nd posterior samples if (x[infl.obs,],y[infl.obs]) were removed from the dataset,
         # where the weight would be applied to any predictions in the corresponding hyperrect
         # for this draw.  Otherwise, the weight is just 1, giving usual posterior average -- this
         # is handled in the reweight() function.
         for(i in 1:nd) w[i,kk]=(2*pi)^(1/2)*pp$sdraws[i,infl.obs[kk]]*exp(1/(2*pp$sdraws[i,infl.obs[kk]]^2)*(y.infl[infl.obs[kk]]-pp$mdraws[i,infl.obs[kk]])^2)
#         wsum[kk]=sum(w[,kk])
#         w[,kk]=w[,kk]/sum(w[,kk])

         # Perhaps a simple indicator of which observation has a lot of influence:
         # infl.max=apply(w,2,max)
         # infl.mean=apply(w,2,mean)
         # infl.ids=infl.obs
      } 
      # end for kk in 1:length(infl.obs) loop
   } 
   #end is_local_*
   for(i in 1:ninfl) wsum[i]=sum(w[,i])

   res=list()
   res$nd=nd
   # res$np=np
   res$x.infl=x.infl
   res$y.infl=y.infl
   res$w=w
   res$wsum=wsum
   # res$infl.max=infl.max
   # res$infl.mean=infl.mean
   res$infl.obs=infl.obs
   res$hyperrects=NULL
   if(method=="is_local_int") {
      res$hyperrects=hrects.int
   }
   if(method=="is_local_uint") {
      res$hyperrects=hrects.uint
   }
   if(method=="is_local_union") {
      res$hyperrects=hrects.union   
   }
   if(method=="is_local_l1") {
      res$hyperrects=hrects.l1   
   }
   res$method=method

   class(res)="OpenBT_influence"

   return(res)
}


summary.OpenBT_influence<-function(infl)
{
   cat("No summary method for object.\n")
}

print.OpenBT_influence<-function(infl)
{
   cat("OpenBT Influence\n")
   cat("metric: ",infl$method,"\n")
   cat(infl$nd, " posterior samples.\n")
   cat(infl$np, " observations.\n")

   if(infl$method=="is_global") {
      idx=sort(infl$infl,decreasing=TRUE,index.return=TRUE)$ix
      cat("\nTop 5 maximum weights: ", infl$infl[idx][1:5],"\n")
      cat("\nTop 5 influential observations by weights: \n")
      for(i in 1:5) {
         cat("Input ",idx[i]," max.influence=",infl$infl[idx[i]],
            " y=",infl$y.infl[idx[i]], " x=",infl$x.infl[idx[i],][1],"\n")
      }

      tab=sort(table(infl$infl.ids),decreasing=TRUE)
      cat("\nTop 5 influential observations by frequency: \n")
      print(tab[1:5])
   }
}

plot.OpenBT_influence<-function(infl)
{
   tab.ids=table(infl$infl.ids)

   if(infl$method=="is_global") {
      par(mfrow=c(1,2))
      plot(1:infl$np,infl$infl,xlab="Observation Index",ylab="max.influence",type="h",lwd=2)
      title(paste("metric: ",infl$method,sep=""))
      plot(tab.ids,xlab="Observation Index",ylab="Frequency")
      title(paste("metric: ",infl$method,sep=""))
   }

   if(infl$method=="is_local_int" || infl$method=="is_local_union")
   {
      # plot hyperrectangles
      plot(0,0,xlim=c(0,1),ylim=c(0,1),type="n",xlab="x1",ylab="x2")
      for(i in 1:nrow(infl$hyperrects$a)) rect(infl$hyperrects$a[i,1],infl$hyperrects$a[i,2],infl$hyperrects$b[i,1],infl$hyperrects$b[i,2],col=gray(0.5,alpha=0.1),border=NA)
   }
}



# Computer Model Calibration a BART emulator and BART discrepancy.
calibrate.openbt<-function(xc,yc,xf,yf,xdims=NULL,caldims=NULL,
                           calprior="normal",calpriora=NULL,calpriorb=NULL,
                           ntreec=NULL,ntreef=NULL,ndpost=1000,
                           nskip=100,kc=NULL,kf=NULL,power=2,base=0.95,tc=2,
                           sigmac=rep(1,length(yc)),sigmaf=rep(1,length(yf)),
                           fmeanc=0.0,fmeanf=0.0,overallsdc=NULL,
                           overallnuc=NULL,overallsdf=NULL,overallnuf=NULL,
                           chv=cor(xc,method="spearman"),pbd=0.7,pb=0.5,
                           stepwpert=0.1,probchv=0.1,minnumbot=5,printevery=100,
                           numcut=100,xicuts=NULL,nadapt=1000,adaptevery=100,
                           model="bart",modelname="model",summarystats=FALSE)
{
   if(is.null(xdims)) stop("xdims must be specified.\n")
   if(is.null(caldims)) stop("caldims must be specified.\n")
   numx=length(xdims)
   numcal=length(caldims)
   if(ncol(xc)!=numx+numcal) stop("xc not compatible with xdims+caldims.\n")
   if(ncol(xf)!=numx) stop("xf not compatible with xdims.\n")
   if(is.null(calprior)) stop("Invalid calibration parameter prior type specified.\n")
   if(is.null(calpriora)) stop("Invalid calibration parameter prior: calpriora.\n")
   if(is.null(calpriorb)) stop("Invalid calibration parameter prior: calpriorb.\n")

   if(length(calprior)==1) calprior=rep(calprior,numcal)
   if(length(calpriora)==1) calpriora=rep(calpriora,numcal)
   if(length(calpriorb)==1) calpriorb=rep(calpriorb,numcal)

   if(length(calprior)!=numcal || length(calpriora)!=numcal || length(calpriorb)!=numcal)
      stop("Invalid calibration parameter prior: does not match number of calibration parameters.\n")

   calpriortype=rep(NULL,numcal)
   for(i in 1:numcal) {
      if(calprior[i]=="normal") calpriortype[i]=1
      if(calprior[i]=="uniform") calpriortype[i]=0
      if(is.null(calpriortype[i])) stop("Invalid calibration parameter prior type specified.\n")
      if(calpriortype[i]==1 && calpriorb[i]<=0) stop("Invalid variance specified in calibration parameter prior.\n")
   }

   ntreehc=NULL
   ntreehf=NULL
   #--------------------------------------------------
   # model type definitions
   modeltype=rep(0,2) # undefined
   MODEL_BT=1
   MODEL_BINOMIAL=2
   MODEL_POISSON=3
   MODEL_BART=4
   MODEL_HBART=5
   MODEL_PROBIT=6
   MODEL_MODIFIEDPROBIT=7
   MODEL_MERCK_TRUNCATED=8


   if(length(model)==1) model=rep(model,2)
   if( (model[1]!="bart" && model[1]!="hbart") || (model[2]!="bart" && model[2]!="hbart") )
   { 
      cat("Model type not specified.\n")
      cat("Available options are:\n")
      cat("model='bart'\n")
      cat("model='hbart'\n")

      stop("missing model type.\n")
   }
   if(model[1]=="bart")
   {
      modeltype[1]=MODEL_BART
      if(is.null(ntreec)) ntreec=200
      if(is.null(ntreehc)) ntreehc=1
      if(is.null(kc)) kc=2
      if(is.null(overallsdc)) overallsdc=sd(yc)
      if(is.null(overallnuc)) overallnuc=10
      pbdc=c(pbd[1],0.0)
   }
   if(model[1]=="hbart")
   {
      modeltype=MODEL_HBART
      if(is.null(ntreec)) ntreec=200
      if(is.null(ntreehc)) ntreehc=40
      if(is.null(kc)) kc=5
      if(is.null(overallsdc)) overallsdc=sd(yc)
      if(is.null(overallnuc)) overallnuc=10
      pbdc=pbd
   }
   if(model[2]=="bart")
   {
      modeltype[2]=MODEL_BART
      if(is.null(ntreef)) ntreef=200
      if(is.null(ntreehf)) ntreehf=1
      if(is.null(kf)) kf=2
      if(is.null(overallsdf)) overallsdf=sd(yf)
      if(is.null(overallnuf)) overallnuf=10
      pbdf=c(pbd[1],0.0)
   }
   if(model[2]=="hbart")
   {
      modeltype[2]=MODEL_HBART
      if(is.null(ntreef)) ntreef=200
      if(is.null(ntreehf)) ntreehf=40
      if(is.null(kf)) kf=5
      if(is.null(overallsdf)) overallsdf=sd(yf)
      if(is.null(overallnuf)) overallnuf=10
      pbdf=pbd
   }


   #--------------------------------------------------
   nd = ndpost
   burn = nskip
   mc = ntreec
   mhc = ntreehc
   mf = ntreef
   mhf = ntreehf
   #--------------------------------------------------
   # simulator data
   nc = length(yc)
   p = ncol(xc)
   pdelta = ncol(xf)
   xpred = matrix(NA,nrow=nf,ncol=p)
   for(i in 1:numcal) {
      if(calpriortype[i])
         xpred[,caldims[i]]=calpriora[i]
      else
         xpred[,caldims[i]]=calpriora[i]+(calpriorb[i]-calpriora[i])/2.0
   }
   for(i in 1:pdelta) {
      xpred[,xdims[i]]=xf[,i]
   }
#   xpred = t(xpred)
#   xc = t(xc)
   nf = length(yf)
#   xf = t(xf)
   # if(modeltype==MODEL_BART || modeltype==MODEL_HBART)
   # {
   yc=yc-fmeanc
   yf=yf-fmeanf
   fmeanc.out=paste(fmeanc)
   fmeanf.out=paste(fmeanf)
   # }
   # if(modeltype==MODEL_PROBIT || modeltype==MODEL_MODIFIEDPROBIT)
   # {
   #    fmean.out=paste(qnorm(fmean))
   #    uniqy=sort(unique(y.train))
   #    if(length(uniqy)>2) stop("Invalid y.train: Probit requires dichotomous response coded 0/1")
   #    if(uniqy[1]!=0 || uniqy[2]!=1) stop("Invalid y.train: Probit requires dichotomous response coded 0/1")
   # }
   #--------------------------------------------------
   #cutpoints
   if(!is.null(xicuts)) # use xicuts
   {
      xic=xicuts
   }
   else # default to equal numcut per dimension
   {
      xic=vector("list",p)
      minxc=floor(apply(xc,2,min))
      maxxc=ceiling(apply(xc,2,max))
      for(i in 1:p)
      {
         xinc=(maxxc[i]-minxc[i])/(numcut+1)
         xic[[i]]=(1:numcut)*xinc+minxc[i]
      }
   }
   xif=vector("list",numx)
   for(i in 1:numx)
      xif[[i]]=xic[[xdims[i]]]

   #--------------------------------------------------
   if(modeltype[1]==MODEL_BART || modeltype[1]==MODEL_HBART)
   {
      rgc = range(yc)
   }
   if(modeltype[2]==MODEL_BART || modeltype[2]==MODEL_HBART)
   {
      rgf = range(yf)
   }
   # if(modeltype==MODEL_PROBIT || modeltype==MODEL_MODIFIEDPROBIT)
   # {
   #    rgy = c(-2,2)
   # }
   tauc =  (rgc[2]-rgc[1])/(2*sqrt(mc)*kc)
   tauf =  (rgf[2]-rgf[1])/(2*sqrt(mf)*kf)

   #--------------------------------------------------
   overalllambdac = overallsdc^2
   overalllambdaf = overallsdf^2
   #--------------------------------------------------
   powerh=power
   baseh=base
   if(length(power)>1) {
      powerh=power[2]
      power=power[1]
   }
   if(length(base)>1) {
      baseh=base[2]
      base=base[1]
   }
   #--------------------------------------------------
   pbdch=pbdc
   pbdfh=pbdf
   pbh=pb
   if(length(pbdc)>1) {
      pbdch=pbdc[2]
      pbdc=pbdc[1]
   }
   if(length(pbdf)>1) {
      pbdfh=pbdf[2]
      pbdf=pbdf[1]
   }
   if(length(pb)>1) {
      pbh=pb[2]
      pb=pb[1]
   }
   cat("Model: Bayesian Regression Tree based Computer Model Calibration\n")
   #--------------------------------------------------
   if(modeltype[1]==MODEL_BART)
   {
      cat("\tEmulator: BART\n")
   }
   #--------------------------------------------------
   if(modeltype[1]==MODEL_HBART)
   {
      cat("\tEmulator: Heteroscedastic BART\n")
   }
   #--------------------------------------------------
   if(modeltype[2]==MODEL_BART)
   {
      cat("\tDiscrepancy: BART\n")
   }
   #--------------------------------------------------
   if(modeltype[2]==MODEL_HBART)
   {
      cat("\tDiscrepancy: Heteroscedastic BART\n")
   }

   #--------------------------------------------------
   stepwperth=stepwpert
   if(length(stepwpert)>1) {
      stepwperth=stepwpert[2]
      stepwpert=stepwpert[1]
   }
   #--------------------------------------------------
   probchvh=probchv
   if(length(probchv)>1) {
      probchvh=probchv[2]
      probchv=probchv[1]
   }
   #--------------------------------------------------
   minnumboth=minnumbot
   if(length(minnumbot)>1) {
      minnumboth=minnumbot[2]
      minnumbot=minnumbot[1]
   }

   #--------------------------------------------------
   #write out config file
   xcroot="xc"
   ycroot="yc"
   scroot="sc"
   chgvcroot="chgvc"
   xicroot="xic"
   xfroot="xf"
   yfroot="yf"
   sfroot="sf"
   xproot="xpred"
   chgvfroot="chgvf"
   xifroot="xif"
   folder=tempdir(check=TRUE)
   if(!dir.exists(folder)) dir.create(folder)
   tmpsubfolder=tempfile(tmpdir="")
   tmpsubfolder=substr(tmpsubfolder,5,nchar(tmpsubfolder))
   tmpsubfolder=paste("openbt",tmpsubfolder,sep="")
   folder=paste(folder,"/",tmpsubfolder,sep="")
   if(!dir.exists(folder)) dir.create(folder)
   fout=file(paste(folder,"/config.calibrate",sep=""),"w")
   writeLines(c(paste(modeltype),xcroot,ycroot,xfroot,yfroot,xproot,fmeanc.out,fmeanf.out,
               paste(mc),paste(mhc),paste(mf),paste(mhf),paste(nd),paste(burn),
               paste(nadapt),paste(adaptevery),paste(tauc),paste(overalllambdac),
               paste(overallnuc),paste(tauf),paste(overalllambdaf),paste(overallnuf),
               paste(base),paste(power),paste(baseh),paste(powerh),
               paste(tc),paste(scroot),paste(sfroot),paste(chgvcroot),paste(chgvfroot),
               paste(pb),paste(pbdc),paste(pbdch),
               paste(pbdf),paste(pbdfh),paste(stepwpert),paste(stepwperth),
               paste(probchv),paste(probchvh),paste(minnumbot),paste(minnumboth),
               paste(printevery),paste(xicroot),paste(xifroot),paste(modelname),
               paste(as.integer(summarystats)),paste(numcal),paste(caldims-1),paste(calpriortype),
               paste(calpriora),paste(calpriorb)),fout)
   close(fout)


   #--------------------------------------------------
   #write out data subsets
   nslv=tc-1
   # yc data
   yclist=split(yc,(seq(nc)-1) %/% (nc/nslv))
   for(i in 1:nslv) write(yclist[[i]],file=paste(folder,"/",ycroot,i,sep=""))
   xclist=split(as.data.frame(xc),(seq(nc)-1) %/% (nc/nslv))
   for(i in 1:nslv) write(t(xclist[[i]]),file=paste(folder,"/",xcroot,i,sep=""))
#stop("done")

   sclist=split(sigmac,(seq(nc)-1) %/% (nc/nslv))
   for(i in 1:nslv) write(sclist[[i]],file=paste(folder,"/",scroot,i,sep=""))
   chv[is.na(chv)]=0 # if a var has 0 levels it will have a cor of NA so we'll just set those to 0.
   write(chv,file=paste(folder,"/",chgvcroot,sep=""))
   for(i in 1:p) write(xic[[i]],file=paste(folder,"/",xicroot,i,sep=""))
   # yf data
   yflist=split(yf,(seq(nf)-1) %/% (nf/nslv))
   for(i in 1:nslv) write(yflist[[i]],file=paste(folder,"/",yfroot,i,sep=""))
   xflist=split(as.data.frame(xf),(seq(nf)-1) %/% (nf/nslv))
   for(i in 1:nslv) write(t(xflist[[i]]),file=paste(folder,"/",xfroot,i,sep=""))
   sflist=split(sigmaf,(seq(nf)-1) %/% (nf/nslv))
   for(i in 1:nslv) write(sflist[[i]],file=paste(folder,"/",sfroot,i,sep=""))
   write(chv[xdims,xdims],file=paste(folder,"/",chgvfroot,sep=""))
   for(i in 1:pdelta) write(xif[[i]],file=paste(folder,"/",xifroot,i,sep=""))
   rm(chv)
   # xpred
   xplist=split(as.data.frame(xpred),(seq(nf)-1) %/% (nf/nslv))
   for(i in 1:nslv) write(t(xplist[[i]]),file=paste(folder,"/",xproot,i,sep=""))

   #--------------------------------------------------
   #run program
   cmdopt=100 #default to serial/OpenMP
   runlocal=FALSE
   cmd="openbtcalibrate --conf"
   if(Sys.which("openbtcalibrate")[[1]]=="") # not installed in a global locaiton, so assume current directory
      runlocal=TRUE

   if(runlocal) cmd="./openbtcalibrate --conf"

   cmdopt=system(cmd)

   if(cmdopt==101) # MPI
   {
      cmd=paste("mpirun -np ",tc," openbtcalibrate ",folder,sep="")
   }

   if(cmdopt==100)  # serial/OpenMP
   { 
      if(runlocal)
         cmd=paste("./openbtcalibrate ",folder,sep="")
      else
         cmd=paste("openbtcalibrate ",folder,sep="")
   }
   #
   system(cmd)
   system(paste("rm -f ",folder,"/config.calibrate",sep=""))


   #--------------------------------------------------
   # load in the theta posterior draws
   thetas=scan(paste(folder,"/model.eta.theta.fit",sep=""),quiet=TRUE)
   thetas=matrix(thetas,ncol=numcal,byrow=T)


   res=list()
   res$modeltype=modeltype
   res$model=model
   res$xcroot=xcroot; res$ycroot=ycroot; 
   #res$xfroot=xfroot; res$yfroot=yfroot; 
   res$xproot=xproot;
   res$mc=mc; res$mhc=mhc; 
   #res$mf=mf; res$mhf=mhf; 
   res$p=p; 
   #res$pdelta=pdelta;
   res$nd=nd; res$burn=burn; res$nadapt=nadapt; res$adaptevery=adaptevery; res$tauc=tauc; res$overalllambdac=overalllambdac
   res$overallnuc=overallnuc; 
   #res$tauf=tauf; res$overalllambdaf=overalllambdaf; res$overallnuf=overallnuf; 
   res$base=base; res$power=power; res$baseh=baseh; res$powerh=powerh
   res$k=numx+numcal;
   res$tc=tc; res$scroot=scroot; 
   #res$sfroot=sfroot; 
   res$chgvcroot=chgvcroot;
   #res$chgvfroot=chgvfroot; 
   res$pb=pb; res$pbdc=pbdc; res$pbdch=pbdch; 
   #res$pbdf=pbdf; res$pbdfh=pbdfh;
   res$stepwpert=stepwpert; res$stepwperth=stepwperth;
   res$probchv=probchv; res$probchvh=probchvh; res$minnumbot=minnumbot; res$minnumboth=minnumboth
   res$printevery=printevery; res$xicroot=xicroot; 
   #res$xifroot=xifroot; 
   res$minxc=minxc; res$maxxc=maxxc;
   res$xdims=xdims; res$numcal=numcal; res$caldims=caldims; res$calpriortype=calpriortype; res$calpriora=calpriora;
   res$calpriorb=calpriorb; res$summarystats=summarystats; res$modelname=paste(modelname,".eta",sep="")
   class(xic)="OpenBT_cutinfo"
   class(xif)="OpenBT_cutinfo"
   res$xiccuts=xic
#   res$xifcuts=xif
   res$fmeanc=fmeanc.out;
#   res$fmeanf=fmeanf.out;
   res$theta=thetas
   res$folder=folder

   res$delta$modeltype=modeltype[2]
   res$delta$model=model
   res$delta$xroot=xfroot
   res$delta$yroot=yfroot
   res$delta$xproot=xproot
   res$delta$k=numx
   res$delta$m=mf
   res$delta$mh=mhf
   res$delta$p=pdelta
   res$delta$nd=nd
   res$delta$burn=burn
   res$delta$nadapt=nadapt
   res$delta$adaptevery=adaptevery
   res$delta$tau=tauf
   res$delta$overalllambda=overalllambdaf
   res$delta$overallnu=overallnuf
   res$delta$base=base
   res$delta$power=power
   res$delta$baseh=baseh
   res$delta$powerh=powerh
   res$delta$chgvroot=chgvfroot
   res$delta$pb=pb
   res$delta$pbd=pbdf
   res$delta$pbdh=pbdfh
   res$delta$stepwpert=stepwpert
   res$delta$stepwperth=stepwperth
   res$delta$probchv=probchv
   res$delta$probchvh=probchvh
   res$delta$minnumbot=minnumbot
   res$delta$minnumboth=minnumboth
   res$delta$printevery=printevery
   res$delta$xiroot=xifroot
   res$delta$summarystats=summarystats
   res$delta$modelname=paste(modelname,".delta",sep="")
   res$delta$xicuts=xif
   res$delta$fmean=fmeanf.out
   res$delta$folder=folder
   class(res$delta)="OpenBT_posterior"


   class(res)="OpenBT_calibrate_posterior"

   return(res)
}


emulate.openbt = function(
fit=NULL,
x.test=NULL,
tc=2,
fmeanc=fit$fmeanc,
fmeanf=fit$delta$fmean,
q.lower=0.025,
q.upper=0.975
)
{

   # model type definitions
   MODEL_BT=1
   MODEL_BINOMIAL=2
   MODEL_POISSON=3
   MODEL_BART=4
   MODEL_HBART=5
   MODEL_PROBIT=6
   MODEL_MODIFIEDPROBIT=7
   MODEL_MERCK_TRUNCATED=8

   #--------------------------------------------------
   # params
   if(is.null(fit)) stop("No fitted model specified!\n")
   if(is.null(x.test)) stop("No emulation points specified!\n")
   if(class(fit)!="OpenBT_calibrate_posterior") stop("Invalid object passed to emulate.openbt()\n")

   xdims=fit$xdims
   caldims=fit$caldims
   nslv=tc
   x.test=as.matrix(x.test)
   pdelta=ncol(x.test)
   if(pdelta!=fit$delta$p) stop("x.test has incorrect dimensions.\n")
   n=nrow(x.test)
   p=fit$p
   x.eta.test=matrix(0,nrow=n,ncol=p)
   x.eta.test[,xdims]=x.test

   # Get discrepancy posterior first
   cat("Drawing from discrepancy posterior...\n")
   fit.delta=predict.openbt(fit$delta,x.test=x.test,tc=tc,fmean=fmeanf)

   # Get eta posterior next
   cat("Drawing from emulator posterior...\n")
   xeroot="xe"
   xdroot="xd"

   if( (fit$modeltype[1]==MODEL_BART || fit$modeltype[1]==MODEL_HBART) && (fit$modeltype[2]==MODEL_BART || fit$modeltype[2]==MODEL_HBART) )
   {
      fmeanc.out=paste(fmeanc)
      fmeanf.out=paste(fmeanf)
   }
   else stop("Invalid model passed to emulate.openbt()\n")
   # if(fit$modeltype==MODEL_PROBIT || fit$modeltype==MODEL_MODIFIEDPROBIT)
   # {
   #    fmean.out=paste(qnorm(fmean))
   # }


   #--------------------------------------------------
   #write out config file
   fout=file(paste(fit$folder,"/config.emulate",sep=""),"w")
   writeLines(c(paste(fit$modelname),paste(fit$modeltype),fit$xicroot,
               xeroot,paste(fit$nd),paste(fit$mc),
               paste(fit$mhc),paste(p),paste(pdelta),
               paste(fit$numcal),paste(caldims-1),paste(tc),
               fmeanc.out), fout)
   close(fout)



   #--------------------------------------------------
   #write out data subsets
   #folder=paste(".",fit$modelname,"/",sep="")
   xelist=split(as.data.frame(x.eta.test),(seq(n)-1) %/% (n/nslv))
   for(i in 1:nslv) write(t(xelist[[i]]),file=paste(fit$folder,"/",xeroot,i-1,sep=""))
   xdlist=split(as.data.frame(x.test),(seq(n)-1) %/% (n/nslv))
   for(i in 1:nslv) write(t(xdlist[[i]]),file=paste(fit$folder,"/",xdroot,i-1,sep=""))
   for(i in 1:p) write(fit$xiccuts[[i]],file=paste(fit$folder,"/",fit$xicroot,i,sep=""))
   for(i in 1:pdelta) write(fit$xifcuts[[i]],file=paste(fit$folder,"/",fit$xifroot,i,sep=""))


   #--------------------------------------------------
   #run prediction program
   cmdopt=100 #default to serial/OpenMP
   runlocal=FALSE
   cmd="openbtcli --conf"
   if(Sys.which("openbtcli")[[1]]=="") # not installed in a global location, so assume current directory
      runlocal=TRUE

   if(runlocal) cmd="./openbtcli --conf"

   cmdopt=system(cmd)

   if(cmdopt==101) # MPI
   {
      cmd=paste("mpirun -np ",tc," openbtemulate ",fit$folder,sep="")
   }

   if(cmdopt==100)  # serial/OpenMP
   { 
      if(runlocal)
         cmd=paste("./openbtemulate ",fit$folder,sep="")
      else
         cmd=paste("openbtemulate ",fit$folder,sep="")
   }

   #cmd=paste("mpirun -np ",tc," openbtpred",sep="")
   #cat(cmd)
   system(cmd)
   system(paste("rm -f ",fit$folder,"/config.emulate",sep=""))


#--------------------------------------------------
#format and return
res=list()


# Read in eta posterior
fnames=list.files(fit$folder,pattern=paste(fit$modelname,".mdraws*",sep=""),full.names=TRUE)
res$mdraws=as.matrix(do.call(cbind,sapply(fnames,data.table::fread)))
fnames=list.files(fit$folder,pattern=paste(fit$modelname,".sdraws*",sep=""),full.names=TRUE)
res$sdraws=as.matrix(do.call(cbind,sapply(fnames,data.table::fread)))

system(paste("rm -f ",fit$folder,"/",fit$modelname,".mdraws*",sep=""))
system(paste("rm -f ",fit$folder,"/",fit$modelname,".sdraws*",sep=""))

res$mmean=apply(res$mdraws,2,mean)
res$smean=apply(res$sdraws,2,mean)
res$msd=apply(res$mdraws,2,sd)
res$ssd=apply(res$sdraws,2,sd)
res$m.5=apply(res$mdraws,2,quantile,0.5)
res$m.lower=apply(res$mdraws,2,quantile,q.lower)
res$m.upper=apply(res$mdraws,2,quantile,q.upper)
res$s.5=apply(res$sdraws,2,quantile,0.5)
res$s.lower=apply(res$sdraws,2,quantile,q.lower)
res$s.upper=apply(res$sdraws,2,quantile,q.upper)
res$pdraws=NULL
res$pmean=NULL
res$psd=NULL
res$p.5=NULL
res$p.lower=NULL
res$p.upper=NULL
res$zdraws=res$mdraws+fit.delta$mdraws
res$zmean=apply(res$zdraws,2,mean)
res$zsd=apply(res$zdraws,2,sd)
res$z.5=apply(res$zdraws,2,quantile,0.5)
res$z.lower=apply(res$zdraws,2,quantile,q.lower)
res$z.upper=apply(res$zdraws,2,quantile,q.upper)

if(fit$modeltype==MODEL_PROBIT || fit$modeltype==MODEL_MODIFIEDPROBIT)
{
   res$pdraws=read.table(paste(fit$folder,"/",fit$modelname,".pdraws",0,sep=""))
   for(i in 2:nslv)
   {
      res$pdraws=cbind(res$pdraws,read.table(paste(fit$folder,"/",fit$modelname,".pdraws",i-1,sep="")))
   }

   system(paste("rm -f ",fit$folder,"/",fit$modelname,".pdraws*",sep=""))

   res$pdraws=as.matrix(res$pdraws)
   res$pmean=apply(res$pdraws,2,mean)
   res$psd=apply(res$pdraws,2,sd)
   res$p.5=apply(res$pdraws,2,quantile,0.5)
   res$p.lower=apply(res$pdraws,2,quantile,q.lower)
   res$p.upper=apply(res$pdraws,2,quantile,q.upper)
}

res$q.lower=q.lower
res$q.upper=q.upper
res$modeltype=fit$modeltype

res$delta=fit.delta

class(res)="OpenBT_emulate"

return(res)
}



# Scan the trees in the posterior to extract tree properties
# Returns the mean trees as a list of lists in object mt
# and the variance trees as a list of lists in object st.
# The format is mt[[i]][[j]] is the jth posterior tree from the ith posterior
# sum-of-trees (ensemble) sample.
# The tree is encoded in 4 vectors - the node ids, the node variables,
# the node cutpoints and the node thetas.
openbt.scanpost<-function(post)
{
   fp=file(paste(post$folder,"/",post$modelname,".fit",sep=""),open="r")
   if(scan(fp,what=integer(),nmax=1,quiet=TRUE) != post$nd) stop("Error scanning posterior\n")
   if(scan(fp,what=integer(),nmax=1,quiet=TRUE) != post$m) stop("Error scanning posterior\n")
   if(scan(fp,what=integer(),nmax=1,quiet=TRUE) != post$mh) stop("Error scanning posterior\n")
   if(scan(fp,what=integer(),nmax=1,quiet=TRUE) != post$nd*post$m) stop("Error scanning posterior\n")

   # scan mean trees
   numnodes=scan(fp,what=integer(),nmax=post$nd*post$m,quiet=TRUE)
   lenvec=scan(fp,what=integer(),nmax=1,quiet=TRUE)
   ids=scan(fp,what=integer(),nmax=lenvec,quiet=TRUE)
   lenvec=scan(fp,what=integer(),nmax=1,quiet=TRUE)
   vars=scan(fp,what=integer(),nmax=lenvec,quiet=TRUE)
   lenvec=scan(fp,what=integer(),nmax=1,quiet=TRUE)
   cuts=scan(fp,what=integer(),nmax=lenvec,quiet=TRUE)
   lenvec=scan(fp,what=integer(),nmax=1,quiet=TRUE)
   thetas=scan(fp,what=double(),nmax=lenvec,quiet=TRUE)

   # scan var trees
   if(scan(fp,what=integer(),nmax=1,quiet=TRUE) != post$nd*post$mh) stop("Error scanning posterior\n")
   snumnodes=scan(fp,what=integer(),nmax=post$nd*post$mh,quiet=TRUE)
   lenvec=scan(fp,what=integer(),nmax=1,quiet=TRUE)
   sids=scan(fp,what=integer(),nmax=lenvec,quiet=TRUE)
   lenvec=scan(fp,what=integer(),nmax=1,quiet=TRUE)
   svars=scan(fp,what=integer(),nmax=lenvec,quiet=TRUE)
   lenvec=scan(fp,what=integer(),nmax=1,quiet=TRUE)
   scuts=scan(fp,what=integer(),nmax=lenvec,quiet=TRUE)
   lenvec=scan(fp,what=integer(),nmax=1,quiet=TRUE)
   sthetas=scan(fp,what=double(),nmax=lenvec,quiet=TRUE)

   close(fp)

   # Now rearrange things into lists of lists so its easier to manipulate
   mt=list()
   ndx=2
   cs.numnodes=c(0,cumsum(numnodes))
   for(i in 1:post$nd) {
      ens=list()
      for(j in 1:post$m)
      {
         tree=list(id=ids[(cs.numnodes[ndx-1]+1):cs.numnodes[ndx]],
                  var=vars[(cs.numnodes[ndx-1]+1):cs.numnodes[ndx]],
                  cut=cuts[(cs.numnodes[ndx-1]+1):cs.numnodes[ndx]],
                  theta=thetas[(cs.numnodes[ndx-1]+1):cs.numnodes[ndx]])
         ens[[j]]=tree
         ndx=ndx+1
      }
      mt[[i]]=ens
   }


   st=list()
   ndx=2
   cs.numnodes=c(0,cumsum(snumnodes))
   for(i in 1:post$nd) {
      ens=list()
      for(j in 1:post$mh)
      {
         tree=list(id=sids[(cs.numnodes[ndx-1]+1):cs.numnodes[ndx]],
                  var=svars[(cs.numnodes[ndx-1]+1):cs.numnodes[ndx]],
                  cut=scuts[(cs.numnodes[ndx-1]+1):cs.numnodes[ndx]],
                  theta=sthetas[(cs.numnodes[ndx-1]+1):cs.numnodes[ndx]])
         ens[[j]]=tree
         ndx=ndx+1
      }
      st[[i]]=ens
   }

   return(list(mt=mt,st=st))
}



# Save a posterior tree fit from the tmp working directory
# into a local zip file given by [file].zip
# If not file option specified, uses [model name].zip as the file.
openbt.save<-function(post,fname=NULL)
{
   if(class(post)!="OpenBT_posterior") stop("Invalid object.\n")

   if(is.null(fname)) fname=post$modelname
   if(substr(fname,nchar(fname)-3,nchar(fname))!=".obt") fname=paste(fname,".obt",sep="")

   save(post,file=paste(post$folder,"/post.RData",sep=""))

   zipr(fname,paste(post$folder,"/",list.files(post$folder),sep=""))
   cat("Saved posterior to ",fname,"\n")
}


# Load a posterior tree fit from a zip file into a tmp working directory.
openbt.load<-function(fname)
{
   if(substr(fname,nchar(fname)-3,nchar(fname))!=".obt") fname=paste(fname,".obt",sep="")

   folder=tempdir(check=TRUE)
   if(!dir.exists(folder)) dir.create(folder)
   tmpsubfolder=tempfile(tmpdir="")
   tmpsubfolder=substr(tmpsubfolder,5,nchar(tmpsubfolder))
   tmpsubfolder=paste("openbt",tmpsubfolder,sep="")
   folder=paste(folder,"/",tmpsubfolder,sep="")
   if(!dir.exists(folder)) dir.create(folder)

   unzip(fname,exdir=folder)
   post=loadRData(paste(folder,"/post.RData",sep=""))
   post$folder=folder

   return(post)
}


# This is so lame. Seriously.
loadRData <- function(fname)
{
    load(fname)
    get(ls()[ls() != "fname"])
}


print.OpenBT_posterior<-function(post)
{
   MODEL_BT=1
   MODEL_BINOMIAL=2
   MODEL_POISSON=3
   MODEL_BART=4
   MODEL_HBART=5
   MODEL_PROBIT=6
   MODEL_MODIFIEDPROBIT=7
   MODEL_MERCK_TRUNCATED=8

   cat("OpenBT Posterior\n")
   cat("Model type: ")
   if(post$modeltype==MODEL_BART)
   {
      cat("Bayesian Additive Regression Trees model (BART)\n")
      cat("k=",post$k,"\n")
      cat("tau=",post$tau,"\n")
      cat("lambda=",post$overalllambda,"\n")
      cat("nu=",post$overallnu,"\n")
   }
   if(post$modeltype==MODEL_HBART)
   {
      cat("Heteroscedastic Bayesian Additive Regression Trees model (HBART)\n")
   }
   if(post$modeltype==MODEL_PROBIT)
   {
      cat("Dichotomous outcome model: Albert & Chib Probit (fixed)\n")
   }
   if(post$modeltype==MODEL_MODIFIEDPROBIT)
   {
      cat("Dichotomous outcome model: Modified Albert & Chib Probit\n")
   }
   if(post$modeltype==MODEL_MERCK_TRUNCATED)
   {
      cat("Truncated BART model\n")
   }

   cat("ntree=", post$m, " \n",sep="")
   cat("ntreeh=",post$mh," \n",sep="")
   cat(post$nd," posterior draws.\n")
   summary(post$xicuts)
}

summary.OpenBT_posterior<-function(post)
{
   cat("No summary method for object.\n")
}

print.OpenBT_predict<-function(pred)
{
   MODEL_BT=1
   MODEL_BINOMIAL=2
   MODEL_POISSON=3
   MODEL_BART=4
   MODEL_HBART=5
   MODEL_PROBIT=6
   MODEL_MODIFIEDPROBIT=7
   MODEL_MERCK_TRUNCATED=8

   cat("OpenBT Prediction\n")
   cat(ncol(pred$mdraws), " prediction locations.\n")
   cat(nrow(pred$mdraws), " realizations.\n")
   if(pred$modeltype==MODEL_PROBIT || pred$modeltype==MODEL_MODIFIEDPROBIT)
   {
      cat("Probability quantiles: ",pred$q.lower,",",pred$q.upper,"\n")
   }
   cat("Mean quantiles: ",pred$q.lower,",",pred$q.upper,"\n")
   cat("Variance quantiles: ",pred$q.lower,",",pred$q.upper,"\n\n")
}

summary.OpenBT_predict<-function(pred)
{
   cat("No summary method for object.\n")
}

print.OpenBT_sobol<-function(sobol)
{
   cat("OpenBT Sobol Indices\n")
   cat(ncol(sobol$vidraws), " variables.\n")
   cat(nrow(sobol$vidraws), " realizations.\n")
}

summary.OpenBT_sobol<-function(sobol)
{
   cat("Summary of Posterior Sobol Sensitivity Indices\n")

   cat("Expected Sobol Indices (Mean)\n")
   print(sobol$msi)
   print(sobol$mtsi)
   print(sobol$msij)

   cat("\nStd. Dev. of Sobol Indices (Mean)\n")
   print(sobol$msi.sd)
   print(sobol$tsi.sd)
   print(sobol$sij.sd)
}

plot.OpenBT_sobol<-function(sobol)
{
   par(mfrow=c(3,1))
   boxplot(sobol$sidraws,ylab="Sobol Sensitivity",main="First Order Sobol Indices",xlab="Variables")
   boxplot(sobol$sijdraws,ylab="Sobol Sensitivity",main="Two-way Sobol Indices",xlab="Variables")
   boxplot(sobol$tsidraws,ylab="Sobol Sensitivity",main="Total Sobol Indices",xlab="Variables")
}

print.OpenBT_vartivity<-function(vartivity)
{
   cat("OpenBT Variable Activity\n")
   cat(ncol(vartivity$vdraws), " variables.\n")
   cat(nrow(vartivity$vdraws), " realizations.\n")
}

summary.OpenBT_vartivity<-function(vartivity)
{
   cat("Summary of Posterior Variable Activity\n")

   p=length(vartivity$mvdraws)
   if(p<11)
   {
      cat("Expected Variable Activity (Mean)\n")
      mean.vartivity=vartivity$mvdraws
      ix=sort(mean.vartivity,index.return=TRUE,decreasing=TRUE)$ix
      mean.vartivity=round(mean.vartivity[ix],2)
      names(mean.vartivity)=ix
      print(mean.vartivity)
      cat("Expected Variable Activity (Variance)\n")
      sd.vartivity=vartivity$mvdrawsh
      ix=sort(sd.vartivity,index.return=TRUE,decreasing=TRUE)$ix
      sd.vartivity=round(sd.vartivity[ix],2)
      names(sd.vartivity)=ix
      print(sd.vartivity)
   }
   else
   {
      cat("Expected Variable Activity (Mean)\n")
      mean.vartivity=vartivity$mvdraws
      ix=sort(mean.vartivity,index.return=TRUE,decreasing=TRUE)$ix
      mean.vartivity=round(mean.vartivity[ix],2)
      rest=sum(mean.vartivity[11:p])
      mean.vartivity=mean.vartivity[1:11]
      mean.vartivity[11]=rest
      names(mean.vartivity)=c(ix[1:10],"...")
      print(mean.vartivity)
      cat("Expected Variable Activity (Variance)\n")
      sd.vartivity=vartivity$mvdrawsh
      ix=sort(sd.vartivity,index.return=TRUE,decreasing=TRUE)$ix
      sd.vartivity=round(sd.vartivity[ix],2)
      rest=sum(sd.vartivity[11:p])
      sd.vartivity=sd.vartivity[1:11]
      sd.vartivity[11]=rest
      names(sd.vartivity)=c(ix[1:10],"...")
      print(sd.vartivity)
   }
}

plot.OpenBT_vartivity<-function(vartivity)
{
   par(mfrow=c(1,2))
   yrange=c(0,max(max(vartivity$vdraws),max(vartivity$vdrawsh)))
   boxplot(vartivity$vdraws,ylab="% node splits",main="Mean Trees",xlab="Variables",ylim=yrange)
   boxplot(vartivity$vdrawsh,ylab="% node splits",main="Variance Trees",xlab="Variables",ylim=yrange)
}

summary.OpenBT_cutinfo<-function(xi)
{
   p=length(xi)
   cat("Number of variables: ",p,"\n")
   cat("Number of cutpoints per variable\n")
   for(i in 1:p)
   {
      cat("Variable ",i,": ",length(xi[[i]])," cutpoints\n")
   }
}

print.OpenBT_cutinfo<-function(xi)
{
   summary.OpenBT_cutinfo(xi)
}

# Takes the n x p design matrix and a scalar or vector of number of cutpoints 
# per variable, returns a BARTcutinfo object with variables/cutpoints initalized.
makecuts.openbt<-function(x,numcuts)
{
   p=ncol(x)
   if(length(numcuts)==1)
   {
      numcuts=rep(numcuts,p)
   }
   else if(ncol(x) != length(numcuts))
   {
      cat("Number of variables does not equal length of numcuts vector!\n")
      return(0)
   }

   xi=vector("list",p)
   minx=apply(x,2,min)
   maxx=apply(x,2,max)
   for(i in 1:p)
   {
      xinc=(maxx[i]-minx[i])/(numcuts[i]+1)
      xi[[i]]=(1:numcuts[i])*xinc+minx[i]
   }

   class(xi)="OpenBT_cutinfo"
   return(xi)
}

# Takes an existing OpenBT_cutinfo object xi and the particular variable to update, id,
# and the vector of cutpoints to manually assign to that variable, updates and returns
# the modified OpenBT_cutinfo object.
setvarcuts.openbt<-function(xi,id,cutvec)
{
   p=length(xi)
   if(id>p || id<1)
   {
      cat("Invalid variable specified\n")
      return(0)
   }
   xi[[id]]=cutvec
   return(xi)
}
