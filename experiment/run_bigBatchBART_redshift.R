
obt_baseline=function(x,y,m)
{
  fit=openbt(x,y,pbd=c(1.0,0.0),ntree=m,ntreeh=1,numcut=100,tc=4,
            probchv=0.0,shardepth=-1,
            model="bart",modelname="branin")

  return(fit)
}

obt_shard0=function(x,y,m)
{
  fit=openbt(x,y,pbd=c(1.0,0.0),ntree=m,ntreeh=1,numcut=100,tc=4,
            probchv=0.0,shardepth=0,shardpsplit=1.0,randshard=FALSE,
            model="bart",modelname="branin")

  return(fit)
}

obt_shard1=function(x,y,m)
{
  fit=openbt(x,y,pbd=c(1.0,0.0),ntree=m,ntreeh=1,numcut=100,tc=4,
            probchv=0.0,shardepth=1,shardpsplit=1.0,randshard=FALSE,
            model="bart",modelname="branin")

  return(fit)
}

obt_shard2=function(x,y,m)
{
  fit=openbt(x,y,pbd=c(1.0,0.0),ntree=m,ntreeh=1,numcut=100,tc=4,
            probchv=0.0,shardepth=2,shardpsplit=1.0,randshard=FALSE,
            model="bart",modelname="branin")

  return(fit)
}

obt_shard0_ps5=function(x,y,m)
{
  fit=openbt(x,y,pbd=c(1.0,0.0),ntree=m,ntreeh=1,numcut=100,tc=4,
            probchv=0.0,shardepth=0,shardpsplit=0.5,randshard=FALSE,
            model="bart",modelname="branin")

  return(fit)
}

obt_shard1_ps5=function(x,y,m)
{
  fit=openbt(x,y,pbd=c(1.0,0.0),ntree=m,ntreeh=1,numcut=100,tc=4,
            probchv=0.0,shardepth=1,shardpsplit=0.5,randshard=FALSE,
            model="bart",modelname="branin")

  return(fit)
}

obt_shard2_ps5=function(x,y,m)
{
  fit=openbt(x,y,pbd=c(1.0,0.0),ntree=m,ntreeh=1,numcut=100,tc=4,
            probchv=0.0,shardepth=2,shardpsplit=0.5,randshard=FALSE,
            model="bart",modelname="branin")

  return(fit)
}


obt_plot=function(fit,x,y)
{
  fitp=predict.openbt(fit,x,tc=4)
  #plot(y,fitp$mmean,xlab="observed",ylab="fitted")
  #abline(0,1)
  sqrt(mean((y-fitp$mmean)^2))
}

obt_quantile=function(fit,x,y)
{
#This function exports a RData file for further analysis. 
  fitp=predict.openbt(fit,x,tc=4,q.lower=0.025,q.upper=0.975)
  #RMSE/quantiles
  res1<-(fitp$m.lower<y) & (fitp$m.upper>y)
  #res1<-(fitp$m.lower<fit$y.train) & (fitp$m.upper>fit$y.train)
  return( sum(res1)/length(res1) )
  #return(list(sqrt(mean((y-fitp$mmean)^2)),fitp$m.lower,fitp$m.upper))
}





#-----------------------------------------------------------------------
# Simulate branin data for testing
#-----------------------------------------------------------------------
set.seed(99)
n=300
p=2
x = matrix(runif(n*p),ncol=p)
xp = matrix(runif(n*p),ncol=p)
y=rep(0,n)
for(i in 1:n) y[i] = 0#braninsc(x[i,])
yp=rep(0,n)
for(i in 1:n) yp[i] = 0#braninsc(xp[i,])
print('???')
print(typeof(x))

#Read from commandline if applicable.
#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  cat("\n No file supplied, Branin will be fitted. Exactly 2 arguments must be supplied: seed numtrees x.train.csv y.train.csv x.test.csv y.test.csv modelname", call.=FALSE)
  m=1
}else{
  # default output file
  print(args)
  sed=as.numeric(args[1])
  set.seed(as.numeric(sed))
  m = as.numeric(args[2])
  cat('number of trees:',m)
  xcsvfile = as.character(args[3])
  x <- read.csv(file = xcsvfile,header=FALSE)
  x <- as.matrix(x)
  xcsvfile = as.character(args[5])
  xp <- read.csv(file = xcsvfile,header=FALSE)
  xp <- as.matrix(xp)
  n = dim(x)[1]
  p = dim(x)[2]
  #print('???')
  #print(x)
  #print(as.numeric(x))
  ycsvfile = as.character(args[4])
  y <- read.csv(file = ycsvfile,header=FALSE)
  y <- as.matrix(y)
  ycsvfile = as.character(args[6])
  yp <- read.csv(file = ycsvfile,header=FALSE)
  yp <- as.matrix(yp)
  cat('x dimensions:',n,p)
  cat('y dimensions:',length(y))
  summaryfile = as.character(args[7])
}
#Check if we already have this result, if so, skip it.
X_file_check = paste0(summaryfile,'_',as.character(m),'_',as.character(sed),'_summary_LHS.csv')
if(file.exists(X_file_check)){
  cat(X_file_check,' already exists, skipping!\n')
  quit()
}

# Load the R wrapper functions to the OpenBT library.
source("openbt.R")

# Save rmsep's
rmse=rep(0,9)
coverage=rep(0,9)
modeltime=rep(0,9)

# Fit BART models
ptm <- proc.time()
fit.baseline=obt_baseline(x,y,m)
rmse[1]=obt_plot(fit.baseline,xp,yp)
coverage[1]=obt_quantile(fit.baseline,xp,yp)
cat('fit.baseline Used time:')
modeltime[1]<-(proc.time() - ptm)[1]

ptm <- proc.time()
fit.sh0.ps1=obt_shard0(x,y,m)
rmse[2]=obt_plot(fit.sh0.ps1,xp,yp)
coverage[2]=obt_quantile(fit.sh0.ps1,xp,yp)
cat('fit.sh0.ps1 Used time:')
modeltime[2]<-(proc.time() - ptm)[1]

ptm <- proc.time()
fit.sh1.ps1=obt_shard1(x,y,m)
rmse[3]=obt_plot(fit.sh1.ps1,xp,yp)
coverage[3]=obt_quantile(fit.sh1.ps1,xp,yp)
cat('fit.sh1.ps1 Used time:')
modeltime[3]<-(proc.time() - ptm)[1]

ptm <- proc.time()
fit.sh2.ps1=obt_shard2(x,y,m)
rmse[4]=obt_plot(fit.sh2.ps1,xp,yp)
coverage[4]=obt_quantile(fit.sh2.ps1,xp,yp)
cat('fit.sh2.ps1 Used time:')
modeltime[4]<-(proc.time() - ptm)[1]

ptm <- proc.time()
fit.baseline.half=obt_baseline(x[1:as.integer(n/2),],y[1:as.integer(n/2)],m)
rmse[5]=obt_plot(fit.baseline.half,xp,yp)
coverage[5]=obt_quantile(fit.baseline.half,xp,yp)
cat('fit.baseline.half Used time:')
modeltime[5]<-(proc.time() - ptm)[1]

ptm <- proc.time()
fit.baseline.quad=obt_baseline(x[1:as.integer(n/4),],y[1:as.integer(n/4)],m)
rmse[6]=obt_plot(fit.baseline.quad,xp,yp)
coverage[6]=obt_quantile(fit.baseline.quad,xp,yp)
cat('fit.baseline.quad Used time:')
modeltime[6]<-(proc.time() - ptm)[1]

ptm <- proc.time()
fit.sh0.ps05=obt_shard0_ps5(x,y,m)
rmse[7]=obt_plot(fit.sh0.ps05,xp,yp)
coverage[7]=obt_quantile(fit.sh0.ps05,xp,yp)
cat('fit.sh0.ps05 Used time:')
modeltime[7]<-(proc.time() - ptm)[1]

ptm <- proc.time()
fit.sh1.ps05=obt_shard1_ps5(x,y,m)
rmse[8]=obt_plot(fit.sh1.ps05,xp,yp)
coverage[8]=obt_quantile(fit.sh1.ps05,xp,yp)
cat('fit.sh1.ps05 Used time:')
modeltime[8]<-(proc.time() - ptm)[1]

ptm <- proc.time()
fit.sh2.ps05=obt_shard2_ps5(x,y,m)
rmse[9]=obt_plot(fit.sh2.ps05,xp,yp)
coverage[9]=obt_quantile(fit.sh2.ps05,xp,yp)
cat('fit.sh2.ps05 Used time:')
modeltime[9]<-(proc.time() - ptm)[1]

cat('\n\n\n\n\n Summary>>>>>>>>>>')
cat('\n RMSE:')
cat(rmse)
cat('\n 95% coverage:')
cat(coverage)
cat('\n time:')
cat(modeltime)
#[1] 0.1302223 0.1473772 0.1870550 0.3041656 0.1779247 0.9777898 0.9889601
#[8] 0.15124085 0.18129714
cat('\n folder path:')
cat(fit.baseline$folder)
cat('\n')
#save.image(file=paste0(fit.baseline$folder,'/all.RData'))
save.image(file=paste0(paste0(xcsvfile,'_',as.character(m),'_',as.character(sed),'_all.RData')))
# datasummary <- as.data.frame( cbind( c(dim(fit.baseline$x.train)[1],dim(fit.sh0.ps1$x.train)[1],
#   dim(fit.sh1.ps1$x.train)[1],dim(fit.sh2.ps1$x.train)[1],dim(fit.baseline.half$x.train)[1],
#   dim(fit.baseline.quad$x.train)[1],dim(fit.sh0.ps05$x.train)[1],dim(fit.sh1.ps05$x.train)[1],
#   dim(fit.sh2.ps05$x.train)[1]),rep(m,9),c(-1,0,1,2,-1,-1,0,1,2),c(-1,1,1,1,1,1,0.5,0.5,0.5),
#   t(t(rmse)),t(t(coverage)),t(t(modeltime)) ) )
datasummary <- as.data.frame( cbind( c(n,n,n,n,as.integer(n/2),as.integer(n/4),n,n,n),
  rep(m,9),c(-1,0,1,2,-1,-1,0,1,2),c(-1,1,1,1,1,1,0.5,0.5,0.5),
  t(t(rmse)),t(t(coverage)),t(t(modeltime)) ) )
colnames(datasummary)<-c('n','m','shardepth','shardpsplit','RMSE','coverage','time')
print(datasummary)
if(length(args)>0){
	write.csv(datasummary, file=paste0(xcsvfile,'_',as.character(m),'_',as.character(sed),'_summary_noise.csv'),row.names = FALSE)
}else{
	write.csv(datasummary, file=paste0(xcsvfile,'_redshift_',as.character(m),'_',as.character(sed),'_',as.character(99),'_summary.csv'),row.names = FALSE)
}
