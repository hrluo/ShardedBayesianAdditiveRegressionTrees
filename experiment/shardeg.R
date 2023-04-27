source('../openbt.R')
braninsc <- function(xx)
{  
  x1 <- xx[1]
  x2 <- xx[2]
  
  x1bar <- 15*x1 - 5
  x2bar <- 15 * x2
  
  term1 <- x2bar - 5.1*x1bar^2/(4*pi^2) + 5*x1bar/pi - 6
  term2 <- (10 - 10/(8*pi)) * cos(x1bar)
  
  y <- (term1^2 + term2 - 44.81) / 51.95
  return(y)
}

fried <- function(xx)
{
  x1 <- xx[1]
  x2 <- xx[2]
  x3 <- xx[3]
  x4 <- xx[4]
  x5 <- xx[5]
  
  term1 <- 10 * sin(pi*x1*x2)
  term2 <- 20 * (x3-0.5)^2
  term3 <- 10*x4
  term4 <- 5*x5
  
  y <- term1 + term2 + term3 + term4
  return(y)
}

fried20 <- function(xx)
{
  y <- fried(xx[1:5])+fried(xx[6:10])+fried(xx[11:15])+fried(xx[16:20])

  return(y)
}

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
  plot(y,fitp$mmean,xlab="observed",ylab="fitted")
  abline(0,1)
  sqrt(mean((y-fitp$mmean)^2))
}





#-----------------------------------------------------------------------
# Simulate branin data for testing
#-----------------------------------------------------------------------
set.seed(99)
n=5000
p=2
x = matrix(runif(n*p),ncol=p)
xp = matrix(runif(n*p),ncol=p)
y=rep(0,n)
for(i in 1:n) y[i] = braninsc(x[i,])
yp=rep(0,n)
for(i in 1:n) yp[i] = braninsc(xp[i,])

# Load the R wrapper functions to the OpenBT library.
#source("openbt.R")

# Save rmsep's
rmse=rep(0,6)

# Fit BART models
m=1
fit.baseline=obt_baseline(x,y,m)
rmse[1]=obt_plot(fit.baseline,xp,yp)
fit.sh0.ps1=obt_shard0(x,y,m)
rmse[2]=obt_plot(fit.sh0.ps1,xp,yp)
fit.sh1.ps1=obt_shard1(x,y,m)
rmse[3]=obt_plot(fit.sh1.ps1,xp,yp)
fit.sh2.ps1=obt_shard2(x,y,m)
rmse[4]=obt_plot(fit.sh2.ps1,xp,yp)
fit.baseline.2500=obt_baseline(x[1:2500,],y[1:2500],m)
rmse[5]=obt_plot(fit.baseline.2500,xp,yp)
fit.baseline.1250=obt_baseline(x[1:1250,],y[1:1250],m)
rmse[6]=obt_plot(fit.baseline.1250,xp,yp)

#[1] 0.1302223 0.1473772 0.1870550 0.3041656 0.1779247 0.9777898 0.9889601
#[8] 0.15124085 0.18129714






#-----------------------------------------------------------------------
# Simulate friedman data for testing
#-----------------------------------------------------------------------
set.seed(99)
n=5000
p=5
x = matrix(runif(n*p),ncol=p)
xp = matrix(runif(n*p),ncol=p)
y=rep(0,n)
for(i in 1:n) y[i] = fried(x[i,])
yp=rep(0,n)
for(i in 1:n) yp[i] = fried(xp[i,])

# Save rmsep's
rmse=rep(0,6)

# Fit BART models
m=50
fit.baseline=obt_baseline(x,y,m)
rmse[1]=obt_plot(fit.baseline,xp,yp)
fit.sh0.ps1=obt_shard0(x,y,m)
rmse[2]=obt_plot(fit.sh0.ps1,xp,yp)
fit.sh1.ps1=obt_shard1(x,y,m)
rmse[3]=obt_plot(fit.sh1.ps1,xp,yp)
fit.sh2.ps1=obt_shard2(x,y,m)
rmse[4]=obt_plot(fit.sh2.ps1,xp,yp)
fit.baseline.2500=obt_baseline(x[1:2500,],y[1:2500],m)
rmse[5]=obt_plot(fit.baseline.2500,xp,yp)
fit.baseline.1250=obt_baseline(x[1:1250,],y[1:1250],m)
rmse[6]=obt_plot(fit.baseline.1250,xp,yp)








#-----------------------------------------------------------------------
# Simulate branin data with null variables
#-----------------------------------------------------------------------
set.seed(99)
n=5000
p=10
x = matrix(runif(n*p),ncol=p)
xp = matrix(runif(n*p),ncol=p)
y=rep(0,n)
for(i in 1:n) y[i] = braninsc(x[i,1:2])
yp=rep(0,n)
for(i in 1:n) yp[i] = braninsc(xp[i,1:2])

# Fit BART models
m=1
fit.baseline=obt_baseline(x,y,m)
rmse[1]=obt_plot(fit.baseline,xp,yp)
fit.sh0.ps1=obt_shard0(x,y,m)
rmse[2]=obt_plot(fit.sh0.ps1,xp,yp)
fit.sh1.ps1=obt_shard1(x,y,m)
rmse[3]=obt_plot(fit.sh1.ps1,xp,yp)
fit.sh2.ps1=obt_shard2(x,y,m)
rmse[4]=obt_plot(fit.sh2.ps1,xp,yp)
fit.baseline.2500=obt_baseline(x[1:2500,],y[1:2500],m)
rmse[5]=obt_plot(fit.baseline.2500,xp,yp)
fit.baseline.1250=obt_baseline(x[1:1250,],y[1:1250],m)
rmse[6]=obt_plot(fit.baseline.1250,xp,yp)






#-----------------------------------------------------------------------
# Simulate friedman data with null variables
#-----------------------------------------------------------------------
set.seed(99)
n=5000
p=20
x = matrix(runif(n*p),ncol=p)
xp = matrix(runif(n*p),ncol=p)
y=rep(0,n)
for(i in 1:n) y[i] = fried(x[i,1:5])
yp=rep(0,n)
for(i in 1:n) yp[i] = fried(xp[i,1:5])

# Save rmsep's
rmse=rep(0,6)

# Fit BART models
m=50
fit.baseline=obt_baseline(x,y,m)
rmse[1]=obt_plot(fit.baseline,xp,yp)
fit.sh0.ps1=obt_shard0(x,y,m)
rmse[2]=obt_plot(fit.sh0.ps1,xp,yp)
fit.sh1.ps1=obt_shard1(x,y,m)
rmse[3]=obt_plot(fit.sh1.ps1,xp,yp)
fit.sh2.ps1=obt_shard2(x,y,m)
rmse[4]=obt_plot(fit.sh2.ps1,xp,yp)
fit.baseline.2500=obt_baseline(x[1:2500,],y[1:2500],m)
rmse[5]=obt_plot(fit.baseline.2500,xp,yp)
fit.baseline.1250=obt_baseline(x[1:1250,],y[1:1250],m)
rmse[6]=obt_plot(fit.baseline.1250,xp,yp)






#-----------------------------------------------------------------------
# Simulate friedman20 data
#-----------------------------------------------------------------------
set.seed(99)
n=100000
p=20
x = matrix(runif(n*p),ncol=p)
xp = matrix(runif(n*p),ncol=p)
y=rep(0,n)
for(i in 1:n) y[i] = fried20(x[i,])
yp=rep(0,n)
for(i in 1:n) yp[i] = fried20(xp[i,])

# Save rmsep's
rmse=rep(0,6)

# Fit BART models
m=200
fit.baseline=obt_baseline(x,y,m)
rmse[1]=obt_plot(fit.baseline,xp,yp)
fit.sh0.ps1=obt_shard0(x,y,m)
rmse[2]=obt_plot(fit.sh0.ps1,xp,yp)
fit.sh1.ps1=obt_shard1(x,y,m)
rmse[3]=obt_plot(fit.sh1.ps1,xp,yp)
fit.sh2.ps1=obt_shard2(x,y,m)
rmse[4]=obt_plot(fit.sh2.ps1,xp,yp)
fit.baseline.2500=obt_baseline(x[1:(n/2),],y[1:(n/2)],m)
rmse[5]=obt_plot(fit.baseline.2500,xp,yp)
fit.baseline.1250=obt_baseline(x[1:(n/4),],y[1:(n/4)],m)
rmse[6]=obt_plot(fit.baseline.1250,xp,yp)


#Todos:
#- Simple way to change the cuts for U to be 100 cuts or 2^sharddepth-1 cuts.
#    - This would only work easily if root node splits on .5, left child splits on .25, etc.
#      Otherwise will not be balanced for the Tu.
#- Enable perturb proposal for U splits if it is not already enabled.
