
## fill with nulls randomly rows that exceed n.col number
fill.null = function(i, n.col){
  smp.r <- which(i>0)
  if(length(smp.r)>n.col){
    smp.r <- sample(smp.r, size=length(smp.r)-n.col)
    i[smp.r] <- 0
  }
  return(i)
}

hor2xyd <- function(x, U="UHDICM", L="LHDICM", treshold.T=15){
  x$DEPTH <- x[,U] + (x[,L] - x[,U])/2
  x$THICK <- x[,L] - x[,U]
  sel <- x$THICK < treshold.T
  ## begin and end of the horizon:
  x1 <- x[!sel,]; x1$DEPTH = x1[,L]
  x2 <- x[!sel,]; x2$DEPTH = x1[,U]
  y <- do.call(rbind, list(x, x1, x2))
  return(y)
}

temp.from.geom <- function(fi, day, a=30.419375, 
                           b=-15.539232, elev=0, t.grad=0.6) {
  f = ifelse(fi==0, 1e-10, fi)
  costeta = cos( (day-18 )*pi/182.5 + 2^(1-sign(fi) ) *pi) 
  cosfi = cos(fi*pi/180 )
  A = cosfi
  B = (1-costeta ) * abs(sin(fi*pi/180 ) )
  x = a*A + b*B - t.grad * elev / 100
  return(x)
}


comp.var = function(x, r1, r2, v1, v2){
  r =  rowSums( x[,c(r1, r2)] * 1/(x[,c(v1, v2)]^2) ) / rowSums( 1/(x[,c(v1, v2)]^2) )
  ## https://stackoverflow.com/questions/13593196/standard-deviation-of-combined-data
  v = sqrt( rowMeans(x[,c(r1, r2)]^2 + x[,c(v1, v2)]^2) - rowMeans( x[,c(r1, r2)])^2 )
  return(data.frame(response=r, stdev=v))
}

# grand.sd   <- function(S, M, N) {sqrt(weighted.mean(S^2 + M^2, N) - weighted.mean(M, N)^2)}
#comp.var(x=data.frame(r1=6.2, r2=7, v1=0.4, v2=0.6), "r1", "r2", "v1", "v2")
#comp.var(x=data.frame(r1=6.4, r2=6.6, v1=0.4, v2=0.6), "r1", "r2", "v1", "v2")

pfun <- function(x,y, ...){
  panel.hexbinplot(x,y, ...)
  panel.abline(0,1,lty=1,lw=2,col="black")
}

plot_hexbin <- function(varn, breaks, main, meas, pred, colorcut=c(0,0.01,0.03,0.07,0.15,0.25,0.5,0.75,1), pal = openair::openColours("increment", 18)[-18], in.file, log.plot=FALSE, out.file){ ## pal=R_pal[["bpy_colors"]][1:18]
  require("hexbin"); require("plotKML"); require("latticeExtra"); require("openair")
  if(missing(out.file)){ out.file = paste0("./img/plot_CV_", varn, ".png") }
  if(!file.exists(out.file)){
    if(missing(pred)){
      m <- readRDS.gz(in.file)
      #pred <- t.m$learner.model$super.model$learner.model$fitted.values
      pred <- m$predictions
    }
    if(missing(meas)){
      meas <- t.m$learner.model$super.model$learner.model$model[,1]
    }
    if(log.plot==TRUE){
      R.squared = yardstick::ccc(data.frame(pred, meas), truth="meas", estimate="pred")
      #pred <- 10^pred-1
      pred = expm1(pred)
      #meas <- 10^meas-1
      meas = expm1(meas)
      d.meas <- min(meas, na.rm=TRUE)
    } else {
      d.meas <- min(meas, na.rm=TRUE)
      R.squared = yardstick::ccc(data.frame(pred, meas), truth="meas", estimate="pred")
    }
    main.txt = paste0(main, "  (CCC: ", signif(R.squared$.estimate, 3), ")")
    png(file = out.file, res = 150, width=850, height=850, type="cairo")
    if(log.plot==TRUE){
      pred <- pred+ifelse(d.meas==0, 1, d.meas)
      meas <- meas+ifelse(d.meas==0, 1, d.meas)
      lim <- range(breaks)+ifelse(d.meas==0, 1, d.meas)
      meas <- ifelse(meas<lim[1], lim[1], ifelse(meas>lim[2], lim[2], meas))
      plt <- hexbinplot(meas~pred, colramp=colorRampPalette(pal), main=main.txt, ylab="measured", xlab="predicted", type="g", lwd=1, lcex=8, inner=.2, cex.labels=.8, scales=list(x = list(log = 2, equispaced.log = FALSE), y = list(log = 2, equispaced.log = FALSE)), asp=1, xbins=50, ybins=50, xlim=lim, ylim=lim, panel=pfun, colorcut=colorcut)
    } else {
      lim <- range(breaks)
      meas <- ifelse(meas<lim[1], lim[1], ifelse(meas>lim[2], lim[2], meas))
      plt <- hexbinplot(meas~pred, colramp=colorRampPalette(pal), main=main.txt, ylab="measured", xlab="predicted", type="g", lwd=1, lcex=8, inner=.2, cex.labels=.8, xlim=lim, ylim=lim, asp=1, xbins=50, ybins=50, panel=pfun, colorcut=colorcut)
    }
    print(plt)
    dev.off()
  }
}
