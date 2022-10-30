
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

pred_lst <- function(tvar, id, yl, t.m, eml.cf, in.dir="./input/"){
  if(missing(t.m)){
    t.m <- readRDS.gz(paste0("./modelsT/eml.m_", tvar, ".rds"))
    m.train = t.m$learner.model$super.model$learner.model$model
    m.terms = all.vars(t.m$learner.model$super.model$learner.model$terms)
    eml.MSE0 = matrixStats::rowSds(as.matrix(m.train[,m.terms[-1]]), na.rm=TRUE)^2
    eml.MSE = deviance(t.m$learner.model$super.model$learner.model)/df.residual(t.m$learner.model$super.model$learner.model)
    ## correction factor:
    eml.cf = eml.MSE/mean(eml.MSE0, na.rm = TRUE)
  }
  for(year in yl){
    pred_mlr(tvar=tvar, id=id, year=year, t.m, eml.cf=eml.cf, in.dir=in.dir)
  }
}

pred_mlr <- function(tvar, id, year, t.m, eml.cf, in.dir="./input/", out.dir="./output/", multiplier=10, min.md=1, depths = c(0, 30, 60, 100)){
  out.tif = paste0(out.dir, id, "/", tvar, "_M_", depths, "cm_", year, "_1km_T", id, ".tif")
  if(any(!file.exists(out.tif))){
    retry::wait_until( memuse::Sys.meminfo()[[2]]@size > 10 , timeout = 3600)
    g30m = cbind(readRDS(paste0(in.dir, id, "/data_static.rds")),
                   readRDS(paste0(in.dir, id, "/data_", year, ".rds")))
    g30m$hzn_depth = 0
    g1kx = g30m["hzn_depth"]
    #x.lst = sapply(names(g30m@data), function(i){sum(is.na(g30m@data[,i]))})
    #str(g30m@data[,names(g30m@data)[which(x.lst>10000)]])
    #x = t.m$features[which(!t.m$features %in% names(g30m))]
    g30m = do.call(data.frame, lapply(g30m@data[,t.m$features], function(x) replace(x, is.infinite(x) | is.nan(x), NA)))
    #gc()
    cc = complete.cases(g30m)
    g1kx = g1kx[cc,"hzn_depth"]
    if(sum(cc)>0){
      retry::wait_until( memuse::Sys.meminfo()[[2]]@size > 6 , timeout = 3600)
      if(tvar=="clay_tot_psa" | tvar=="sand_tot_psa"){
        multiplier = 1
        #g1kx$pred = expm1(pred$data$response)
      }
      if(tvar=="db_od"){
        multiplier = 100
        #g1kx$pred = expm1(pred$data$response)
      }
      for(d in 1:length(depths)){
        g30m$hzn_depth = depths[d]
        pred = predict(t.m, newdata=g30m[cc,])
        out.p <- as.matrix(as.data.frame(mlr::getStackedBaseLearnerPredictions(t.m, newdata=g30m[cc,]))) 
        g1kx$model.error <- sqrt(matrixStats::rowSds(out.p, na.rm=TRUE)^2 * eml.cf) * multiplier
        #rm(out.p)
        #gc()
        g1kx$model.error <- ifelse(g1kx$model.error<min.md, min.md, g1kx$model.error)
        g1kx$pred = ifelse(pred$data$response < 0, 0, pred$data$response)*multiplier
        rgdal::writeGDAL(g1kx["pred"], out.tif[d], type="Byte", mvFlag=255, options="COMPRESS=DEFLATE")
        rgdal::writeGDAL(g1kx["model.error"], gsub("_M_", "_md_", out.tif[d]), type="Byte", mvFlag=255, options="COMPRESS=DEFLATE")
      }
    }
  }
}

OCSKGM <- function(ORCDRC, BLD, CRFVOL=0, HSIZE=30, ORCDRC.sd=10, BLD.sd=100, CRFVOL.sd=5, se.prop=TRUE){
  if(any(ORCDRC[!is.na(ORCDRC)]<0)|any(BLD[!is.na(BLD)]<0)|any(CRFVOL[!is.na(CRFVOL)]<0)){
    warning("Negative values for 'ORCDRC', 'BLD', 'CRFVOL' found")
  }
  if(any(is.na(BLD))){
    BLD = ifelse(is.na(BLD), 1.38-0.31*log1p(ORCDRC/10), BLD)
  }
  OCSKG <- ORCDRC/1000 * HSIZE/100 * BLD * (100-CRFVOL)/100
  if(se.prop==TRUE){
    ## Formula derived by Gerard. See also: [http://books.google.nl/books?id=C\_XWjSsboeUC]
    OCSKG.sd <- 1E-7*HSIZE*sqrt(BLD^2*(100-CRFVOL)^2*ORCDRC.sd^2 + ORCDRC^2*(100-CRFVOL)^2*BLD.sd^2 + ORCDRC^2*BLD^2*CRFVOL.sd^2)
    ## "kilograms per square-meter"
  }
  if(se.prop){ 
    return(data.frame(response=OCSKG, sd=OCSKG.sd))
  } else {
    return(OCSKG)
  }
}

agg_layers <- function(year, t.var, tile, in.dir="./output/", depths=c(0,30,60,100), out.depths=c(0,30,100), type="Byte", mvFlag=255){
  ## sol_db.od_mangroves.typology_m_30m_s100..100cm_2002_global_v0.1
  in.tifs = paste0(in.dir, tile, "/sol_", t.var, "_mangroves.typology_m_30m_s", depths, "..", depths, "cm_", year, "_global_v0.1.tif")
  in.mdtifs = paste0(in.dir, tile, "/sol_", t.var, "_mangroves.typology_md_30m_s", depths, "..", depths, "cm_", year, "_global_v0.1.tif")
  out.tifs = paste0(in.dir, tile, "/sol_", t.var, "_mangroves.typology_m_30m_s", out.depths[-3], "..", out.depths[-1], "cm_", year, "_global_v0.1.tif")
  if(all(file.exists(in.tifs)) & any(!file.exists(out.tifs))){
    x = raster::stack(in.tifs)
    x = as(x, "SpatialGridDataFrame")
    x$d1 = rowMeans(x@data[,1:2], na.rm=TRUE)
    x$d2 = rowMeans(x@data[,2:4], na.rm=TRUE)
    for(k in 1:2){
      writeGDAL(x[paste0("d", k)], out.tifs[k], type=type, mvFlag=mvFlag, options="COMPRESS=DEFLATE")
    }
    ## https://stats.stackexchange.com/questions/168971/variance-of-an-average-of-random-variables
    gc()
    x = raster::stack(in.mdtifs)
    x = as(x, "SpatialGridDataFrame")
    u1 = sqrt((x@data[,1]^2 + x@data[,2]^2)/4)
    u2 = sqrt((x@data[,2]^2 + x@data[,3]^2 + x@data[,4]^2)/9)
    x$d1 = ifelse(u1<1, 1, u1)
    x$d2 = ifelse(u2<1, 1, u2)
    for(k in 1:2){
      writeGDAL(x[paste0("d", k)], gsub("_m_", "_md_", out.tifs[k]), type=type, mvFlag=mvFlag, options="COMPRESS=DEFLATE")
    }
    gc()
  }
}

soc_calc <- function(year, tile, t.vars=c("log.oc", "db.od"), in.dir="./output/", layers=c("0..30", "30..100"), HSIZE.lst=c(30, 70), type="Int16", mvFlag=-32768){ 
  ## out.vars=c("soc.tha", "soc.tha.l90", "soc.tha.u90"),
  in.tifs = as.vector(unlist(sapply(t.vars, function(i){ c(paste0(in.dir, tile, "/sol_", i, "_mangroves.typology_m_30m_s", layers, "cm_", year, "_global_v0.1.tif"), paste0(in.dir, tile, "/sol_", i, "_mangroves.typology_md_30m_s", layers, "cm_", year, "_global_v0.1.tif")) })))
  out.tifs = paste0(in.dir, tile, "/sol_soc.tha_mangroves.typology_m_30m_s", layers, "cm_", year, "_global_v0.1.tif")
  if(all(file.exists(in.tifs)) & any(!file.exists(out.tifs))){
    x = raster::stack(in.tifs)
    x = as(x, "SpatialGridDataFrame")
    ## SOC content
    for(k in 1:length(layers)){
      x$soc = expm1(x@data[,k]/10)
      ## Prediction error:
      x$soc.md = (expm1((x@data[,k]+x@data[,k+2])/10)-expm1((x@data[,k]-x@data[,k+2])/10))/4
      xx = OCSKGM(ORCDRC=x$soc, BLD=x@data[,k+4]*10, CRFVOL=0, HSIZE=HSIZE.lst[k], ORCDRC.sd=x$soc.md, BLD.sd=x@data[,k+6]*10/2, CRFVOL.sd=5, se.prop=TRUE)
      x$mstock = xx$response * 10
      x$mstock.l = (xx$response - xx$sd) * 10
      x$mstock.l = ifelse(x$mstock.l<0, 0, x$mstock.l)
      x$mstock.u = (xx$response + xx$sd) * 10
      writeGDAL(x["soc"], gsub("soc.tha", "soc.wpct", out.tifs[k]), type=type, mvFlag=mvFlag, options="COMPRESS=DEFLATE")
      writeGDAL(x["mstock"], out.tifs[k], type=type, mvFlag=mvFlag, options="COMPRESS=DEFLATE")
      writeGDAL(x["mstock.l"], gsub("_m_", "_l.std_", out.tifs[k]), type=type, mvFlag=mvFlag, options="COMPRESS=DEFLATE")
      writeGDAL(x["mstock.u"], gsub("_m_", "_u.std_", out.tifs[k]), type=type, mvFlag=mvFlag, options="COMPRESS=DEFLATE")
    }
    gc()
  }
}

