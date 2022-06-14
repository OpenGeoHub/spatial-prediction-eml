
filter_na = function(m){
  if(any(is.na(m@data[,1]))){
    r = raster::raster(m)
    ## first using proximity filter:
    rf = raster::focal(r, w=matrix(1,15,15), fun=mean, na.rm=TRUE, NAonly=TRUE)
    repn = as(rf, "SpatialGridDataFrame")@data[,1]
    return(repn)
  } else {
    return(m@data[,1])
  }
}


saveRDS.gz <- function(object,file,threads=parallel::detectCores()) {
  con <- pipe(paste0("pigz -p",threads," > ",file),"wb")
  saveRDS(object, file = con)
  close(con)
}

readRDS.gz <- function(file,threads=parallel::detectCores()) {
  con <- pipe(paste0("pigz -d -c -p",threads," ",file))
  object <- readRDS(file = con)
  close(con)
  return(object)
}

#require(plyr)
plyrChunks <- function(d, n){
  is <- seq(from = 1, to = length(d), by = ceiling(n))
  if(tail(is, 1) != length(d)) {
    is <- c(is, length(d))
  }
  chunks <- plyr::llply(head(seq_along(is), -1),
                        function(i){
                          start <-  is[i];
                          end <- is[i+1]-1;
                          d[start:end]})
  lc <- length(chunks)
  td <- tail(d, 1)
  chunks[[lc]] <- c(chunks[[lc]], td)
  return(chunks)
}

## spacetime overlay
extract_st_annual <- function(tif, df, year, year.tif, coords=c("x","y"), crs, variable.name){
  if(any(!coords %in% colnames(df))){
    stop(paste("Coordinate columns", coords, "could not be found"))
  }
  sel <- df[,year] == year.tif
  if(sum(sel)>0){
    pnts = as.matrix(df[sel, coords])
    attr(pnts, "dimnames")[[2]] = c("x","y")
    df.v = terra::vect(pnts, crs=crs)
    if(file.exists(tif)){
      ov = terra::extract(terra::rast(tif), df.v)
    } else {
      ov = matrix(nrow=length(df.v), ncol=2)
    }
    ov = as.data.frame(ov)
    if(missing(variable.name)){
      variable.name = tools::file_path_sans_ext(basename(tif))
    }
    names(ov) = c("ID", variable.name)
    ov$row.id = which(sel)
    ov$ID = NULL
    return(ov)
  }
}

tune_learners <- function(data, formula, blocking, out.dir="./modelsS/", predict.type = "prob", SL.library=c("classif.ranger","classif.xgboost","classif.glmnet"), parallel="multicore", rdesc = mlr::makeResampleDesc("CV", iters = 2L) , num.trees = 85){
  tv <- all.vars(formula)[1]
  r.sel <- stats::complete.cases(data[,all.vars(formula)])
  df.s = data[which(r.sel),all.vars(formula)]
  l = sapply(df.s[,-c(1)], sd, na.rm = TRUE)
  if(length(which(l == 0))>0){
    x = which(l==0)+1
    message(paste0("The following covariates were removed (sd(x) = 0): ", paste(colnames(df.s)[x], collapse = ", "), "..."), immediate. = TRUE)
    df.s = df.s[,-x]
    pr.vars = colnames(df.s[-1])
    formula <- stats::as.formula(paste(tv, " ~", paste(pr.vars, collapse="+")))
  }
  rm(l)
  gc()
  ## Set hyperparameter space
  ## ranger
  min.mtry = round(sqrt(length(all.vars(formula)))/3)
  max.mtry = ifelse(length(all.vars(formula))>10, length(all.vars(formula))-4, length(all.vars(formula))-1)
  discrete_ps <- ParamHelpers::makeParamSet(ParamHelpers::makeDiscreteParam("mtry", values = unique(round(seq(min.mtry, max.mtry, length.out = 5)))))
  ## xgboost
  xg_model_Params = ParamHelpers::makeParamSet(
    ParamHelpers::makeDiscreteParam("nrounds", value=c(10,20,50)),
    ParamHelpers::makeDiscreteParam("max_depth", value=c(2,4,6)),
    ParamHelpers::makeDiscreteParam("eta", value=c(0.2,0.4)),
    ParamHelpers::makeDiscreteParam("subsample", value=c(1)),
    ParamHelpers::makeDiscreteParam("min_child_weight", value=c(1,2,5)),
    ParamHelpers::makeDiscreteParam("colsample_bytree", value=c(0.6))
  )
  ## mlr parameters
  ctrl = mlr::makeTuneControlGrid()
  message(paste0("Using learners: ", paste(SL.library, collapse = ", "), "..."), immediate. = TRUE)
  tsk <- mlr::makeClassifTask(data = df.s, 
                              target = tv, 
                              positive = "1",
                              blocking = blocking[which(r.sel)])
  
  out.rf = paste0(out.dir, tv, "_rf_model.rds")
  if(!file.exists(out.rf)){
    ## fine-tune mtry
    message("Running tuneParams for ranger... ", immediate. = TRUE)
    parallelMap::parallelStartSocket(parallel::detectCores())
    resR.lst <- mlr::tuneParams(mlr::makeLearner("classif.ranger", 
                                                 num.threads = round(parallel::detectCores()/length(discrete_ps$pars$mtry$values)), 
                                                 num.trees=num.trees,
                                                 predict.type = predict.type), 
                                task = tsk, 
                                resampling = rdesc, 
                                par.set = discrete_ps, 
                                control = ctrl,
                                measures = list(logloss, setAggregation(logloss, test.sd)))
    if(resR.lst$x$mtry >= length(all.vars(formula))){
      lrn.rf = mlr::makeLearner("classif.ranger", num.threads = parallel::detectCores(), num.trees=num.trees, importance="impurity", predict.type = predict.type)
    } else {
      lrn.rf = mlr::makeLearner("classif.ranger", num.threads = parallel::detectCores(), mtry=resR.lst$x$mtry, num.trees=num.trees, importance="impurity", predict.type = predict.type)
    }
    var.mod1 <- mlr::train(lrn.rf, task = tsk)
    parallelMap::parallelStop()
    saveRDS(var.mod1, out.rf)
    gc()
  } else {
    var.mod1 = readRDS(out.rf)
  }
  out.x = paste0(out.dir, tv, "_xgb_model.rds")
  if(!file.exists(out.x)){
    ## fine-tune xgboost
    lrn.xg = mlr::makeLearner("classif.xgboost", par.vals = list(objective ='multi:softprob'))
    message("Running tuneParams for xgboost... ", immediate. = TRUE)
    parallelMap::parallelStartSocket(parallel::detectCores())
    resX.lst = mlr::tuneParams(mlr::makeLearner("classif.xgboost", 
                                                predict.type = predict.type), 
                               task = tsk, resampling = rdesc, 
                               par.set = xg_model_Params, 
                               control = ctrl,
                               measures = list(logloss, setAggregation(logloss, test.sd)))
    lrn.xg = mlr::setHyperPars(lrn.xg, par.vals = resX.lst$x)
    var.mod2 <- mlr::train(lrn.xg, task = tsk)
    parallelMap::parallelStop()
    saveRDS(var.mod2, out.x)
    gc()
  } else {
    var.mod2 = readRDS(out.x)
  }
  out.glm = paste0(out.dir, tv, "_glm_model.rds")
  if(!file.exists(out.glm)){
    parallelMap::parallelStartSocket(parallel::detectCores())
    var.mod3 = mlr::train(mlr::makeLearner("classif.glmnet", predict.type = predict.type), task = tsk)
    saveRDS(var.mod3, out.glm)
    parallelMap::parallelStop()
  } else {
    var.mod3 = readRDS(out.glm)
  }  
  finetuning = list(var.mod1, var.mod2, var.mod3, formula)
  saveRDS(finetuning, paste0(out.dir, tv, "_finetune.rds"))
  return(finetuning)
}

## Train spacetime model for predicting species occurrences
train_sp_eml <- function(data, tune_result, blocking, out.dir="./modelsS/", predict.type = "prob", SL.library=c("classif.ranger","classif.xgboost","classif.glmnet"), super.learner = "classif.logreg", parallel="multicore", num.trees = 85, xyn = c("x", "y"), method = "stack.cv"){
  
  tv <- all.vars(tune_result[[4]][[2]])
  r.sel <- stats::complete.cases(data[,all.vars(tune_result[[4]])])
  df.s = data[which(r.sel),all.vars(tune_result[[4]])]
  l = sapply(df.s[,-c(1)], sd, na.rm = TRUE)
  blocking = blocking[which(r.sel)]
  ## get models
  var.mod1 = tune_result[[1]]
  var.mod2 = tune_result[[2]]
  var.mod3 = tune_result[[3]]
  ## output model:
  out.eml = paste0(out.dir, "eml.m_", tv,".rds")
  if(!file.exists(out.eml)){
    ## fit the mlr model:
    mlr::configureMlr()
    if(parallel=="multicore"){
      parallelMap::parallelStartSocket(parallel::detectCores())
    }
    message(paste0("Using learners: ", paste(SL.library, collapse = ", "), "..."), immediate. = TRUE)
    lrns <- lapply(SL.library, mlr::makeLearner)
    message("Fitting a spatial learner using 'mlr::makeClassifTask'...", immediate. = TRUE)
    tsk <- mlr::makeClassifTask(data = df.s, 
                                target = tv, 
                                positive = "1",
                                blocking = blocking)
    if(any(SL.library %in% "classif.xgboost")){
      lrns[[which(SL.library %in% "classif.xgboost")]] = mlr::makeLearner("classif.xgboost", par.vals = list(objective ='multi:softprob'))
    }
    
    lrns <- lapply(lrns, mlr::setPredictType, "prob")
    lrns[[1]] = mlr::setHyperPars(lrns[[1]], par.vals = mlr::getHyperPars(var.mod1$learner))
    lrns[[2]] = mlr::setHyperPars(lrns[[2]], par.vals = mlr::getHyperPars(var.mod2$learner))
    lrns[[3]] = mlr::setHyperPars(lrns[[3]], par.vals = list(s = min(var.mod3$learner.model$lambda)))
    gc()
    init.m <- mlr::makeStackedLearner(base.learners = lrns, 
                                      predict.type = predict.type, 
                                      method = method, 
                                      super.learner = super.learner, 
                                      resampling=mlr::makeResampleDesc(method = "CV", blocking.cv=TRUE))
    m <- mlr::train(init.m, tsk)
    if(parallel=="multicore"){
      parallelMap::parallelStop()
    }
    saveRDS(m, out.eml)
  } else {
    m = readRDS(out.eml)
  }
  return(m)
}


## Model fine-tuning wrapper
## by default run spatial CV
train_eml <- function(t.var, pr.var, X, out.dir="./modelsT/", SL.library = c("regr.ranger", "regr.rpart", "regr.nnet"), discrete_ps = makeParamSet(makeDiscreteParam("mtry", values = seq(6,40,by=2))), ctrl = mlr::makeTuneControlGrid(), rdesc = mlr::makeResampleDesc("CV", iters = 2L), outer = mlr::makeResampleDesc("CV", iters = 2L), inner = mlr::makeResampleDesc("Holdout"), ctrlF = mlr::makeFeatSelControlRandom(maxit = 30), out.m.rds, save.rm=TRUE, rf.feature=TRUE){
  require(mlr)
  ## "regr.glmboost"
  ## regression matrix
  rm.x = X[,c(t.var, pr.var)]
  r.sel = complete.cases(rm.x)
  rm.x = rm.x[r.sel,]
  ## remove columns without enough variation:
  c2.sd = sapply(rm.x, function(i){var(i, na.rm=TRUE)})
  rm.pr = which(c2.sd < 0.005)
  if(length(rm.pr)>0){
    rm.x = rm.x[,-rm.pr]
  }
  if(save.rm==TRUE){
    saveRDS.gz(rm.x, paste0(out.dir, "rm.", t.var, ".rds"))
  }
  parallelMap::parallelStartSocket(parallel::detectCores())
  tsk0 <- mlr::makeRegrTask(data = rm.x, target = t.var, coordinates = X[r.sel,c("x", "y")], blocking = as.factor(X$ID[r.sel]))
  out.t.mrf = paste0(out.dir, "t.mrf.", t.var, ".rds")
  if(!file.exists(out.t.mrf)){
    ## fine-tune mtry
    resR.lst = tuneParams(mlr::makeLearner("regr.ranger", num.threads = round(parallel::detectCores()/length(discrete_ps$pars$mtry$values)), num.trees=85), task = tsk0, resampling = rdesc, par.set = discrete_ps, control = ctrl)
    ## feature selection
    lrn.rf = mlr::makeLearner("regr.ranger", num.threads = parallel::detectCores(), mtry=resR.lst$x$mtry, num.trees=85, importance="impurity")
    if(rf.feature==TRUE){
      lrn1 = mlr::makeFeatSelWrapper(lrn.rf, resampling = inner, control = ctrlF, show.info=TRUE)
    } else {
      lrn1 = lrn.rf
    }
    var.mod1 = mlr::train(lrn1, task = tsk0)
    saveRDS.gz(var.mod1, out.t.mrf)
  } else {
    var.mod1 = readRDS.gz(out.t.mrf)
  }
  if(rf.feature==TRUE & any(class(var.mod1)=="FeatSelModel")){
    var.sfeats1 = mlr::getFeatSelResult(var.mod1)
  } else {
    var.sfeats1 = data.frame(x=pr.var[which(pr.var %in% names(rm.x))])
  }
  ## new shorter formula
  ## we add depth otherwise not a 3D model
  formulaString.y = as.formula(paste(t.var, ' ~ ', paste(var.sfeats1$x, collapse="+")))
  ## final EML model
  if(missing(out.m.rds)){ out.m.rds <- paste0(out.dir, "eml.m_", t.var,".rds") }
  if(!file.exists(out.m.rds)){
    tskF <- mlr::makeRegrTask(data = rm.x[,all.vars(formulaString.y)], target = t.var, blocking = as.factor(X$ID[r.sel])) ## coordinates = X[r.sel,c("X", "Y")],
    var.mod1 <- readRDS.gz(paste0(out.dir, "t.mrf.", t.var, ".rds"))
    lrn.rf = mlr::setHyperPars(mlr::makeLearner(SL.library[1]), par.vals = getHyperPars(var.mod1$learner))
    lrnsE <- list(lrn.rf, mlr::makeLearner(SL.library[2]), mlr::makeLearner(SL.library[3])) #, mlr::makeLearner(SL.library[4]))
    init.m <- mlr::makeStackedLearner(base.learners = lrnsE, predict.type = "response", method = "stack.cv", super.learner = "regr.lm", resampling=makeResampleDesc(method = "CV", blocking.cv=TRUE))
    t.m <- mlr::train(init.m, tskF)
    #t.m$learner.model$super.model$learner.model
    saveRDS.gz(t.m, out.m.rds)
  }
  parallelMap::parallelStop()
}

cat_eml = function(t.var, out.dir="./modelsT/", n.max=30){
  r.file = paste0(out.dir, "resultsFit_", t.var, ".txt")
  if(!file.exists(r.file)){
    out.m.rds = paste0(out.dir, "eml.m_", t.var,".rds")
    t.m = readRDS.gz(out.m.rds)
    x.s = summary(t.m$learner.model$super.model$learner.model)
    cat("Results of ensemble model fitting 'ranger', ...:\n", file=r.file)
    cat("\n", file=r.file, append=TRUE)
    cat(paste("Variable:", t.var, "\n"), file=r.file, append=TRUE)
    cat(paste("R-square:", round(x.s$adj.r.squared, 3), "\n"), file=r.file, append=TRUE)
    cat(paste("Fitted values sd:", signif(sd(t.m$learner.model$super.model$learner.model$fitted.values), 3), "\n"), file=r.file, append=TRUE)
    cat(paste("RMSE:", signif(sqrt(sum(t.m$learner.model$super.model$learner.model$residuals^2) / t.m$learner.model$super.model$learner.model$df.residual), 3), "\n\n"), file=r.file, append=TRUE)
    sink(file=r.file, append=TRUE, type="output")
    cat("EML model summary:", file=r.file, append=TRUE)
    print(x.s)
    cat("\n", file=r.file, append=TRUE)
    imp.rds = paste0(out.dir, "t.mrf.", t.var, ".rds")
    if(file.exists(imp.rds)){
      var.mod1 <- readRDS.gz(imp.rds)
      cat("Variable importance:\n", file=r.file, append=TRUE)
      xl <- as.data.frame(mlr::getFeatureImportance(var.mod1)$res)
      write.csv(xl[order(xl$importance, decreasing=TRUE),], paste0(out.dir, "rf_varImportance_", t.var, ".csv"))
      print(xl[order(xl$importance, decreasing=TRUE),][1:n.max,])
    }
    sink()
  }
}

pred_mlr <- function(tvar, year, t.m, eml.cf=1, g1km, in.dir="./mood1km/", out.dir="./pred/", multiplier=10, min.md=1){
  out.tif = paste0(out.dir, tvar, "_M_", year, "_1km_mood.tif")
  if(any(!file.exists(out.tif))){
    require("mlr")
    #require("randomForestSRC")
    #require("xgboost")
    require("nnet")
    gc()
    retry::wait_until( memuse::Sys.meminfo()[[2]]@size > 10 , timeout = 3600)
    if(any(t.m$features %in% "CRP")){ g1km$CRP <- readGDAL(list.files(paste0(in.dir, "CRP"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "FOR")){ g1km$FOR <- readGDAL(list.files(paste0(in.dir, "FOR"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "HFP")){ g1km$HFP <- readGDAL(list.files(paste0(in.dir, "HFP"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "LST")){ g1km$LST <- readGDAL(list.files(paste0(in.dir, "LST"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "N02")){ g1km$N02 <- readGDAL(list.files(paste0(in.dir, "N02"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "N08")){ g1km$N08 <- readGDAL(list.files(paste0(in.dir, "N08"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "NLT")){ g1km$NLT <- readGDAL(list.files(paste0(in.dir, "NLT"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "PPD")){ g1km$PPD <- readGDAL(list.files(paste0(in.dir, "PPD"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "PRE")){ g1km$PRE <- readGDAL(list.files(paste0(in.dir, "PRE"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "SNW")){ g1km$SNW <- readGDAL(list.files(paste0(in.dir, "SNW"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "T02")){ g1km$T02 <- readGDAL(list.files(paste0(in.dir, "T02"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "T08")){ g1km$T08 <- readGDAL(list.files(paste0(in.dir, "T08"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    g1km$Year = year
    #x = t.m$features[which(!t.m$features %in% names(g1km))]
    cc = complete.cases(g1km@data)
    g1kx = g1km[1]
    if(sum(cc)>0){
      itr = plyrChunks(which(cc), n=round(length(cc)/parallel::detectCores()))
      pred <- unlist(parallel::mclapply(itr, function(i){ predict(t.m, newdata=g1km@data[i,t.m$features])$data$response }, mc.cores = parallel::detectCores()))
      g1kx@data[cc,"pred"] = ifelse(pred < 0, 0, pred)*multiplier
      rgdal::writeGDAL(g1kx["pred"], out.tif, type="Byte", mvFlag=255, options="COMPRESS=DEFLATE")
      out.p <- do.call(rbind, parallel::mclapply(itr, function(i){ as.matrix(as.data.frame(mlr::getStackedBaseLearnerPredictions(t.m, newdata=g1km@data[i,t.m$features]))) }, mc.cores = parallel::detectCores()))
      #pred = predict(t.m, newdata=g1km@data[cc,])
      #out.p <- as.matrix(as.data.frame(mlr::getStackedBaseLearnerPredictions(t.m, newdata=g1km[cc,]))) * multiplier
      g1kx@data[cc,"model.error"] <- sqrt(matrixStats::rowSds(out.p, na.rm=TRUE)^2 * eml.cf)
      g1kx$model.error <- ifelse(g1kx$model.error<min.md, min.md, g1kx$model.error)
      rgdal::writeGDAL(g1kx["model.error"], gsub("_M_", "_md_", out.tif), type="Byte", mvFlag=255, options="COMPRESS=DEFLATE")
      rm(out.p); rm(pred); gc()
    }
  }
}

## Predict tiles generated using "train_sp_eml"
predict_mlr_prob <- function(tvar, year, t.m, g1km, eml.cf, in.dir="./mood1km/", out.dir="./probs/", multiplier=100, min.md=1){
  out.tif = paste0(out.dir, tvar, "_M_", year, "_1km_mood.tif")
  if(any(!file.exists(out.tif))){
    require("mlr"); require("xgboost"); require("glmnet"); require("ranger"); require("stats")
    gc()
    retry::wait_until( memuse::Sys.meminfo()[[2]]@size > 10 , timeout = 3600)
    if(any(t.m$features %in% "CRP")){ g1km$CRP <- readGDAL(list.files(paste0(in.dir, "CRP"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "FOR")){ g1km$FOR <- readGDAL(list.files(paste0(in.dir, "FOR"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "HFP")){ g1km$HFP <- readGDAL(list.files(paste0(in.dir, "HFP"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "LST")){ g1km$LST <- readGDAL(list.files(paste0(in.dir, "LST"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "N02")){ g1km$N02 <- readGDAL(list.files(paste0(in.dir, "N02"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "N08")){ g1km$N08 <- readGDAL(list.files(paste0(in.dir, "N08"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "NLT")){ g1km$NLT <- readGDAL(list.files(paste0(in.dir, "NLT"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "PPD")){ g1km$PPD <- readGDAL(list.files(paste0(in.dir, "PPD"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    g1km$PPD <- filter_na(g1km["PPD"])
    if(any(t.m$features %in% "PRE")){ g1km$PRE <- readGDAL(list.files(paste0(in.dir, "PRE"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "SNW")){ g1km$SNW <- readGDAL(list.files(paste0(in.dir, "SNW"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "T02")){ g1km$T02 <- readGDAL(list.files(paste0(in.dir, "T02"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    if(any(t.m$features %in% "T08")){ g1km$T08 <- readGDAL(list.files(paste0(in.dir, "T08"), pattern=glob2rx(paste0("*", year, "*.tif$")), full.names = TRUE))$band1 }
    g1km$Year = year
    #x = t.m$features[which(!t.m$features %in% names(g1km))]
    #na.lst = sapply(g1km@data, function(i){sum(is.na(i))})
    cc = complete.cases(g1km@data)
    g1kx = g1km[1]
    if(sum(cc)>0){
      itr = plyrChunks(which(cc), n=round(length(cc)/parallel::detectCores()))
      pred <- unlist(parallel::mclapply(itr, function(i){ predict(t.m, newdata=g1km@data[i,t.m$features])$data$prob.1 }, mc.cores = parallel::detectCores()))
      g1kx@data[cc,"pred"] = round(pred * multiplier)
      rgdal::writeGDAL(g1kx["pred"], out.tif, type="Byte", mvFlag=255, options="COMPRESS=DEFLATE")
      out.p <- do.call(rbind, parallel::mclapply(itr, function(i){ as.matrix(as.data.frame(mlr::getStackedBaseLearnerPredictions(t.m, newdata=g1km@data[i,t.m$features]))) }, mc.cores = parallel::detectCores()))
      #g1kx@data[cc,"model.error"] <- sqrt(matrixStats::rowSds(out.p*multiplier, na.rm=TRUE)^2 * eml.cf)
      g1kx@data[cc,"model.error"] = matrixStats::rowSds(out.p*multiplier, na.rm=TRUE)
      g1kx$model.error <- ifelse(g1kx$model.error<min.md, min.md, g1kx$model.error)
      rgdal::writeGDAL(g1kx["model.error"], gsub("_M_", "_md_", out.tif), type="Byte", mvFlag=255, options="COMPRESS=DEFLATE")
      rm(out.p); rm(pred); gc()
    }
  }
}
