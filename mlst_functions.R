
## spacetime overlay
extract_st <- function(tif, df, date, date.tif.begin, date.tif.end, coords=c("x","y"), crs, format.date="%Y-%m-%d", variable.name){
  if(any(!coords %in% colnames(df))){
    stop(paste("Coordinate columns", coords, "could not be found"))
  }
  if(is.character(date) & length(date)==1 & date %in% colnames(df)){
    date = as.Date(df[,date], format=format.date, origin="1970-01-01")
  } else {
    stop(paste("Column name", date, "could not be found in the dataframe"))
  }
  if(missing(date.tif.end)){
    date.tif.end = date.tif.begin
  }
  sel <- date <= as.Date(date.tif.end, format=format.date, origin="1970-01-01") & date >= as.Date(date.tif.begin, format=format.date, origin="1970-01-01")
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

## Hyperparameter optimization for component learners of the ensemble model
tune_learners <- function(data, formula, blocking, out.dir="output/", predict.type = "prob", SL.library=c("classif.ranger","classif.xgboost","classif.glmnet"), parallel="multicore", rdesc = mlr::makeResampleDesc("CV", iters = 5L) , xg.skip = FALSE, num.trees = 85){
  
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
    ParamHelpers::makeDiscreteParam("nrounds", value=c(10)),
    ParamHelpers::makeDiscreteParam("max_depth", values=c(5,10)),
    ParamHelpers::makeDiscreteParam("eta", values=seq(0.5,1, by = 0.25)),
    ParamHelpers::makeDiscreteParam("subsample", value=c(1)),
    ParamHelpers::makeDiscreteParam("min_child_weight", values=c(5,10)),
    ParamHelpers::makeDiscreteParam("colsample_bytree", values=seq(0.5,1, by = 0.25))
  )
  
  ## mlr parameters
  ctrl = mlr::makeTuneControlGrid()
  message(paste0("Using learners: ", paste(SL.library, collapse = ", "), "..."), immediate. = TRUE)
  tsk <- mlr::makeClassifTask(data = df.s, 
                              target = tv, 
                              positive = "1",
                              blocking = blocking[which(r.sel)])
  
  out.rf = paste0(out.dir, "Fagus_sylvatica_rf_model.rds")
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
  out.x = paste0(out.dir, "Fagus_sylvatica_xgb_model.rds")
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
  out.glm = paste0(out.dir, "Fagus_sylvatica_glm_model.rds")
  if(!file.exists(out.glm)){
    parallelMap::parallelStartSocket(parallel::detectCores())
    var.mod3 = mlr::train(mlr::makeLearner("classif.glmnet", predict.type = predict.type), task = tsk)
    saveRDS(var.mod3, out.glm)
    parallelMap::parallelStop()
  } else {
    var.mod3 = readRDS(out.glm)
  }  
  finetuning = list(var.mod1, var.mod2, var.mod3, formula)
  saveRDS(finetuning, paste0(out.dir, "Fagus_sylvatica_finetune.rds"))
  return(finetuning)
}

## Train spacetime model for predicting species occurrences
train_sp_eml <- function(data, tune_result, blocking, out.dir="output/", predict.type = "prob", SL.library=c("classif.ranger","classif.xgboost","classif.glmnet"), super.learner = "classif.logreg", parallel="multicore", num.trees = 85, xyn = c("easting", "northing"), method = "stack.cv"){

  tv <- all.vars(tune_result[[4]][[2]])
  r.sel <- stats::complete.cases(data[,all.vars(tune_result[[4]])])
  df.s = data[which(r.sel),all.vars(tune_result[[4]])]
  l = sapply(df.s[,-c(1)], sd, na.rm = TRUE)
  blocking = blocking[which(r.sel)]
  
  ## get models
  var.mod1 = tune_result[[1]]
  var.mod2 = tune_result[[2]]
  var.mod3 = tune_result[[3]]
  
  out.eml = paste0(out.dir, "Fagus_sylvatica_eml.rds")
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

## Predict tiles generated using "train_sp_eml"
predict_tiles <- function(input, model, rds.dir="input/", out.dir="output/"){

  tile_id = unlist(strsplit(input, split = '[.]'))[1]
  year = unlist(strsplit(input, split = '[.]'))[2]
  print(paste0(tile_id, ' - reading the data'))
  out.files = list()
  tmp_folder = file.path(out.dir, tile_id)
  dir.create(tmp_folder, recursive =TRUE, showWarnings = FALSE)
  out.prob.file = paste0("tile_", tile_id, "_", year, ".tif")
  out.md.file = paste0("tile_", tile_id, "_md_", year, ".tif")
  out.prob = file.path(tmp_folder, out.prob.file)
  out.md = file.path(tmp_folder, out.md.file)
  static_data = try(readRDS(paste0(rds.dir, "tile_", tile_id, "_30m_static.rds")))
  yearly_data = try(readRDS(paste0(rds.dir, "tile_", tile_id, "_30m_", year, ".rds")))
  if(class(static_data) == "try-error" | class(yearly_data) == "try-error"){
    message("RDS ", tile_id, " is corrupted or does not exist, skipping to next RDS...", immediate. = TRUE)
    return(out.files)
  }
  yearly_data@data = cbind(yearly_data@data, static_data@data)
  print(paste0(tile_id, ' - running predictions'))
  probability_map = try(predict(model, newdata=yearly_data@data[,model$features]), silent=T)
  if(class(probability_map) == "try-error"){
    print(probability_map[1])
    message("RDS file ", tile_id, " has NA values, skipping to next RDS...", immediate. = TRUE)
    rm(static_data, yearly_data, out.prob, out.md, out.prob.file, out.md.file, probability_map)
    gc()
    return(out.files)
  }
  ## Get base learners predictions to compute standard deviation and variance
  pred = mlr::getStackedBaseLearnerPredictions(model, newdata=yearly_data@data[,model$features])
  ## setup correction factor
  m.train = model$learner.model$super.model$learner.model$data
  m.terms = model$learner.model$super.model$learner.model$terms
  eml.MSE0 = matrixStats::rowSds(as.matrix(m.train[,all.vars(m.terms)[-1]]), na.rm=TRUE)^2
  eml.MSE = deviance(model$learner.model$super.model$learner.model)/df.residual(model$learner.model$super.model$learner.model)
  ## mass-preservation of MSE
  eml.cf = eml.MSE/mean(eml.MSE0, na.rm = TRUE)
  rf.sd = sqrt(matrixStats::rowSds(as.matrix(as.data.frame(pred)*100), na.rm=TRUE)^2 * eml.cf)
  map <- SpatialPixelsDataFrame(yearly_data@coords, data=as.data.frame(probability_map$data$prob.1*100), grid=yearly_data@grid, proj4string=yearly_data@proj4string)
  map@data$md <- rf.sd
  colnames(map@data) = c("Prob", "Prd.err")
  out.files = append(out.files, out.prob)
  out.files = append(out.files, out.md)
  print(paste0(tile_id, ' - writing files'))
  writeGDAL(map["Prob"], out.prob, drivername="GTiff", type="Byte", mvFlag=0, options=c("COMPRESS=DEFLATE"))
  writeGDAL(map["Prd.err"], out.md, drivername="GTiff", type="Byte", mvFlag=0, options=c("COMPRESS=DEFLATE"))
  rm(static_data, yearly_data, out.prob, out.md, out.prob.file, out.md.file, probability_map)
  return(map)
}

