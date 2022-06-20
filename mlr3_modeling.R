setwd('/home/leandro/Code/spatial-prediction-eml/')
source('./mlr3_functions.R')

#######################################################
###### Input data preparation
#######################################################

rm.all = readRDS("./input/regmatrix_aedes_1km.rds")
dim(rm.all)

group.name = 'ID'
target.name = 'occurrence'
rm.all[target.name] = as.factor(ifelse(rm.all$individualCount>0, 1, 0))
rm.names = names(rm.all)

feat.names = c(
  rm.names[grepl('*_1km_*', rm.names)],  
  c("CRP", "FOR", "HFP", "LST", "N02", "N08", "NLT", "PPD", "PRE", "SNW", "T02", "T08")
)

mdl.names = c(target.name, group.name)
rm.data = na.omit(rm.all[c(feat.names, mdl.names)])
rm.data = rm.data[sample(nrow(rm.data), 1000),]

ncores = parallel::detectCores()

#######################################################
###### mlr3Pipeline
#######################################################

eml.graph = gunion(
  list(
    po("learner_cv", lrn("classif.glmnet", predict_type = "prob")),
    po("learner_cv", lrn("classif.xgboost", predict_type = "prob", nthread = ncores)),
    po("learner_cv", lrn('classif.ranger', predict_type = "prob", importance = "impurity", num.threads = ncores))
  )) %>>%
  po("featureunion") %>>%
  lrn("classif.log_reg", predict_type = "prob")

eml.graph$plot(html = FALSE)

eml.learner = as_learner(eml.graph)

#######################################################
###### Feature selection learner
#######################################################

afs.learner = lrn('classif.ranger', num.trees=40, importance = "impurity",  predict_type = "prob",  num.threads = ncores)

#######################################################
###### Hyper-parameter search space
#######################################################

stacked.search_space = ps(
  classif.ranger.num.trees =  p_int(lower = 30, upper = 60),
  classif.xgboost.nrounds = p_int(10, 20),
  classif.xgboost.max_depth = p_int(4, 10),
  classif.xgboost.eta = p_dbl(0.2, 0.4),
  classif.xgboost.subsample = p_dbl(0.9, 1),
  classif.xgboost.min_child_weight = p_int(1, 4),
  classif.xgboost.colsample_bytree = p_dbl(0.5, 0.6)
)

#######################################################
###### Modeling execution
#######################################################

result = train_mlr3_model(
  in.data = rm.data,
  target = target.name,
  learner = eml.learner,
  search_space = stacked.search_space,
  group.blocking = group.name,
  measure=msr('classif.logloss'),
  terminator = trm("evals", n_evals = 20),
  resampling = rsmp("cv", folds = 5),
  tuner = tnr("random_search"),
  fselector = fs("genetic_search"),
  afs.learner = lrn('classif.ranger', num.trees=40, importance = "impurity",  predict_type = "prob"),
  mlr_log_level = 'fatal',
  verbose = TRUE,
  subsample.pct = 0.10
)

#######################################################
###### Modeling result
#######################################################

summary(result$learner$model$classif.log_reg$model)

truth.prob = as.numeric(as.character(result$cv_pedictions$truth))
pred.prob = result$cv_pedictions$prob.1

verification::roc.plot(truth.prob, pred.prob, show.thres=F, plot.thres=T)
verification::roc.area(truth.prob, pred.prob)
mlr3measures::confusion_matrix(result$cv_pedictions$truth, result$cv_pedictions$response, positive='1')

rf.importance = result$learner$model$classif.ranger$model$variable.importance
rf.var.imp = as.data.frame(rf.importance, col.names=list('importance'))

#######################################################
###### Saving output
#######################################################

write.csv(rf.var.imp, './output/mlr3.eml.var.imp.csv')
saveRDS(result, './output/mlr3.eml.model.rds')