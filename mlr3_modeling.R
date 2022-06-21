#######################################################
###### r environment preparation
#######################################################

#setwd('/home/leandro/Code/spatial-prediction-eml/')

ls <- c("dplyr", "mlr3", "mlr3learners", "mlr3pipelines", "mlr3filters", 
        "mlr3extralearners", "mlr3fselect", "mlr3misc", "paradox", 
        "mlr3tuning", "paradox", "igraph")
new.packages <- ls[!(ls %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(ls, require, character.only = TRUE)

source('./mlr3_functions.R')

#######################################################
###### Input data preparation
#######################################################

all.data = readRDS("./input/regmatrix_aedes_1km.rds")

target.name = 'occurrence'
group.name = 'ID'
seed = 1989

all.data[target.name] = as.factor(ifelse(all.data$individualCount>0, 1, 0))
print(dim(all.data))

split = split_by_blocking(all.data, group.name, pct = 0.5, seed = seed)

train.data = split$train
print(dim(train.data))

test.data =  split$test
print(dim(test.data))

# Checking if the groups
# are complementary
#print(paste0(
#  length(unique(train.data$ID)), ' ',
#  length(unique(test.data$ID)), ' ',
#  length(unique(all.data$ID))
#))

v1km.names = names(train.data)

feat.names = c(
  v1km.names[grepl('*_1km_*', v1km.names)],  
  c("CRP", "FOR", "HFP", "LST", "N02", "N08", "NLT", "PPD", "PRE", "SNW", "T02", "T08")
)

mdl.names = c(target.name, group.name)
train.data = na.omit(train.data[c(feat.names, mdl.names)])

set.seed(seed)
train.rows = sample(nrow(train.data), 3000)
train.data = train.data[train.rows,]
print(dim(train.data))

ncores = parallel::detectCores()

#######################################################
###### mlr3Pipeline
#######################################################

stacked.graph = gunion(
  list(
    po("learner_cv", lrn("classif.glmnet", predict_type = "prob")),
    po("learner_cv", lrn("classif.xgboost", predict_type = "prob", nthread = ncores)),
    po("learner_cv", lrn('classif.ranger', predict_type = "prob", importance = "impurity", num.threads = ncores))
  )) %>>%
  po("featureunion") %>>%
  lrn("classif.log_reg", predict_type = "prob")

stacked.graph$plot(html = FALSE)
stacked.learner = as_learner(stacked.graph)

# Single learner example
# to be on train_mlr3_model method
# single.learner = lrn('classif.ranger', predict_type = "prob", importance = "impurity", num.threads = ncores)

#######################################################
###### Feature selection learner
#######################################################

fs.learner = lrn('classif.ranger', num.trees=60,  predict_type = "prob",  num.threads = ncores)

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
  classif.xgboost.colsample_bytree = p_dbl(0.5, 0.6),
  classif.glmnet.alpha = p_dbl(0, 1),
  classif.glmnet.lambda = p_dbl(50, 150)
)

# Search space for a single learner
# to be on train_mlr3_model method
#single.search_space = ps(
#  num.trees =  p_int(lower = 30, upper = 60)
#)

#######################################################
###### Modeling execution
#######################################################

fs.selector = fs(
  "genetic_search", 
  popSize = length(feat.names), 
  elitism = round(length(feat.names) * 0.8), 
  mutationChance = 0.1
)

result = train_mlr3_model(
  in.data = train.data,
  target = target.name,
  learner = stacked.learner,
  measure=msr('classif.logloss'),
  fs.selector = fs.selector,
  fs.resampling = rsmp("cv", folds = 5),
  fs.terminator = trm("run_time", secs = 120),
  fs.learner = fs.learner,
  
  tnr.search_space = stacked.search_space,
  tnr.subsample.pct = 0.10,
  tnr.terminator = trm("evals", n_evals = 10),
  tnr.resampling = rsmp("cv", folds = 5),
  tnr.tuner = tnr("random_search"),
  
  val.resampling = rsmp("cv", folds = 5),
  
  group.blocking = group.name,
  
  mlr_log_level = 'fatal', # fatal, error, warn, info, debug, trace
  verbose = TRUE,
  seed = seed
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
rf.var.imp[order(-rf.var.imp$rf.importance),, drop=FALSE]

#######################################################
###### Prediction / Generalization
#######################################################

new.data = na.omit(test.data[,c(feat.names,target.name)])
print(dim(new.data))

test.pred.prob = predict(result$learner, new.data[,feat.names], predict_type = 'prob')[,'1']
test.pred.truth = as.numeric(as.character(new.data[, 'occurrence']))

verification::roc.plot(test.pred.truth, test.pred.prob, show.thres=F, plot.thres=T)
verification::roc.area(test.pred.truth, test.pred.prob)
mlr3measures::confusion_matrix(result$cv_pedictions$truth, result$cv_pedictions$response, positive='1')

#######################################################
###### Saving output
#######################################################

write.csv(rf.var.imp, './output/mlr3.eml.var.imp.csv')
saveRDS(result, './output/mlr3.eml.model.rds')
