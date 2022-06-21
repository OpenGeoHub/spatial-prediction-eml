library("dplyr")
library("mlr3")
library("mlr3learners")
library("mlr3pipelines")
library("mlr3filters")
library("mlr3extralearners")
library("mlr3fselect")
library("mlr3misc")
library("paradox")
library("mlr3tuning")
library("paradox")
library("igraph")

ncores = parallel::detectCores()

split_by_blocking = function(
  df,
  group.name,
  pct = 0.5,
  seed = 1989
) {

  set.seed(seed)
  uniq.group = unique(df[,group.name])
  train.groups = uniq.group[ sample(length(uniq.group), round(length(uniq.group) * pct) )]

  return(list(
    train = filter(df, df[,group.name] %in% train.groups),
    test = filter(df, !df[,group.name] %in% train.groups)
  ))

}

train_mlr3_model = function(
  in.data, 
  target, 
  learner, 
  measure,
  tnr.search_space, 
  fs.selector = fs("random_search"),
  fs.resampling = rsmp("cv", folds = 5),
  fs.terminator = trm("evals", n_evals = 10),
  fs.learner = lrn('classif.ranger', num.trees=40, importance = "impurity", num.threads = ncores),
  tnr.subsample.pct = 0.1,
  tnr.terminator = trm("evals", n_evals = 10),
  tnr.resampling = rsmp("cv", folds = 5),
  tnr.tuner = tnr("random_search"),
  val.resampling = rsmp("cv", folds = 5),
  mlr_log_level = 'info',
  group.blocking = '',
  seed = 1989,
  verbose = TRUE
) 
{
  
  task_type <- learner$task_type
  id = paste0(learner$id, '_', target)
  tnr.subsample.size = as.integer(nrow(in.data) * tnr.subsample.pct)
  
  lgr::get_logger("mlr3")$set_threshold(mlr_log_level)
  lgr::get_logger("bbotk")$set_threshold(mlr_log_level)
  
  tnr.subsample.rows <- sample(nrow(in.data), tnr.subsample.size)
  
  ########################################################
  #### Feature selection
  ########################################################
  
  if (task_type == 'regr') {
    fs.task <- TaskRegr$new(target, in.data, target = target)
  } else {
    fs.task <- TaskClassif$new(target, in.data, target = target, positive ="1")
  }
  
  if (group.blocking != '') {
    fs.task$set_col_roles(group.blocking, roles = "group")
  }
  
  if (verbose) print(paste0(Sys.time(), ' - ', 'Finding the most important covariates using ', nrow(in.data), ' samples'))
  set.seed(seed)

  afs = AutoFSelector$new(
    learner = fs.learner,
    resampling = fs.resampling, 
    measure = measure,
    terminator = fs.terminator,
    fselector = fs.selector,
    store_fselect_instance = TRUE
  )
  
  invisible(capture.output({
    afs$train(fs.task)
  }))
  #return(fs.result)
  fs.result = afs$fselect_result$features[[1]]
  
  ########################################################
  #### Hyper-parameter optimization
  ########################################################
  
  sub.names = c(fs.result, c(target, group.blocking))
  
  if (verbose) print(paste0(Sys.time(), ' - ', length(fs.result), ' covariates selected from ', (ncol(in.data) - 2)))
  
  # The attempt to use TaskSupervised does not work
  if (task_type == 'regr') {
    tnr.task <- TaskRegr$new(target, in.data[tnr.subsample.rows, sub.names], target = target)
    val.task <- TaskRegr$new(target, in.data[,sub.names], target = target)
    fin.task <- TaskRegr$new(target, in.data[,sub.names], target = target)
  } else {
    tnr.task <- TaskClassif$new(target, in.data[tnr.subsample.rows, sub.names], target = target, positive ="1")
    val.task <- TaskClassif$new(target, in.data[,sub.names], target = target, positive ="1")
    fin.task <- TaskClassif$new(target, in.data[,sub.names], target = target, positive ="1")
  }
  
  if (group.blocking != '') {
    tnr.task$set_col_roles(group.blocking, roles = "group")
    val.task$set_col_roles(group.blocking, roles = "group")
    fin.task$set_col_roles(group.blocking, roles = "group")
  }
  
  if (verbose) print(paste0(Sys.time(), ' - ', 'Tunning hyperparameters using ', length(tnr.subsample.rows), ' samples'))
  set.seed(seed)

  instance = TuningInstanceSingleCrit$new(
    task = tnr.task,
    learner = learner$clone(),
    resampling = tnr.resampling,
    measure = measure,
    search_space = tnr.search_space,
    terminator = tnr.terminator
  )
  
  invisible(capture.output({
    tnr.tuner$optimize(instance)
  }))
  
  if (verbose) print(paste0(Sys.time(), ' - ', 'Validating model'))
  set.seed(seed)

  bst.learner = learner$clone()
  bst.learner$param_set$values = instance$result$learner_param_vals[[1]]

  invisible(capture.output({
    cv_pedictions = resample(
      task = val.task,
      learner = bst.learner,
      resampling = val.resampling,
    )$prediction()
  }))
  
  if (verbose) print(paste0(Sys.time(), ' - ', 'Fiting the final model'))
  set.seed(seed)

  bst.learner = learner$clone()
  bst.learner$param_set$values = instance$result$learner_param_vals[[1]]
  invisible(capture.output({
    bst.learner$train(fin.task) 
  }))
  
  return(list(
    learner = bst.learner,
    fs_result = fs.result,
    cv_pedictions = as.data.table(cv_pedictions)
  ))
  
}