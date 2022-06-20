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

train_mlr3_model = function(
  in.data, 
  target, 
  learner, 
  search_space, 
  measure,
  terminator = trm("evals", n_evals = 10),
  resampling = rsmp("cv", folds = 5),
  tuner = tnr("random_search"),
  fselector = fs("random_search"),
  afs.learner = lrn('classif.ranger', num.trees=40, importance = "impurity", num.threads = ncores),
  subsample.pct = 0.1, # Hyperparameter tunning
  mlr_log_level = 'info',
  group.blocking = '',
  verbose = TRUE
) 
{
  
  task_type <- learner$task_type
  id = paste0(learner$id, '_', target)
  sample_size = as.integer(nrow(in.data) * subsample.pct)
  
  lgr::get_logger("mlr3")$set_threshold(mlr_log_level)
  lgr::get_logger("bbotk")$set_threshold(mlr_log_level)
  
  sample.rows <- sample(nrow(in.data), sample_size)
  
  ########################################################
  #### Feature selection
  ########################################################
  
  if (task_type == 'regr') {
    task.afs <- TaskRegr$new(target, in.data, target = target)
  } else {
    task.afs <- TaskClassif$new(target, in.data, target = target, positive ="1")
  }
  
  if (group.blocking != '') {
    task.afs$set_col_roles(group.blocking, roles = "group")
  }
  
  if (verbose) print(paste0('Finding the most important covariates using ', nrow(in.data), ' samples'))
  
  afs = AutoFSelector$new(
    learner = afs.learner,
    resampling = resampling, 
    measure = measure,
    terminator = terminator,
    fselector = fselector,
    store_fselect_instance = TRUE
  )
  
  invisible(capture.output({
    afs$train(task.afs)
  }))
  #return(afs.result)
  afs.result = afs$fselect_result$features[[1]]
  
  ########################################################
  #### Hyper-parameter optimization
  ########################################################
  
  sub.names = c(afs.result, c(target, group.blocking))
  
  if (verbose) print(paste0(length(afs.result), ' covariates selected from ', (ncol(in.data) - 2)))
  
  # The attempt to use TaskSupervised does not work
  if (task_type == 'regr') {
    task.tnr <- TaskRegr$new(target, in.data[sample.rows, sub.names], target = target)
    task.rsp <- TaskRegr$new(target, in.data[,sub.names], target = target)
    task.bst <- TaskRegr$new(target, in.data[,sub.names], target = target)
  } else {
    task.tnr <- TaskClassif$new(target, in.data[sample.rows, sub.names], target = target, positive ="1")
    task.rsp <- TaskClassif$new(target, in.data[,sub.names], target = target, positive ="1")
    task.bst <- TaskClassif$new(target, in.data[,sub.names], target = target, positive ="1")
  }
  
  if (group.blocking != '') {
    task.tnr$set_col_roles(group.blocking, roles = "group")
    task.rsp$set_col_roles(group.blocking, roles = "group")
    task.bst$set_col_roles(group.blocking, roles = "group")
  }
  
  if (verbose) print(paste0('Tunning hyperparameters using ', length(sample.rows), ' samples'))
  
  instance = TuningInstanceSingleCrit$new(
    task = task.tnr,
    learner = learner$clone(),
    resampling = resampling,
    measure = measure,
    search_space = search_space,
    terminator = terminator
  )
  
  invisible(capture.output({
    tuner$optimize(instance)
  }))
  
  if (verbose) print(paste0('Running model evaluation'))
  best.learner = learner$clone()
  best.learner$param_set$values = instance$result$learner_param_vals[[1]]
  
  invisible(capture.output({
    cv_pedictions = resample(
      task = task.rsp,
      learner = best.learner,
      resampling = resampling,
    )$prediction()
  }))
  
  if (verbose) print('Fiting the final model')
  best.learner = learner$clone()
  best.learner$param_set$values = instance$result$learner_param_vals[[1]]
  invisible(capture.output({
    best.learner$train(task.bst) 
  }))
  
  return(list(
    learner = best.learner,
    fs_result = afs.result,
    cv_pedictions = as.data.table(cv_pedictions)
  ))
  
}