# Summary

::: {.rmdnote}
You are reading the work-in-progress Spatial and spatiotemporal interpolation using Ensemble Machine Learning. This chapter is currently currently draft version, a peer-review publication is pending. You can find the polished first edition at <https://opengeohub.github.io/spatial-prediction-eml/>.
:::



The tutorial above demonstrates how to use Ensemble Machine Learning for predictive 
mapping going from numeric 2D, to factor and to 3D variables. Have in mind that 
the examples shown are based on relatively small datasets, but can still become 
computational if you add even more learners. In principle we do not recommend:
  
- adding learners that are significantly less accurate than your best learners (i.e. focus on the top 4–5 best performing learners),  
- fitting EML for <50–100 training points,
- fitting EML for spatial interpolation where points are heavily spatially clustered,
- using landmap package with large datasets,

For derivation of prediction error and prediction interval we recommend using the 
method of @lu2021unified. This method by default produces three measures of uncertainty:
  
- Root Mean Square Prediction Error (RMSPE) = the estimated conditional mean squared prediction errors of the random forest predictions,  
- bias = the estimated conditional biases of the random forest predictions,  
- lower and upper bounds / prediction intervals for a given probability e.g. 0.05 for the 95% probability interval,  

You can also follow an introduction to Ensemble Machine Learning from the [Open Data Science Europe workshop video recordings](https://av.tib.eu/series/1146/opendatascience+europe+workshop+2021).

Please note that the mlr package is discontinued, so some of the example above might become unstable with time. We are working on migrating the code in the landmap package to make the `train.spLearner` function work with the new [mlr3 package](https://mlr3.mlr-org.com/).

If you have a dataset that you have used to test Ensemble Machine Learning, please come back to us and share your experiences by posting [an issue](https://github.com/Envirometrix/landmap/issues) and/or providing a screenshot of your results.



# References
