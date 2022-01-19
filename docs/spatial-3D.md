# Spatial interpolation in 3D using Ensemble ML

::: {.rmdnote}
You are reading the work-in-progress Spatial and spatiotemporal interpolation using Ensemble Machine Learning. This chapter is currently draft version, a peer-review publication is pending. You can find the polished first edition at <https://opengeohub.github.io/spatial-prediction-eml/>.
:::



## Mapping concentrations of geochemical elements

Ensemble ML can also be used for mapping soil variables in 3D. Consider for example 
the Geochemical and minerological data set for USA48 [@smith2014geochemical]. This is a public data set 
produced and maintained by the [USA Geological Survey](https://mrdata.usgs.gov/ngdb/soil/) and contains laboratory measurements 
of chemical elements and minerals in soil at 4,857 sites (for three depths 0 to 5 cm, 
A horizon and C horizon; Fig. \@ref(fig:ds801-example)).

We have previously imported and overlaid the sampling points vs large stack of 
covariate layers using e.g.:


```r
## Training data ----
ngs.m <- read.csv("./training_data/ds_801_all.csv")
ngs.m$olc_id = olctools::encode_olc(ngs.m$latitude, ngs.m$longitude, 11)
## Spatial overlay ----
sel.pnts = !duplicated(ngs.m$olc_id)
summary(sel.pnts)
ngs.m.pnts = ngs.m[which(sel.pnts), c("olc_id", "longitude", "latitude")]
coordinates(ngs.m.pnts) <- ~ longitude + latitude
proj4string(ngs.m.pnts) <- "+init=epsg:4326"
ngs.xy = spTransform(ngs.m.pnts, crs(mask))
tif.lst = list.files("./1km", glob2rx("*.tif$"), full.names = TRUE)
## 262 layers
ov.tmp = parallel::mclapply(1:length(tif.lst), function(j){ terra::extract(terra::rast(tif.lst[j]), terra::vect(ngs.xy)) }, mc.cores = 80)
ov.tmp = dplyr::bind_cols(lapply(ov.tmp, function(i){i[,2]}))
names(ov.tmp) = tools::file_path_sans_ext(basename(tif.lst))
ov.tmp$olc_id = ngs.xy$olc_id
reg.matrix = plyr::join(ngs.m, ov.tmp)
saveRDS.gz(reg.matrix, "./input/ds801_geochem1km.rds")
```


We can load the regression matrix which contains all target variables and covariates:


```r
ds801 = readRDS("./input/ds801_geochem1km.rds")
dim(ds801)
#> [1] 14275   407
```

which has a total of 4818 unique locations:


```r
str(levels(as.factor(ds801$olc_id)))
#>  chr [1:4818] "75WXFVQG+WH6" "75WXJHVG+62X" "75WXJVM9+995" "75WXRG2G+5PV" ...
```

The individual records can be browsed directly via <https://mrdata.usgs.gov/ngdb/soil/>, 
for example a single record includes:

<div class="figure" style="text-align: center">
<img src="./img/Fig_geochemical_USGS_record.jpg" alt="Example of a geochemical sample with observations and measurements and coordiates of site." width="100%" />
<p class="caption">(\#fig:ds801-example)Example of a geochemical sample with observations and measurements and coordiates of site.</p>
</div>

Covariates prepared to help interpolation of the geochemicals include:

- Distance to cities [@nelson2019suite];  
- [MODIS LST](https://doi.org/10.5281/zenodo.1420114) (monthly daytime and nighttime);  
- [MODIS EVI](https://lpdaac.usgs.gov/products/mod13q1v006/) (long-term monthly values);  
- [Lights at night images](https://eogdata.mines.edu/products/dmsp/);  
- [Snow occurrence probability](https://climate.esa.int/en/odp/#/project/snow);  
- Soil property and class maps for USA48 [@ramcharan2018soil];  
- Terrain / hydrological indices;  

We can focus on predict concentration of lead (Pb). Pb seems to change 
with depth and this is seems to be consistent for different land cover classes:


```r
openair::scatterPlot(ds801[ds801$pb_ppm<140,], x = "hzn_depth", y = "pb_ppm", method = "hexbin", col = "increment", log.x=TRUE, log.y=TRUE, xlab="Depth", ylab="Pb [ppm]", z.lim=c(0,100), type="landcover1")
#> Warning: removing 11 missing rows due to landcover1
```

<div class="figure" style="text-align: center">
<img src="spatial-3D_files/figure-html/cor-depth-1.png" alt="Distribution of Pb as a function of land cover classes." width="100%" />
<p class="caption">(\#fig:cor-depth)Distribution of Pb as a function of land cover classes.</p>
</div>


Because we are aiming at producing predictions of geochemical elements for different 
depths in soil, we can also use depth of the soil sample as one of the covariates. 
This makes the prediction system 3D and the model can thus be used to predict at 
any new 3D location ($X, Y, d$). To fit a RF model for this data we can use:


```r
ds801$log.pb = log1p(ds801$pb_ppm)
pr.vars = c(readRDS("./input/pb.pr.vars.rds"), "hzn_depth")
sel.pb = complete.cases(ds801[,c("log.pb", pr.vars)])
mrf = ranger::ranger(y=ds801$log.pb[sel.pb], x=ds801[sel.pb, pr.vars], 
            num.trees = 85, importance = 'impurity')
mrf
#> Ranger result
#> 
#> Call:
#>  ranger::ranger(y = ds801$log.pb[sel.pb], x = ds801[sel.pb, pr.vars],      num.trees = 85, importance = "impurity") 
#> 
#> Type:                             Regression 
#> Number of trees:                  85 
#> Sample size:                      14264 
#> Number of independent variables:  188 
#> Mtry:                             13 
#> Target node size:                 5 
#> Variable importance mode:         impurity 
#> Splitrule:                        variance 
#> OOB prediction error (MSE):       0.09636483 
#> R squared (OOB):                  0.670695
```

which results in R-square of about 0.67. Because many training samples have exactly 
the same coordinates (same site, three depths), we assume that this model is 
over-fitting i.e. that the out-of-bag accuracy is [probably over-optimistic](https://opengeohub.github.io/spatial-sampling-ml/). 
Instead we can fit an Ensemble model where we block points within 30 by 30-km blocks:


```r
if(!exists("eml.pb")){
  lrn.rf = mlr::makeLearner("regr.ranger", num.trees=85, importance="impurity",
                            num.threads = parallel::detectCores())
  lrns.pb <- list(lrn.rf, mlr::makeLearner("regr.xgboost"), mlr::makeLearner("regr.cvglmnet"))
  tsk0.pb <- mlr::makeRegrTask(data = ds801[sel.pb, c("log.pb", pr.vars)], 
                               target = "log.pb", blocking = as.factor(ds801$ID[sel.pb]))
  init.pb <- mlr::makeStackedLearner(lrns.pb, method="stack.cv", super.learner="regr.lm", 
                                      resampling=mlr::makeResampleDesc(method="CV", blocking.cv=TRUE))
  parallelMap::parallelStartSocket(parallel::detectCores())
  eml.pb = train(init.pb, tsk0.pb)
  parallelMap::parallelStop()
}
#> [10:57:29] WARNING: amalgamation/../src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
```


```r
summary(eml.pb$learner.model$super.model$learner.model)
#> 
#> Call:
#> stats::lm(formula = f, data = d)
#> 
#> Residuals:
#>     Min      1Q  Median      3Q     Max 
#> -2.3979 -0.1990 -0.0117  0.1699  6.2948 
#> 
#> Coefficients:
#>               Estimate Std. Error t value Pr(>|t|)    
#> (Intercept)   -0.64745    0.04296 -15.069  < 2e-16 ***
#> regr.ranger    0.85552    0.02152  39.755  < 2e-16 ***
#> regr.xgboost   0.26550    0.05337   4.974 6.62e-07 ***
#> regr.cvglmnet  0.25631    0.02024  12.665  < 2e-16 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 0.4075 on 14260 degrees of freedom
#> Multiple R-squared:  0.4328,	Adjusted R-squared:  0.4327 
#> F-statistic:  3627 on 3 and 14260 DF,  p-value: < 2.2e-16
```

which shows somewhat lower R-square of 0.44, this time that whole sites have 
been taken out and hence this seems to be somewhat more realistic estimate of the 
mapping accuracy [@roberts2017cross]. The accuracy plot shows that the model has 
some problems with predicting higher values, but overall matches the observed values:


```r
t.pb = quantile(ds801$log.pb, c(0.001, 0.01, 0.999), na.rm=TRUE)
plot_hexbin(varn="log.pb", breaks=c(t.pb[1], seq(t.pb[2], t.pb[3], length=25)), 
      meas=eml.pb$learner.model$super.model$learner.model$model$log.pb, 
      pred=eml.pb$learner.model$super.model$learner.model$fitted.values,
      main="Pb [EML]")
```

<div class="figure" style="text-align: center">
<img src="./img/plot_CV_log.pb.png" alt="Accuracy plot for Pb concentration in soil fitted using Ensemble ML." width="90%" />
<p class="caption">(\#fig:ac-pb1)Accuracy plot for Pb concentration in soil fitted using Ensemble ML.</p>
</div>

Variables most important for explaining distribution of the target variable (based on the variable importance)
seem to be soil depth (`hnz_depth`), soil type maps, annual day time temperature and travel time to cities:


```r
library(ggplot2)
#> 
#> Attaching package: 'ggplot2'
#> The following object is masked from 'package:latticeExtra':
#> 
#>     layer
xl.pb <- as.data.frame(mlr::getFeatureImportance(eml.pb[["learner.model"]][["base.models"]][[1]])$res)
xl.pb$relative_importance = 100*xl.pb$importance/sum(xl.pb$importance)
xl.pb = xl.pb[order(xl.pb$relative_importance, decreasing = T),]
xl.pb$variable = paste0(c(1:nrow(xl.pb)), ". ", xl.pb$variable)
ggplot(data = xl.pb[1:20,], aes(x = reorder(variable, relative_importance), y = relative_importance)) +
  geom_bar(fill = "steelblue",
           stat = "identity") +
  coord_flip() +
  labs(title = "Variable importance",
       x = NULL,
       y = NULL) +
  theme_bw() + theme(text = element_text(size=15))
```

<div class="figure" style="text-align: center">
<img src="spatial-3D_files/figure-html/varimp-pb-1.png" alt="Variable importance for 3D prediction model for Pb concentrations." width="90%" />
<p class="caption">(\#fig:varimp-pb)Variable importance for 3D prediction model for Pb concentrations.</p>
</div>

If we plot the travel time to cities vs Pb concentrations, we can clearly see that 
Pb is negatively correlated with travel time to cities (following a log-log linear relationship):


```r
openair::scatterPlot(ds801[ds801$pb_ppm<140,], x = "travel_time_to_cities_1_usa48", y = "pb_ppm", method = "hexbin", col = "increment", log.x=TRUE, log.y=TRUE, xlab="Travel time to cities 1", ylab="Pb [ppm]", type="hzn_depth")
#> Warning: removing 11 missing rows due to hzn_depth
```

<div class="figure" style="text-align: center">
<img src="spatial-3D_files/figure-html/cor-cities-1.png" alt="Distribution of Pb as a function of travel time to cities for different depths." width="100%" />
<p class="caption">(\#fig:cor-cities)Distribution of Pb as a function of travel time to cities for different depths.</p>
</div>

## Predictions in 3D

To produce predictions we can focus on area around Chicago conglomeration. We can 
load the covariate layers by using:


```r
g1km = readRDS("./input/chicago_grid1km.rds")
```

This contains all layers we used for training. We can generate predictions by 
adding a depth column, then write predictions to GeoTIFFs:


```r
for(k in c(5, 30, 60)){
  out.tif = paste0("./output/pb_ppm_", k, "cm_1km.tif")
  if(!file.exists(out.tif)){
    g1km$hzn_depth = k
    sel.na = complete.cases(g1km)
    newdata = g1km[sel.na, eml.pb$features]
    pred = predict(eml.pb, newdata=newdata)
    g1km.sp = SpatialPixelsDataFrame(as.matrix(g1km[sel.na,c("x","y")]), 
                data=pred$data, proj4string=CRS("EPSG:5070"))
    g1km.sp$pred = expm1(g1km.sp$response)
    rgdal::writeGDAL(g1km.sp["pred"], out.tif, type="Int16", mvFlag=-32768, options=c("COMPRESS=DEFLATE"))
    #gc()
  }
}
```

This finally gives the following pattern:

<div class="figure" style="text-align: center">
<img src="./img/Pb_predictions_d1.jpg" alt="Predictions of Pb concentration for different soil depths based on Ensemble ML. Points indicate training points used to build the predictive mapping model. Red color indicates high values. Values of Pb clearly drop with soil depth." width="90%" /><img src="./img/Pb_predictions_d2.jpg" alt="Predictions of Pb concentration for different soil depths based on Ensemble ML. Points indicate training points used to build the predictive mapping model. Red color indicates high values. Values of Pb clearly drop with soil depth." width="90%" /><img src="./img/Pb_predictions_d3.jpg" alt="Predictions of Pb concentration for different soil depths based on Ensemble ML. Points indicate training points used to build the predictive mapping model. Red color indicates high values. Values of Pb clearly drop with soil depth." width="90%" />
<p class="caption">(\#fig:pred-pb1)Predictions of Pb concentration for different soil depths based on Ensemble ML. Points indicate training points used to build the predictive mapping model. Red color indicates high values. Values of Pb clearly drop with soil depth.</p>
</div>

Based on these results, it can be said that in general:

1. Distribution of Pb across USA seem to be controlled by a mixture of factors 
   including climatic factors, soil-terrain units and anthropogenic factors (travel distance to cities);  
2. Overall big urban areas show significantly higher concentrations for some heavy 
   metals and this relationship if log-log linear;  
3. Soil depth for some geochemical elements comes as the overall most important 
   covariate hence mapping soil variables in 3D is fully justified;


## Advantages and limitations of running 3D predictive mapping

In summary 3D soil mapping is relatively straight forward to implement especially 
for mapping soil variables from soil profiles (multiple samples per soil layer). 
Before modeling the target variable with depth, it is a good idea to plot 
relationship between target variable and depth under different settings 
(as in Fig. \@ref(fig:cor-depth)). 3D soil mapping based on Machine Learning is 
now increasingly common [@hengl2019predictive; @sothe2022large].

A limitation for 3D predictive mapping is the size of data i.e. data volumes 
increasing proportionally to number of slices we need to predict (in this case three). 
Also, we show that points with exactly the same coordinates might result in e.g. 
Random Forest overfitting emphasizing some covariate layers that are possibly 
less important, hence it is again important to use the blocking parameter that 
separates training and validation points.

Soil depth is for many soil variables most important explanatory variables, but 
in the cases it does not correlate with the target variable, there is probably 
also no need for 3D soil mapping. In the case depth is not significantly 
correlated, one could simply first aggregate all values to fixed block depth and 
convert modeling from 3D to 2D.
