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
ngs.m <- read.csv(gzfile("./input/ds_801_all.csv.gz"))
ngs.m$olc_id = olctools::encode_olc(ngs.m$latitude, ngs.m$longitude, 11)
## Spatial overlay ----
sel.pnts = !duplicated(ngs.m$olc_id)
summary(sel.pnts)
#>    Mode   FALSE    TRUE 
#> logical    9714    4857
ngs.m.pnts = ngs.m[which(sel.pnts), c("olc_id", "longitude", "latitude")]
coordinates(ngs.m.pnts) <- ~ longitude + latitude
proj4string(ngs.m.pnts) <- "+init=epsg:4326"
prj = "+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
ngs.xy = spTransform(ngs.m.pnts, prj)
```


```r
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
#> [18:20:28] WARNING: amalgamation/../src/objective/regression_obj.cu:170: reg:linear is now deprecated in favor of reg:squarederror.
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

## Modeling multiple geochemicals using a single ML model

The package **randomForestSRC** [@Ishwaran2021] provides functionality to run so-called 
[“Multivariate Outcomes”](https://www.randomforestsrc.org/articles/getstarted.html#multivariate-outcomes) which means that one can fit a model with multivariate outcomes (one model to predict multiple target variables). 
In the case of geochemical data, usually multitude of geochemicals (50+) are analyzed in lab, so that 
it seems more practical to fit a single model to predict p+ target variables (outcomes) all at once, than 
to fit p+ models instead. 

In the case of `ds801` dataset we can first specify number of target variables that we would like to map:


```r
tvs = c("as_ppm", "al_pct", "cd_ppm", "cr_ppm", "cu_ppm", "fe_pct",
        "mo_ppm", "ni_ppm", "se_ppm", "th_ppm", "zn_ppm", "pb_ppm")
## number of missing values per variable:
sapply(tvs, function(i){sum(is.na(ds801[,i]))})
#> as_ppm al_pct cd_ppm cr_ppm cu_ppm fe_pct mo_ppm ni_ppm se_ppm th_ppm zn_ppm 
#>    198      4   4290     41     23     24     32     43   6906     19     31 
#> pb_ppm 
#>     11
```

These variables are obviously highly cross-correlated:

<div class="figure" style="text-align: center">
<img src="spatial-3D_files/figure-html/corPlot-1.png" alt="Correlation matrics for a selection of geochemicals. Number indicates correlation coefficient (0 to 100%). All variables seem to be positively correlated." width="100%" />
<p class="caption">(\#fig:corPlot)Correlation matrics for a selection of geochemicals. Number indicates correlation coefficient (0 to 100%). All variables seem to be positively correlated.</p>
</div>

Here especially `zn`, `cu`, `fe`, `al` and `ni` seem to be highly cross-correlated. 
The **multicolinearity** of these variables could be reduced by converting them e.g. 
to Principal Components [@Venables2002Springer]. Principal Component Analysis helps 
reduce overlap in data and can help with modeling. 
Before transferring the original values to Principal Components we can also filter out 
all missing values by replacing them with a mean value (for scaled variable this is 0):


```r
ds801.x[is.na(ds801.x)] <- 0
ds801.xd <- as.data.frame(ds801.x)
pcs.ds801 <- stats::prcomp(x = ds801.xd)
```

<div class="figure" style="text-align: center">
<img src="spatial-3D_files/figure-html/biplot-1.png" alt="Results of PCA in a biplot." width="90%" />
<p class="caption">(\#fig:biplot)Results of PCA in a biplot.</p>
</div>

This again shows that most of geochemicals are positively correlated with about 
6 groups of most correlated variables.

Note that, in the previous step, we convert all log-normal or percent variables 
using `log1p` and/or logit transformation to: (1) produce close-to-normal distribution 
and (2) avoid producing predictions beyond the physical range of data (e.g. negative values or values >100%).

First 6 components explain majority of variance in data, but it appears that to be on a safe-side, 
we can model up to PC11:


```r
summary(pcs.ds801)
#> Importance of components:
#>                           PC1    PC2     PC3     PC4     PC5     PC6     PC7
#> Standard deviation     2.4221 1.1343 0.98982 0.79734 0.74930 0.67263 0.63508
#> Proportion of Variance 0.5244 0.1150 0.08759 0.05684 0.05019 0.04045 0.03606
#> Cumulative Proportion  0.5244 0.6395 0.72706 0.78390 0.83409 0.87454 0.91060
#>                            PC8     PC9    PC10    PC11    PC12
#> Standard deviation     0.58353 0.50063 0.41659 0.37258 0.31070
#> Proportion of Variance 0.03044 0.02241 0.01551 0.01241 0.00863
#> Cumulative Proportion  0.94104 0.96344 0.97896 0.99137 1.00000
```

As a rule of thumb, we advise to focus on components that explain >95% of variance in the data; 
to be certain one can also use >99% as the threshold.

Note the steps we have taken so far include: (1) we first transform the target variables to achieve close to normal-distribution, 
(2) we scale all numbers to have 0 = mean and 1 = standard deviation, (3) we replace missing value with the 
mean value (0 in this case), (4) we derive Principle Components with all data so nothing is lost.

Next, we can fit a model using the randomForestSRC package and all PCs expect for the last PC (which is 
usually pure noise):


```r
nComp = 11; Ntree = 85
xdta = ds801[,pr.vars]
ydta = pcs.ds801$x[,1:nComp]
tfm = get.mv.formula(colnames(ydta))
## spatial tiles 30x30km
sp.tiles = unique(ds801$ID)
## we define own custom bootstrapping principle by  
## organizing training points using spatial blocks;
samp = randomForestSRC:::make.sample(ntree=Ntree, samp.size=length(sp.tiles), 
                                     boot.size=length(sp.tiles)/2)
samp.df = plyr::join(ds801["ID"], cbind(data.frame(ID=sp.tiles), data.frame(samp)), type="left")
#> Joining by: ID
samp.df = as.matrix(samp.df[,-1])
## sampsize must be identical for each tree
n.col = min(colSums(samp.df))
samp.dfa = apply(X = samp.df, MARGIN = 2, FUN = function(i){fill.null(i, n.col)})
m.s1 = randomForestSRC::rfsrc(tfm, data.frame(ydta, xdta),
                              bootstrap = "by.user", 
                              samp = samp.dfa,
                              importance=TRUE)
print(m.s1, outcome.target = "PC1")
#>                          Sample size: 14275
#>                      Number of trees: 85
#>            Forest terminal node size: 5
#>        Average no. of terminal nodes: 1365.482
#> No. of variables tried at each split: 63
#>               Total no. of variables: 188
#>               Total no. of responses: 11
#>          User has requested response: PC1
#>        Resampling used to grow trees: by.user
#>     Resample size used to grow trees: 7109
#>                             Analysis: mRF-R
#>                               Family: regr+
#>                       Splitting rule: mv.mse *random*
#>        Number of random split points: 10
#>                 % variance explained: 53.49
#>                           Error rate: 2.73
```

This basically fits a multivariate RF of form `rfsrc(Multivar(y1, y2, ..., yd) ~ . , my.data, ...)`. 
The resulting model show that the models explain significant part of variance. As expected the significance of the model 
is usually the highest for the higher level components (PC1, PC2), then it gradually drops. 

We can access the variable importance of each individual model by using:


```r
vmp <- as.data.frame(get.mv.vimp(m.s1))["PC1"]
vmp$relative_importance = 100*vmp$PC1/sum(vmp$PC1)
vmp = vmp[order(vmp$relative_importance, decreasing = TRUE),]
vmp$variable = paste0(c(1:nrow(vmp)), ". ", row.names(vmp))
ggplot(data = vmp[1:20,], aes(x = reorder(variable, relative_importance), y = relative_importance)) +
  geom_bar(fill = "steelblue",
           stat = "identity") +
  coord_flip() +
  labs(title = "Variable importance",
       x = NULL,
       y = NULL) +
  theme_bw() + theme(text = element_text(size=15))
```

<div class="figure" style="text-align: center">
<img src="spatial-3D_files/figure-html/varimp-pc1-1.png" alt="Variable importance for 3D prediction model for PC1." width="90%" />
<p class="caption">(\#fig:varimp-pc1)Variable importance for 3D prediction model for PC1.</p>
</div>

## Predicting geochemicals using decomposition-composition method

To generate predictions for different depths and for different PCs, we can loop 
the `predict.rfsrc` function across different soil depths:


```r
for(j in 1:nComp){
 for(k in c(5, 30, 60)){
  out.tif = paste0("./output/pc", j, "_", k, "cm_1km.tif")
  if(!file.exists(out.tif)){
    g1km$hzn_depth = k
    sel.na = complete.cases(g1km)
    newdata = g1km[sel.na, pr.vars]
    pred = predict(m.s1, m.target = paste0("PC", j), newdata=newdata)
    g1km.sp = SpatialPixelsDataFrame(as.matrix(g1km[sel.na,c("x","y")]), 
                data=data.frame(pred$regrOutput[[paste0("PC", j)]][[1]]), 
                proj4string=CRS("EPSG:5070"))
    g1km.sp$pred = g1km.sp@data[,1]*100
    rgdal::writeGDAL(g1km.sp["pred"], out.tif, type="Int16", mvFlag=-32768, options=c("COMPRESS=DEFLATE"))
    #gc()
  }
 }
}
```

This gives predictions (in the scaled space) for each of the original target variables and for all depths. 
Note that, because the randomForestSRC package [uses by default all cores](https://www.randomforestsrc.org/articles/parallel.html) to run model training and prediction, 
there is no need to specify any further parallelization i.e. it is fine to run processing in a loop.

After we generate predictions, we can (1) first [back-transform values to original scale](https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com) using the PCA model eigenvectors, (2) scale-back from log-transformation. 
For example the predictions for top-soil, we first load all predicted PC's:


```r
ls.5cm = list.files('./output', pattern=glob2rx("^pc*_5cm_1km.tif$"), full.names = TRUE)
pred.tops = raster::stack(ls.5cm)
pred.tops = as(pred.tops, "SpatialGridDataFrame")
```

next, we [back-transform values](https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com) to the original scale by:


```r
Xhat = as.matrix(pred.tops@data[,paste0("pc", 1:nComp, "_5cm_1km")]/100) %*% t(pcs.ds801$rotation[,1:nComp])
## scale-back to the original mean / stdev:
Xhat.s = sweep(Xhat %*% diag(attr(ds801.x, 'scaled:scale')), 2, attr(ds801.x, 'scaled:center'), "+")
Xhat.s = as.data.frame(expm1( Xhat.s ))
names(Xhat.s) = tvs
```

We can plot two predictions of `Pb` (using EML produced in the previous example; and using PCA analysis) next to each other:


```r
pred.tops$Pb_5cm_1km.PCA = log1p(Xhat.s[,"pb_ppm"])
pred.tops$pb.pca_ppm_5cm_1km = pred.tops$Pb_5cm_1km.PCA * 100
#rgdal::writeGDAL(pred.tops["pb.pca_ppm_5cm_1km"], "./output/pb.pca_ppm_5cm_1km.tif", 
#                 type="Int16", mvFlag=-32768, options=c("COMPRESS=DEFLATE"))
pred.tops$Pb_5cm_1km.EML = log1p(readGDAL("./output/pb_ppm_5cm_1km.tif")$band1)
#> ./output/pb_ppm_5cm_1km.tif has GDAL driver GTiff 
#> and has 342 rows and 378 columns
r.pal = rev(rainbow(65)[1:48])
pred.tops.df = as.data.frame(pred.tops[c("Pb_5cm_1km.EML", "Pb_5cm_1km.PCA")])
all_df_long <- tidyr::gather(pred.tops.df, 'layer', 'value', -s1, -s2)
pred.pnts = as.data.frame(ngs.xy[!is.na(over(y=pred.tops[1], x=ngs.xy)[,1]),], xy=TRUE)
ggplot() + 
  geom_raster(data = all_df_long, aes(x=s1, y=s2, fill = value)) + 
  facet_wrap(. ~ layer, ncol = 2) + 
  geom_point(data = pred.pnts, aes(x=longitude, y=latitude), shape="+", size=2) +
  scale_fill_gradientn(colours=r.pal, name="", limits=c(2.8,3.9)) + 
  coord_equal() + 
  theme_minimal() + xlab("Westing")+ ylab("Northing")
```

<div class="figure" style="text-align: center">
<img src="spatial-3D_files/figure-html/pca-pred-1.png" alt="Predicted Pb concentration using ensemble ML (left) vs using the decomposition-composition PCA method (right)." width="100%" />
<p class="caption">(\#fig:pca-pred)Predicted Pb concentration using ensemble ML (left) vs using the decomposition-composition PCA method (right).</p>
</div>

This shows that the predictions are quite similar, except the PCA-method somewhat smooths out some very 
high values (Chicago city area). The difference also happens due to the fact that, 
for the map on the left, we have use an Ensemble model, while the map on the right is 
purely based on RF.

It seem to be also possible to derive prediction errors using the `quantreg` [function](https://www.randomforestsrc.org/reference/quantreg.rfsrc.html), although 
this can get at the order of magnitude more computational. In addition, for multiple 
predicted PCs one would need to derive a composite prediction error (usually a 
geometric mean of variances).

## Predicting geochemicals assuming No-human-influence

Using this integrated model for geochemicals, we could also map hypothetical distribution of the target geochemical 
**assuming no or marginal human influence** (NoHI). For this we can use the same model, 
but during prediction replace values of the covariates for those that quantify 
human impact. For example layers `travel_time_to_cities`, `population_density`, 
lights at night images, density of industrial complexes and proximity to roads and 
cities, are all variables that basically quantify human / industrial footprint [@mu2022global].
To predict values of geochemicals assuming no human influence i.e. to estimate the 
background or natural concentrations of geochemicals we can thus replace values 
for all such layers using the following simple principles [@sanderman2017soil]:

- for lights at night images and human population density we replace all values with 0 value;  
- for travel time to cities, buffer densities to industrial sources of pollution and similar
  we use some very high value e.g. 99% probability quantile value for the whole of USA;

We can thus re-make all predictions using NoHI settings:


```r
## only top-soil
k = 5
for(j in 1:nComp){
  out.tif = paste0("./output/pc", j, ".nohi_", k, "cm_1km.tif")
  if(!file.exists(out.tif)){
    g1km$hzn_depth = k
    sel.na = complete.cases(g1km)
    newdata = g1km[sel.na, pr.vars]
    ## replace values / simulate noHI:
    newdata[,grep("travel", pr.vars)] = 1000
    newdata[,grep("pop", pr.vars)] = 0
    #newdata[,grep("light", pr.vars)] = 0
    pred = predict(m.s1, m.target = paste0("PC", j), newdata=newdata)
    g1km.sp = SpatialPixelsDataFrame(as.matrix(g1km[sel.na,c("x","y")]), 
                data=data.frame(pred$regrOutput[[paste0("PC", j)]][[1]]), 
                proj4string=CRS("EPSG:5070"))
    g1km.sp$pred = g1km.sp@data[,1]*100
    rgdal::writeGDAL(g1km.sp["pred"], out.tif, type="Int16", mvFlag=-32768, options=c("COMPRESS=DEFLATE"))
    #gc()
  }
}
```

Again we need to back-transform the values to original scale:


```r
lsN.5cm = list.files('./output', pattern=glob2rx("^pc*.nohi_5cm_1km.tif$"), full.names = TRUE)
predN.tops = raster::stack(lsN.5cm)
predN.tops = as(predN.tops, "SpatialGridDataFrame")
XhatN = as.matrix(predN.tops@data[,paste0("pc", 1:nComp, ".nohi_5cm_1km")]/100) %*% t(pcs.ds801$rotation[,1:nComp])
XhatN.s = sweep(XhatN %*% diag(attr(ds801.x, 'scaled:scale')), 2, attr(ds801.x, 'scaled:center'), "+")
XhatN.s = as.data.frame(expm1( XhatN.s ))
names(XhatN.s) = tvs
```

We can plot the two predictions for e.g. Pb to see if there is any difference:


```r
pred.tops$Pb.nohi_5cm_1km.PCA = log1p(XhatN.s[,"pb_ppm"])
pred.tops$pb.pca.nohi_ppm_5cm_1km = pred.tops$Pb.nohi_5cm_1km.PCA * 100
#rgdal::writeGDAL(pred.tops["pb.pca.nohi_ppm_5cm_1km"], "./output/pb.pca.nohi_ppm_5cm_1km.tif", 
#                 type="Int16", mvFlag=-32768, options=c("COMPRESS=DEFLATE"))
predN.tops.df = as.data.frame(pred.tops[c("Pb_5cm_1km.PCA", "Pb.nohi_5cm_1km.PCA")])
allN_df_long <- tidyr::gather(predN.tops.df, 'layer', 'value', -s1, -s2)
ggplot() + 
  geom_raster(data = allN_df_long, aes(x=s1, y=s2, fill = value)) + 
  facet_wrap(. ~ layer, ncol = 2) + 
  geom_point(data = pred.pnts, aes(x=longitude, y=latitude), shape="+", size=2) +
  scale_fill_gradientn(colours=r.pal, name="", limits=c(2.8,3.9)) + 
  coord_equal() + 
  theme_minimal() + xlab("Westing")+ ylab("Northing")
```

<div class="figure" style="text-align: center">
<img src="spatial-3D_files/figure-html/nohi-pred-1.png" alt="Predicted Pb concentration using decomposition-composition PCA method with original covariates (left) and assuming no human influence (right)." width="100%" />
<p class="caption">(\#fig:nohi-pred)Predicted Pb concentration using decomposition-composition PCA method with original covariates (left) and assuming no human influence (right).</p>
</div>

This has significantly reduced the total values of Pb. Basically the map on the right 
is most likely showing the *background* or *natural* concentration of Pb assuming 
no human influence (notice the city of Chicago and similar are not visible on the map any more). 
The difference between the two maps is an estimate of human-induced pollution by Pb. 
If there are historic Pb soil measurements the map on the right could also be validated. 

Note that we have only *modified* values for travel time to cities, population density etc, 
also the MODIS LST night time temperatures should be probably adjusted since they 
also show big urban areas and similar. We have not done it here because converting 
MODIS LST images to no-human influence would probably require modeling in itself, but it is 
most likely doable.

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

By combining PCA analysis and predictive mapping (we refer to this method as the **Decomposition-composition PCA method**) 
one can integrate modeling of multiple cross-correlated variables and multicolinearity reduction (PCA analysis). 
This seems to be especially applicable for cases where there are ≫10 target variables which 
are highly cross-correlated, possibly have inconsistent missing values and after transformation 
match close-to-normal distribution.
As a rule of thumb one can model only PCs that explain up to 99% of variance in data, this way modeling 
effort can be significantly reduced (e.g. for 50+ target variables one could model all target variables 
only a dozen of PCs). A disadvantage of the Decomposition-composition PCA method is that interpretation of 
scaled values is somewhat more complex, certainly more abstract.
