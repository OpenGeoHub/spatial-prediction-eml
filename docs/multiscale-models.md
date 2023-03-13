# Multi-scale spatial prediction models

::: {.rmdnote}
You are reading the work-in-progress Spatial and spatiotemporal interpolation using Ensemble Machine Learning. This chapter is currently draft version, a peer-review publication is pending. You can find the polished first edition at <https://opengeohub.github.io/spatial-prediction-eml/>.
:::



## Rationale for multiscale models

In the previous examples we have shown how to fit spatial and spatiotemporal models 
to generate predictions using multiple covariate layers. In practice spatial layers 
used for predictive mapping could come and different spatial scales i.e. they could 
be represent different part of spatial variation. There are at least two scales of 
spatial variation [@hengl2021african]:

- **Coarse scale** e.g. representing effects of planetary climate;
- **Fine scale** e.g. representing meso-relief and local conditions;

In fact, we can imagine that spatial variation can probably be decomposed into different 
scale components, as illustrated in plot below.

<div class="figure" style="text-align: center">
<img src="./img/Fig_signal_decomposition.png" alt="Decomposition of a signal of spatial variation into four components plus noise. Based on McBratney (1998)." width="100%" />
<p class="caption">(\#fig:decomposition-signal)Decomposition of a signal of spatial variation into four components plus noise. Based on McBratney (1998).</p>
</div>

The idea of modeling soil spatial variation at different scales can be traced back to the work of @McBratney1998.
That also suggests that we could produce predictions models of different components 
of variation, then sum the components to produce ensemble prediction. The rationale 
for this, in the case of large datasets, is that we can (a) significantly reduce size 
of the data, (b) separate and better focus modeling based on the component of variation.

## Fitting and predicting with multiscale models

In the next example we use EML to make spatial predictions using data-set with 
two sets of covariates basically at different resolutions 250-m and 100-m. For 
this we use the Edgeroi data-set [@malone2009mapping] used commonly in the soil 
science to demonstrate 3D soil mapping of soil organic carbon (g/kg) based on 
samples taken from diagnostic soil horizons (multiple depth intervals):
  

```r
data(edgeroi)
edgeroi.sp <- edgeroi$sites
coordinates(edgeroi.sp) <- ~ LONGDA94 + LATGDA94
proj4string(edgeroi.sp) <- CRS("+proj=longlat +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +no_defs")
edgeroi.sp <- spTransform(edgeroi.sp, CRS("+init=epsg:28355"))
out.file = paste(getwd(), "output/edgeroi/edgeroi_training_points.gpkg", sep="/")
#if(!file.exists("out.file")){
#  writeOGR(edgeroi.sp, out.file, layer="edgeroi_training_points", driver = "GPKG")
#}
```

We can fit two independent EML's using the two sets of covariates and then 
produce final predictions by combining them. We will refer to the two models as 
coarse and fine-scale models. The fine-scale models will often be much larger 
datasets and require serious computing capacity.

## Coarse-scale model

First we use the 250-m resolution covariates:


```r
load("input/edgeroi.grids.rda")
gridded(edgeroi.grids) <- ~x+y
proj4string(edgeroi.grids) <- CRS("+init=epsg:28355")
ov2 <- over(edgeroi.sp, edgeroi.grids)
ov2$SOURCEID <- edgeroi.sp$SOURCEID
ov2$x = edgeroi.sp@coords[,1]
ov2$y = edgeroi.sp@coords[,2]
```

This is a 3D soil data set, so we also use the horizon `DEPTH` to explain distribution of SOC in soil: 


```r
source("PSM_functions.R")
h2 <- hor2xyd(edgeroi$horizons)
## regression matrix:
rm2 <- plyr::join_all(dfs = list(edgeroi$sites, h2, ov2))
#> Joining by: SOURCEID
#> Joining by: SOURCEID
formulaStringP2 <- ORCDRC ~ DEMSRT5+TWISRT5+EV1MOD5+EV2MOD5+EV3MOD5+DEPTH
rmP2 <- rm2[complete.cases(rm2[,all.vars(formulaStringP2)]),]
str(rmP2[,all.vars(formulaStringP2)])
#> 'data.frame':	4972 obs. of  7 variables:
#>  $ ORCDRC : num  8.5 7.3 5 4.7 4.7 ...
#>  $ DEMSRT5: num  198 198 198 198 198 198 185 185 185 185 ...
#>  $ TWISRT5: num  19.5 19.5 19.5 19.5 19.5 19.5 19.2 19.2 19.2 19.2 ...
#>  $ EV1MOD5: num  1.14 1.14 1.14 1.14 1.14 1.14 -4.7 -4.7 -4.7 -4.7 ...
#>  $ EV2MOD5: num  1.62 1.62 1.62 1.62 1.62 1.62 3.46 3.46 3.46 3.46 ...
#>  $ EV3MOD5: num  -5.74 -5.74 -5.74 -5.74 -5.74 -5.74 0.01 0.01 0.01 0.01 ...
#>  $ DEPTH  : num  11.5 17.5 26 55 80 ...
```

We can now fit an EML directly by using the derived regression matrix:


```r
if(!exists("m.oc")){
  m.oc = train.spLearner.matrix(rmP2, formulaStringP2, edgeroi.grids, 
                        parallel=FALSE, cov.model="nugget", cell.size=1000)
}
#> as.geodata: 4655 replicated data locations found. 
#>  Consider using jitterDupCoords() for jittering replicated locations. 
#> WARNING: there are data at coincident or very closed locations, some of the geoR's functions may not work.
#>  Use function dup.coords() to locate duplicated coordinates.
#>  Consider using jitterDupCoords() for jittering replicated locations 
#> # weights:  25
#> initial  value 253895.808438 
#> iter  10 value 131424.395513
#> iter  20 value 92375.545449
#> iter  30 value 88023.497878
#> iter  40 value 78161.622563
#> iter  50 value 71869.588437
#> iter  60 value 69482.655270
#> iter  70 value 68642.175705
#> iter  80 value 68405.025197
#> iter  90 value 68402.034647
#> final  value 68402.000553 
#> converged
#> # weights:  25
#> initial  value 254790.970425 
#> final  value 136782.219163 
#> converged
#> # weights:  25
#> initial  value 285010.126650 
#> final  value 135478.529982 
#> converged
#> # weights:  25
#> initial  value 253832.562811 
#> final  value 137484.280876 
#> converged
#> # weights:  25
#> initial  value 254385.881547 
#> iter  10 value 98551.221418
#> iter  20 value 94579.214923
#> iter  30 value 93473.664614
#> iter  40 value 93169.177514
#> iter  50 value 93139.660457
#> iter  60 value 93111.617240
#> iter  70 value 93111.256287
#> iter  80 value 93109.156591
#> iter  90 value 93099.925818
#> iter 100 value 93021.880998
#> final  value 93021.880998 
#> stopped after 100 iterations
#> # weights:  25
#> initial  value 233465.782262 
#> final  value 134647.206038 
#> converged
#> # weights:  25
#> initial  value 246624.689888 
#> final  value 138702.343415 
#> converged
#> # weights:  25
#> initial  value 241227.341802 
#> final  value 138599.168021 
#> converged
#> # weights:  25
#> initial  value 245735.599010 
#> final  value 131152.446689 
#> converged
#> # weights:  25
#> initial  value 258267.657849 
#> iter  10 value 97368.003255
#> iter  20 value 91058.331259
#> iter  30 value 88735.472097
#> iter  40 value 78495.790097
#> iter  50 value 72384.348608
#> iter  60 value 69221.579542
#> iter  70 value 68248.107158
#> iter  80 value 68073.306874
#> iter  80 value 68073.306453
#> iter  90 value 68072.812884
#> iter  90 value 68072.812251
#> final  value 68072.784169 
#> converged
#> # weights:  25
#> initial  value 268786.613045 
#> iter  10 value 128937.282143
#> iter  20 value 105536.706972
#> iter  30 value 100970.717402
#> iter  40 value 89676.958142
#> iter  50 value 80499.715450
#> iter  60 value 76269.748584
#> iter  70 value 74645.989814
#> iter  80 value 74429.111595
#> iter  90 value 74041.488969
#> iter 100 value 73877.377891
#> final  value 73877.377891 
#> stopped after 100 iterations
```

The **geoR** package here reports problems as the data set is 3D and hence there are spatial 
duplicates. We can ignore this problem and use the pre-defined cell size of 1-km 
for spatial blocking, although in theory one can also fit 3D variograms and then 
determine blocking parameter using training data. 

The results show that the EML model is significant:


```r
summary(m.oc@spModel$learner.model$super.model$learner.model)
#> 
#> Call:
#> stats::lm(formula = f, data = d)
#> 
#> Residuals:
#>     Min      1Q  Median      3Q     Max 
#> -17.668  -0.984  -0.066   0.711  64.291 
#> 
#> Coefficients:
#>               Estimate Std. Error t value Pr(>|t|)    
#> (Intercept)    0.11730    0.18365   0.639  0.52302    
#> regr.ranger    1.12706    0.02474  45.553  < 2e-16 ***
#> regr.xgboost  -0.37833    0.07448  -5.080 3.92e-07 ***
#> regr.nnet     -0.04227    0.02399  -1.762  0.07808 .  
#> regr.ksvm      0.08299    0.02900   2.861  0.00423 ** 
#> regr.cvglmnet -0.05140    0.03879  -1.325  0.18525    
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 3.06 on 4966 degrees of freedom
#> Multiple R-squared:  0.6938,	Adjusted R-squared:  0.6935 
#> F-statistic:  2250 on 5 and 4966 DF,  p-value: < 2.2e-16
```

We can now predict values at e.g. 5-cm depth by adding a dummy spatial layer with all fixed values:


```r
out.tif = "output/edgeroi/pred_oc_250m.tif"
edgeroi.grids$DEPTH <- 5
if(!exists("edgeroi.oc")){
  edgeroi.oc = predict(m.oc, edgeroi.grids[,m.oc@spModel$features])
}
#> Predicting values using 'getStackedBaseLearnerPredictions'...TRUE
#> Deriving model errors using forestError package...TRUE
if(!file.exists(out.tif)){
  writeGDAL(edgeroi.oc$pred["response"], out.tif, 
            options = c("COMPRESS=DEFLATE"))
  writeGDAL(edgeroi.oc$pred["model.error"], "output/edgeroi/pred_oc_250m_pe.tif", 
            options = c("COMPRESS=DEFLATE"))
}
```

which shows the following:

<div class="figure" style="text-align: center">
<img src="multiscale-models_files/figure-html/map-oc250m-1.png" alt="Predicted SOC content using 250-m covariates." width="100%" />
<p class="caption">(\#fig:map-oc250m)Predicted SOC content using 250-m covariates.</p>
</div>

The average prediction error in the map is somewhat higher than the average error from the model fitting:


```r
summary(edgeroi.oc$pred$model.error)
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#>   2.323   4.710   6.342   6.929   9.257  15.338
```

This is because we are predicting the top-soil SOC, which is exponentially higher at the soil surface and hence average model errors for top soil should be slightly larger than the mean error for the whole soil.

## Fine-scale model

We can now fit the fine-scale model independently from the coarse-scale model 
using the 100-m resolution covariates. In this case the 100-m covariates are 
based on Landsat 8 and gamma radiometrics images (see `?edgeroi` for more details): 


```r
edgeroi.grids100 = readRDS("input/edgeroi.grids.100m.rds")
#gridded(edgeroi.grids100) <- ~x+y
#proj4string(edgeroi.grids100) <- CRS("+init=epsg:28355")
ovF <- over(edgeroi.sp, edgeroi.grids100)
ovF$SOURCEID <- edgeroi.sp$SOURCEID
ovF$x = edgeroi.sp@coords[,1]
ovF$y = edgeroi.sp@coords[,2]
rmF <- plyr::join_all(dfs = list(edgeroi$sites, h2, ovF))
#> Joining by: SOURCEID
#> Joining by: SOURCEID
formulaStringPF <- ORCDRC ~ MVBSRT6+TI1LAN6+TI2LAN6+PCKGAD6+RUTGAD6+PCTGAD6+DEPTH
rmPF <- rmF[complete.cases(rmF[,all.vars(formulaStringPF)]),]
str(rmPF[,all.vars(formulaStringPF)])
#> 'data.frame':	5001 obs. of  8 variables:
#>  $ ORCDRC : num  8.5 7.3 5 4.7 4.7 ...
#>  $ MVBSRT6: num  5.97 5.97 5.97 5.97 5.97 5.97 6.7 6.7 6.7 6.7 ...
#>  $ TI1LAN6: num  31.8 31.8 31.8 31.8 31.8 31.8 14.3 14.3 14.3 14.3 ...
#>  $ TI2LAN6: num  32.9 32.9 32.9 32.9 32.9 32.9 22.1 22.1 22.1 22.1 ...
#>  $ PCKGAD6: num  1.39 1.39 1.39 1.39 1.39 1.39 1.06 1.06 1.06 1.06 ...
#>  $ RUTGAD6: num  0.14 0.14 0.14 0.14 0.14 0.14 0.16 0.16 0.16 0.16 ...
#>  $ PCTGAD6: num  7.82 7.82 7.82 7.82 7.82 7.82 6.48 6.48 6.48 6.48 ...
#>  $ DEPTH  : num  11.5 17.5 26 55 80 ...
```

We fit the 2nd fine-scale model:


```r
if(!exists("m.ocF")){
  m.ocF = train.spLearner.matrix(rmPF, formulaStringPF, edgeroi.grids100, 
                        parallel=FALSE, cov.model="nugget", cell.size=1000)
}
#> as.geodata: 4655 replicated data locations found. 
#>  Consider using jitterDupCoords() for jittering replicated locations. 
#> WARNING: there are data at coincident or very closed locations, some of the geoR's functions may not work.
#>  Use function dup.coords() to locate duplicated coordinates.
#>  Consider using jitterDupCoords() for jittering replicated locations 
#> # weights:  28
#> initial  value 259952.010839 
#> iter  10 value 101252.397395
#> iter  20 value 92849.746212
#> iter  30 value 84138.890538
#> iter  40 value 81171.674547
#> iter  50 value 80244.316069
#> iter  60 value 79942.755705
#> iter  70 value 79714.285817
#> iter  80 value 78299.292931
#> iter  90 value 77660.229627
#> iter 100 value 76439.632084
#> final  value 76439.632084 
#> stopped after 100 iterations
#> # weights:  28
#> initial  value 226902.901029 
#> iter  10 value 134520.116202
#> iter  20 value 106665.533239
#> iter  30 value 100456.523529
#> iter  40 value 94907.032527
#> iter  50 value 94598.860459
#> iter  60 value 94311.934401
#> iter  70 value 93110.357174
#> iter  80 value 92843.643684
#> iter  90 value 92584.240506
#> iter 100 value 92181.626755
#> final  value 92181.626755 
#> stopped after 100 iterations
#> # weights:  28
#> initial  value 234725.323531 
#> iter  10 value 99738.688037
#> iter  20 value 95202.777671
#> iter  30 value 93332.714310
#> iter  40 value 84502.258499
#> iter  50 value 81245.631274
#> iter  60 value 80530.199169
#> iter  70 value 79322.812976
#> iter  80 value 78753.418713
#> iter  90 value 78202.739332
#> iter 100 value 76862.478845
#> final  value 76862.478845 
#> stopped after 100 iterations
#> # weights:  28
#> initial  value 264624.170952 
#> iter  10 value 101566.765280
#> iter  20 value 93105.271953
#> iter  30 value 79221.559605
#> iter  40 value 75397.256524
#> iter  50 value 74837.821575
#> iter  60 value 74267.997307
#> iter  70 value 73584.671645
#> iter  80 value 72824.651446
#> iter  90 value 67855.243367
#> iter 100 value 64355.252219
#> final  value 64355.252219 
#> stopped after 100 iterations
#> # weights:  28
#> initial  value 269334.603789 
#> iter  10 value 114589.803267
#> iter  20 value 98585.812766
#> iter  30 value 95611.345448
#> iter  40 value 94119.323802
#> iter  50 value 90814.981698
#> iter  60 value 87341.568604
#> iter  70 value 81668.185752
#> iter  80 value 78304.161754
#> iter  90 value 77329.599858
#> iter 100 value 76085.626220
#> final  value 76085.626220 
#> stopped after 100 iterations
#> # weights:  28
#> initial  value 259884.878897 
#> iter  10 value 103726.379375
#> iter  20 value 94570.752266
#> iter  30 value 76708.677450
#> iter  40 value 73787.991288
#> iter  50 value 73126.652279
#> iter  60 value 72979.797218
#> iter  70 value 72965.952547
#> iter  80 value 72468.280591
#> iter  90 value 72359.464671
#> iter 100 value 71873.895824
#> final  value 71873.895824 
#> stopped after 100 iterations
#> # weights:  28
#> initial  value 285193.065615 
#> iter  10 value 102340.204856
#> iter  20 value 89532.696472
#> iter  30 value 83391.937814
#> iter  40 value 79216.548340
#> iter  50 value 74814.819516
#> iter  60 value 71281.809785
#> iter  70 value 70042.875465
#> iter  80 value 69036.157581
#> iter  90 value 66429.444952
#> iter 100 value 62457.856190
#> final  value 62457.856190 
#> stopped after 100 iterations
#> # weights:  28
#> initial  value 209074.598360 
#> iter  10 value 92158.672417
#> iter  20 value 88477.518932
#> iter  30 value 83611.323463
#> iter  40 value 80398.938707
#> iter  50 value 75638.212800
#> iter  60 value 72443.729706
#> iter  70 value 68713.860329
#> iter  80 value 64573.033967
#> iter  90 value 62080.383861
#> iter 100 value 61330.996785
#> final  value 61330.996785 
#> stopped after 100 iterations
#> # weights:  28
#> initial  value 264913.914581 
#> iter  10 value 90799.203779
#> iter  20 value 83326.307027
#> iter  30 value 71756.357852
#> iter  40 value 66352.773014
#> iter  50 value 65346.537874
#> iter  60 value 64426.380008
#> iter  70 value 63807.365080
#> iter  80 value 62839.592211
#> iter  90 value 60737.710893
#> iter 100 value 59915.872278
#> final  value 59915.872278 
#> stopped after 100 iterations
#> # weights:  28
#> initial  value 254133.236100 
#> iter  10 value 117010.387846
#> iter  20 value 96496.685965
#> iter  30 value 81754.875719
#> iter  40 value 80702.591180
#> iter  50 value 79637.660106
#> iter  60 value 77195.047203
#> iter  70 value 74660.187056
#> iter  80 value 70931.471910
#> iter  90 value 69014.578843
#> iter 100 value 68409.500899
#> final  value 68409.500899 
#> stopped after 100 iterations
#> # weights:  28
#> initial  value 325116.468080 
#> iter  10 value 120203.210463
#> iter  20 value 115964.153279
#> iter  30 value 90105.298970
#> iter  40 value 79437.548718
#> iter  50 value 70168.831437
#> iter  60 value 69050.767980
#> iter  70 value 68627.297281
#> iter  80 value 68156.983605
#> iter  90 value 67374.570008
#> iter 100 value 67164.637732
#> final  value 67164.637732 
#> stopped after 100 iterations
summary(m.ocF@spModel$learner.model$super.model$learner.model)
#> 
#> Call:
#> stats::lm(formula = f, data = d)
#> 
#> Residuals:
#>     Min      1Q  Median      3Q     Max 
#> -19.274  -0.960  -0.052   0.725  61.100 
#> 
#> Coefficients:
#>                Estimate Std. Error t value Pr(>|t|)    
#> (Intercept)   -0.466388   0.114924  -4.058 5.02e-05 ***
#> regr.ranger    1.098643   0.024682  44.512  < 2e-16 ***
#> regr.xgboost   0.009181   0.075823   0.121   0.9036    
#> regr.nnet      0.048347   0.021791   2.219   0.0266 *  
#> regr.ksvm     -0.029587   0.029579  -1.000   0.3172    
#> regr.cvglmnet -0.038096   0.028892  -1.319   0.1874    
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 2.967 on 4995 degrees of freedom
#> Multiple R-squared:  0.7147,	Adjusted R-squared:  0.7145 
#> F-statistic:  2503 on 5 and 4995 DF,  p-value: < 2.2e-16
```

which shows that the 100-m resolution covariates help make even more accurate 
predictions with R-square about 0.7. We can also make predictions at 5-cm depth 
by using (note: this takes almost 6x more time to compute predictions than for 
250-m resolution data):


```r
edgeroi.grids100$DEPTH <- 5
sel.grid = complete.cases(edgeroi.grids100@data[,m.ocF@spModel$features])
if(!exists("edgeroi.ocF")){
  edgeroi.ocF = predict(m.ocF, edgeroi.grids100[sel.grid, m.ocF@spModel$features])
}
#> Predicting values using 'getStackedBaseLearnerPredictions'...TRUE
#> Deriving model errors using forestError package...TRUE
out.tif = "output/edgeroi/pred_oc_100m.tif"
if(!file.exists(out.tif)){
  writeGDAL(edgeroi.ocF$pred["response"], out.tif, options = c("COMPRESS=DEFLATE"))
  writeGDAL(edgeroi.ocF$pred["model.error"], "output/edgeroi/pred_oc_100m_pe.tif", options = c("COMPRESS=DEFLATE"))
}
```

which shows the following:

<div class="figure" style="text-align: center">
<img src="multiscale-models_files/figure-html/map-oc100m-1.png" alt="Predicted SOC content using 100-m covariates." width="100%" />
<p class="caption">(\#fig:map-oc100m)Predicted SOC content using 100-m covariates.</p>
</div>

## Merging multi-scale predictions

If we compare the coarse scale and fine scale predictions we see:

<div class="figure" style="text-align: center">
<img src="./img/ebergotzen_two_scale_model_ensemble.gif" alt="Coarse-scale and fine-scale predictions of soil organic carbon at 5-cm depth for the Edgeroi study area." width="100%" />
<p class="caption">(\#fig:two-scale)Coarse-scale and fine-scale predictions of soil organic carbon at 5-cm depth for the Edgeroi study area.</p>
</div>

Overall there is a match between general patterns but there are also differences locally. This is to expect as the two models are fitted independently using completely different covariates. We can merge the two predictions and produce the final ensemble prediction by using the following principles:

- User prediction errors per pixel as weights so that more accurate predictions get higher weights,  
- Derive propagated error using the pooled variance based on individual predictions and errors,  

Before we run this operation, we need to downscale the maps to the same grid, best using Cubic-splines in GDAL:


```r
edgeroi.grids100@bbox
#>       min     max
#> x  741400  789000
#> y 6646000 6678100
outD.file = "output/edgeroi/pred_oc_250m_100m.tif"
if(!file.exists(outD.file)){
  system(paste0('gdalwarp output/edgeroi/pred_oc_250m.tif ', outD.file,  
         ' -r \"cubicspline\" -te 741400 6646000 789000 6678100 -tr 100 100 -overwrite'))
  system(paste0('gdalwarp output/edgeroi/pred_oc_250m_pe.tif output/edgeroi/pred_oc_250m_100m_pe.tif',
         ' -r \"cubicspline\" -te 741400 6646000 789000 6678100 -tr 100 100 -overwrite'))
}
```

We can now read the downscaled predictions, and merge them using the prediction errors as weights (weighted average per pixel): 


```r
sel.pix = edgeroi.ocF$pred@grid.index
edgeroi.ocF$pred$responseC = readGDAL("output/edgeroi/pred_oc_250m_100m.tif")$band1[sel.pix]
#> output/edgeroi/pred_oc_250m_100m.tif has GDAL driver GTiff 
#> and has 321 rows and 476 columns
edgeroi.ocF$pred$model.errorC = readGDAL("output/edgeroi/pred_oc_250m_100m_pe.tif")$band1[sel.pix]
#> output/edgeroi/pred_oc_250m_100m_pe.tif has GDAL driver GTiff 
#> and has 321 rows and 476 columns
X = comp.var(edgeroi.ocF$pred@data, r1="response", r2="responseC", v1="model.error", v2="model.errorC")
edgeroi.ocF$pred$responseF = X$response
out.tif = "output/edgeroi/pred_oc_100m_merged.tif"
if(!file.exists(out.tif)){
  writeGDAL(edgeroi.ocF$pred["responseF"], out.tif, options = c("COMPRESS=DEFLATE"))
}
```

The final map of predictions is a combination of the two independently produced predictions [@hengl2021african]:


```r
plot(raster(edgeroi.ocF$pred["responseF"]), col=R_pal[["rainbow_75"]][4:20],
  main="Merged predictions spLearner", axes=FALSE, box=FALSE)
points(edgeroi.sp, pch="+", cex=.8)
```

<div class="figure" style="text-align: center">
<img src="multiscale-models_files/figure-html/map-2scale-1.png" alt="Merged predictions (coarse+fine scale) of SOC content at 100-m." width="100%" />
<p class="caption">(\#fig:map-2scale)Merged predictions (coarse+fine scale) of SOC content at 100-m.</p>
</div>

To merge the prediction errors, we use the pooled variance formula [@rudmin2010calculating]:


```r
comp.var
#> function (x, r1, r2, v1, v2) 
#> {
#>     r = rowSums(x[, c(r1, r2)] * 1/(x[, c(v1, v2)]^2))/rowSums(1/(x[, 
#>         c(v1, v2)]^2))
#>     v = sqrt(rowMeans(x[, c(r1, r2)]^2 + x[, c(v1, v2)]^2) - 
#>         rowMeans(x[, c(r1, r2)])^2)
#>     return(data.frame(response = r, stdev = v))
#> }
#> <bytecode: 0x7019e838>
edgeroi.ocF$pred$model.errorF = X$stdev
out.tif = "output/edgeroi/pred_oc_100m_merged_pe.tif"
if(!file.exists(out.tif)){
  writeGDAL(edgeroi.ocF$pred["model.errorF"], out.tif, options = c("COMPRESS=DEFLATE"))
}
```

So in summary, merging multi-scale predictions is a straight forward process, 
but it assumes that the reliable prediction errors are available for both coarse and fine scale predictions. 
The pooled variance might show higher errors where predictions between independent 
models differ significantly and this is correct. The 2-scale Ensemble Machine 
Learning method of Predictive Soil Mapping was used, for example, to produce 
predictions of [soil properties and nutrients of Africa at 30-m spatial resolution](https://www.isda-africa.com/isdasoil/) [@hengl2021african].
