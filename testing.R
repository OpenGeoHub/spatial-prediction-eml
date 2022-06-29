## tests
x = list.files("/mnt/landmark/aedesa/mood1km", glob2rx("*.tif$"), full.names = TRUE, recursive = TRUE)
xs = file.size(x)/1e6
write.csv(data.frame(filename=x, size=paste(round(xs, 1), "Mb")), "./input/mood1km_layers.csv")
