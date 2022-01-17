
hor2xyd <- function(x, U="UHDICM", L="LHDICM", treshold.T=15){
  x$DEPTH <- x[,U] + (x[,L] - x[,U])/2
  x$THICK <- x[,L] - x[,U]
  sel <- x$THICK < treshold.T
  ## begin and end of the horizon:
  x1 <- x[!sel,]; x1$DEPTH = x1[,L]
  x2 <- x[!sel,]; x2$DEPTH = x1[,U]
  y <- do.call(rbind, list(x, x1, x2))
  return(y)
}

temp.from.geom <- function(fi, day, a=30.419375, 
                           b=-15.539232, elev=0, t.grad=0.6) {
  f = ifelse(fi==0, 1e-10, fi)
  costeta = cos( (day-18 )*pi/182.5 + 2^(1-sign(fi) ) *pi) 
  cosfi = cos(fi*pi/180 )
  A = cosfi
  B = (1-costeta ) * abs(sin(fi*pi/180 ) )
  x = a*A + b*B - t.grad * elev / 100
  return(x)
}


comp.var = function(x, r1, r2, v1, v2){
  r =  rowSums( x[,c(r1, r2)] * 1/(x[,c(v1, v2)]^2) ) / rowSums( 1/(x[,c(v1, v2)]^2) )
  ## https://stackoverflow.com/questions/13593196/standard-deviation-of-combined-data
  v = sqrt( rowMeans(x[,c(r1, r2)]^2 + x[,c(v1, v2)]^2) - rowMeans( x[,c(r1, r2)])^2 )
  return(data.frame(response=r, stdev=v))
}

# grand.sd   <- function(S, M, N) {sqrt(weighted.mean(S^2 + M^2, N) - weighted.mean(M, N)^2)}
#comp.var(x=data.frame(r1=6.2, r2=7, v1=0.4, v2=0.6), "r1", "r2", "v1", "v2")
#comp.var(x=data.frame(r1=6.4, r2=6.6, v1=0.4, v2=0.6), "r1", "r2", "v1", "v2")
