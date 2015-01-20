elmtrain.formula <-
function(formula,data,nhid,actfun,C=NULL,reduc=1.0,gamma=1.0,largeDB=TRUE,...) {
  mf <- model.frame(formula=formula, data=data)
  x <- model.matrix(attr(mf, "terms"), data=mf)
  y <- model.response(mf)
  model <- elmtrain.default(x=x,y=y,nhid=nhid,actfun=actfun,C=C,reduc=reduc,gamma=gamma,largeDB=largeDB,...)
  model$call <- match.call()
  model$formula <- formula
  model
}
