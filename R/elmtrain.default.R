elmtrain.default <-
function(x,y,nhid,actfun,C=NULL,reduc=1.0,gamma=1.0,largeDB=TRUE,GPU=FALSE,...) {
  require(MASS)
  
  if(nhid < 1) stop("ERROR: number of hidden neurons must be >= 1")
  if(reduc <= 0) stop("ERROR: percentage reduction of weights must be > 1")
  
  T <- t(y)
  P <- t(x)
  inpweight <- randomMatrix(nrow(P),nhid,-1,1) * reduc
  
  if (!GPU){
  
  tempH <- inpweight %*% P
  biashid <- runif(nhid,min=-1,max=1) * reduc
  biasMatrix <- matrix(rep(biashid, ncol(P)), nrow=nhid, ncol=ncol(P), byrow = F) 
  tempH = tempH + biasMatrix
  
  if(actfun == "sig") H = 1 / (1 + exp(-1*tempH))
  else {
    if(actfun == "sin") H = sin(tempH)
    else {
      if(actfun == "radbas") H = exp(-gamma*(tempH^2))
      else {
        if(actfun == "hardlim") H = hardlim(tempH)
        else {
          if(actfun == "hardlims") H = hardlims(tempH)
          else {
            if(actfun == "satlins") H = satlins(tempH)
            else {
              if(actfun == "tansig") H = 2/(1+exp(-2*tempH))-1
              else {
                if(actfun == "tribas") H = tribas(tempH)
                else {
                  if(actfun == "poslin") H = poslin(tempH)
                  else {
                    if(actfun == "purelin") H = tempH
                    else stop(paste("ERROR: ",actfun," is not a valid activation function.",sep=""))
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if (is.null(C))
    {
    # Without Complexity Parameter  
    outweight <- ginv(t(H), tol = sqrt(.Machine$double.eps)) %*% t(T)    
    } else {
    
    # With Complexity Parameter
    if(C==0) stop("ERROR: C must be > 0")
    if (largeDB)
      {
      H.Prod <- H %*% t(H)
      C.Mat <- diag(ncol(H.Prod))/C + H.Prod
      outweight <- solve(C.Mat) %*% H %*% t(T)
      } else
      {
      H.Prod <- t(H) %*% H
      C.Mat <- diag(ncol(H.Prod))/C + H.Prod
      outweight <- H %*% solve(C.Mat)  %*% t(T)  
      }
      

    }

  Y <- t(t(H) %*% outweight)
  
  } else{
    
    require(gputools)
    
    tempH <- gpuMatMult(inpweight, P)
    biashid <- runif(nhid,min=-1,max=1) * reduc
    biasMatrix <- matrix(rep(biashid, ncol(P)), nrow=nhid, ncol=ncol(P), byrow = F)
    tempH <- tempH + biasMatrix
    
    if(actfun == "sig") H = 1 / (1 + exp(-1*tempH))
    else {
      if(actfun == "sin") H = sin(tempH)
      else {
        if(actfun == "radbas") H = exp(-gamma*(tempH^2))
        else {
          if(actfun == "hardlim") H = hardlim(tempH)
          else {
            if(actfun == "hardlims") H = hardlims(tempH)
            else {
              if(actfun == "satlins") H = satlins(tempH)
              else {
                if(actfun == "tansig") H = 2/(1+exp(-2*tempH))-1
                else {
                  if(actfun == "tribas") H = tribas(tempH)
                  else {
                    if(actfun == "poslin") H = poslin(tempH)
                    else {
                      if(actfun == "purelin") H = tempH
                      else stop(paste("ERROR: ",actfun," is not a valid activation function.",sep=""))
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    if (is.null(C))
    {
      # Without Complexity Parameter   
      aux <- ginv(t(H), tol = sqrt(.Machine$double.eps))
      outweight <- gpuMatMult(aux,t(T))    
    } else {
      
      # With Complexity Parameter
      
      if(C==0) stop("ERROR: C must be > 0")
      if (largeDB)
      {
        H.Prod <- gpuMatMult(H,t(H))
        C.Mat <- diag(ncol(H.Prod))/C + H.Prod
        inverse<-solve(C.Mat)
        mult <- gpuMatMult(inverse,H)
        outweight <- gpuMatMult(mult, t(T))
        
      
      } else
      {
        H.Prod <- gpuMatMult(t(H),H)
        C.Mat <- diag(ncol(H.Prod))/C + H.Prod
        inverse <- solve(C.Mat)
        mult <- gpuMatMult(H,inverse)
        outweight <- gpuMatMult(mult,t(T))
      }
      
      
    }
    
    Y <- t(gpuMatMult(t(H), outweight))
    
  }
  
  model = list(inpweight=inpweight,biashid=biashid,outweight=outweight,actfun=actfun,nhid=nhid,predictions=t(Y))
  model$fitted.values <- t(Y)
  model$residuals <- y - model$fitted.values
  model$call <- match.call()
  class(model) <- "elmNN"
  model

}
