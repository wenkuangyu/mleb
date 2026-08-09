#' Modified Likelihood Empirical Bayes Estimator of Normal Means
#'
#' Fits the modified likelihood Empirical Bayes (MLEB) estimator using the kernel Tweedie formula.
#' Parameters \code{(b, s)} can be explicitly supplied or automatically determined using 
#' Stein's Unbiased Risk Estimate (\code{"sure"}) or plug-in (\code{"plugin"}) methods.
#'
#' @param X Numeric vector of observations.
#' @param par Parameter vector \code{(b, s)}. If \code{NULL}, parameters are determined via \code{method}.
#' @param method Selection method if \code{par} is \code{NULL}; either \code{"plugin"} (default) or \code{"sure"}.
#' @param binned Logical. If \code{TRUE}, uses grid-based binning for faster evaluation. Default is \code{FALSE}.
#' @param nbins Integer. Number of grid bins to use when \code{binned = TRUE}. Default is \code{2048}.
#'
#' @return A list containing estimated values and chosen parameters:
#' \item{tweedie}{Raw Tweedie estimates mapped back to original data order.}
#' \item{mono_tweedie}{Monotonized Tweedie estimates via PAVA mapped back to original data order.}
#' \item{b}{Bandwidth parameter used.}
#' \item{s}{Scale parameter used.}
#' \item{sure_val}{SURE risk value at the optimum (NA if method is "plugin" or parameters are user-supplied).}
#' \item{par_selection}{Method or indication of how parameters were determined.}
#' \item{binned}{Logical flag indicating whether binning was used.}
#'
#' @useDynLib mleb, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @export
mleb <- function(X, par = NULL, method = c("plugin", "sure"), binned = FALSE, nbins = 2048) {
  if (!is.numeric(X) || length(X) == 0 || anyNA(X)) {
    stop("'X' must be a non-empty numeric vector without missing values.")
  }
  
  method <- match.arg(method)
  
  # Sort X and keep original indices
  s_X <- sort(X, index.return = TRUE)
  X_sorted <- s_X$x
  ord <- s_X$ix
  
  sure_val <- NA_real_
  
  # 1. Parameter Selection
  if (!is.null(par)) {
    if (!is.numeric(par) || length(par) != 2) {
      stop("'par' must be a numeric vector of length 2: c(b, s).")
    }
    b <- par[1]
    s <- par[2]
    par_selection <- "user-supplied"
  } else if (method == "sure") {
    sure_res <- if (binned) optimize_SURE_binned_lbfgsb(X_sorted, M = nbins) else optimize_SURE_lbfgsb(X_sorted)
    b <- sure_res$b_opt
    s <- sure_res$s_opt
    sure_val <- sure_res$min_risk
    par_selection <- "sure"
  } else { # method == "plugin"
    plugin_res <- plugin_mleb(X_sorted)
    b <- plugin_res$b
    s <- plugin_res$s
    par_selection <- "plugin"
  }
  
  # 2. Kernel Tweedie Evaluation
  res <- if (binned) kde_tweedie_binned(X_sorted, b, s, M = nbins) else kde_tweedie(X_sorted, b, s)
  
  # 3. Map back to original vector order
  tweedie <- numeric(length(X))
  mono_tweedie <- numeric(length(X))
  
  tweedie[ord] <- res$tweedie
  mono_tweedie[ord] <- res$mono_tweedie
  
  list(
    tweedie       = tweedie,
    mono_tweedie  = mono_tweedie,
    b             = b,
    s             = s,
    sure_val      = sure_val,
    par_selection = par_selection,
    binned        = binned
  )
}


#' Modified Likelihood Empirical Bayes Regression Estimator
#'
#' Fits the modified likelihood Empirical Bayes (MLEB) regression model using the kernel Tweedie formula.
#' Semiparametric regression parameters \eqn{\beta} are estimated via the Bickel-Collins risk criterion directly in C++,
#' and Tweedie estimation is applied to the standardized residuals \eqn{Z = (X - W\hat{\beta}) / \sigma}.
#' Parameters \code{(b, s)} can be explicitly supplied or automatically determined using 
#' Stein's Unbiased Risk Estimate (\code{"sure"}) or plug-in (\code{"plugin"}) methods.
#'
#' @param X Numeric vector of observations (length N).
#' @param W Numeric covariate matrix or vector (N x p). Must not be \code{NULL}.
#' @param sigma Standard deviations for heteroscedastic noise. Can be a scalar or a numeric vector of length N. Default is \code{1}.
#' @param par Parameter vector \code{c(b, s)}. If \code{NULL}, parameters are determined via \code{method}.
#' @param method Selection method if \code{par} is \code{NULL}; either \code{"plugin"} (default) or \code{"sure"}.
#' @param binned Logical. If \code{TRUE}, uses grid-based binning for faster evaluation. Default is \code{FALSE}.
#' @param nbins Integer. Number of grid bins to use when \code{binned = TRUE}. Default is \code{2048}.
#' @param beta_init Optional initial parameter vector for regression. If \code{NULL}, initialized via Weighted Least Squares.
#' @param maxit Maximum iterations for BFGS optimizer in Bickel-Collins step. Default is \code{500}.
#' @param abstol Absolute convergence tolerance for optimizer. Default is \code{1e-16}.
#' @param reltol Relative convergence tolerance for optimizer. Default is \code{1e-8}.
#' @param trace Integer. Debugging trace flag for optimizer (0 = silent). Default is \code{0}.
#'
#' @return A list containing estimated values and chosen parameters:
#' \item{tweedie}{Raw Tweedie estimates mapped back to original data scale (\eqn{W\hat{\beta} + \sigma \cdot \hat{\delta}(Z)}).}
#' \item{mono_tweedie}{Monotonized Tweedie estimates via PAVA mapped back to original data scale.}
#' \item{beta_hat}{Estimated regression parameter vector \eqn{\hat{\beta}}.}
#' \item{b}{Bandwidth parameter used for Tweedie estimation.}
#' \item{s}{Scale parameter used for Tweedie estimation.}
#' \item{sure_val}{SURE risk value at the optimum (\code{NA} if method is "plugin" or parameters are user-supplied).}
#' \item{par_selection}{Method or indication of how parameters \code{(b, s)} were determined.}
#' \item{binned}{Logical flag indicating whether binning was used.}
#' \item{bc_fit}{Full return list from C++ Bickel-Collins optimization.}
#'
#' @useDynLib mleb, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @export
mleb_reg <- function(X, 
                     W, 
                     sigma = 1, 
                     par = NULL, 
                     method = c("plugin", "sure"), 
                     binned = FALSE, 
                     nbins = 2048,
                     beta_init = NULL,
                     maxit = 500,
                     abstol = 1e-16,
                     reltol = 1e-8,
                     trace = 0) {
  
  # 1. Input Validation
  if (!is.numeric(X) || length(X) == 0 || anyNA(X)) {
    stop("'X' must be a non-empty numeric vector without missing values.")
  }
  
  N <- length(X)
  
  if (missing(W) || is.null(W)) {
    stop("'W' cannot be NULL for 'mleb_reg()'. For location-only models, use 'mleb()'.")
  }
  
  W <- as.matrix(W)
  if (!is.numeric(W) || nrow(W) != N || anyNA(W)) {
    stop("'W' must be a numeric matrix with row count matching length(X) and no missing values.")
  }
  
  method <- match.arg(method)
  
  # Validate sigma
  if (!is.numeric(sigma) || anyNA(sigma) || any(sigma <= 0)) {
    stop("'sigma' must be a positive numeric vector or scalar without missing values.")
  }
  if (length(sigma) == 1) {
    sigma <- rep(sigma, N)
  } else if (length(sigma) != N) {
    stop("Length of 'sigma' must equal 1 or length(X).")
  }
  
  # Default bandwidth for Bickel-Collins criterion: h = (log N)^(-1/2)
  h <- (log(N))^(-0.5)
  
  # 2. Covariate Regression via Bickel-Collins C++ Exported Functions
  bc_fit <- if (binned) {
    bickel_collins_binned_cpp(
      X         = X,
      W         = W,
      sigma     = sigma,
      h         = h,
      n_grid    = as.integer(nbins),
      beta_init = beta_init,
      maxit     = as.integer(maxit),
      abstol    = abstol,
      reltol    = reltol,
      trace     = as.integer(trace)
    )
  } else {
    bickel_collins_cpp(
      X         = X,
      W         = W,
      sigma     = sigma,
      h         = h,
      beta_init = beta_init,
      maxit     = as.integer(maxit),
      abstol    = abstol,
      reltol    = reltol,
      trace     = as.integer(trace)
    )
  }
  
  beta_hat <- bc_fit$beta_hat
  mu_hat   <- as.vector(W %*% beta_hat)
  
  # Standardized Residuals Z = (X - mu_hat) / sigma
  Z <- (X - mu_hat) / sigma
  
  # Sort Z and keep original indices
  s_Z <- sort(Z, index.return = TRUE)
  Z_sorted <- s_Z$x
  ord <- s_Z$ix
  
  sure_val <- NA_real_
  
  # 3. Parameter Selection for Tweedie Formula
  if (!is.null(par)) {
    if (!is.numeric(par) || length(par) != 2) {
      stop("'par' must be a numeric vector of length 2: c(b, s).")
    }
    b <- par[1]
    s <- par[2]
    par_selection <- "user-supplied"
  } else if (method == "sure") {
    sure_res <- if (binned) {
      optimize_SURE_binned_lbfgsb(Z_sorted, M = nbins)
    } else {
      optimize_SURE_lbfgsb(Z_sorted)
    }
    b <- sure_res$b_opt
    s <- sure_res$s_opt
    sure_val <- sure_res$min_risk
    par_selection <- "sure"
  } else { # method == "plugin"
    plugin_res <- plugin_mleb(Z_sorted)
    b <- plugin_res$b
    s <- plugin_res$s
    par_selection <- "plugin"
  }
  
  # 4. Kernel Tweedie Evaluation on Z
  res <- if (binned) {
    kde_tweedie_binned(Z_sorted, b, s, M = nbins)
  } else {
    kde_tweedie(Z_sorted, b, s)
  }
  
  # 5. Map back to original order & reconstruct full signal: mu_hat + sigma * Tweedie(Z)
  tweedie_res <- numeric(N)
  mono_tweedie_res <- numeric(N)
  
  tweedie_res[ord] <- res$tweedie
  mono_tweedie_res[ord] <- res$mono_tweedie
  
  tweedie      <- mu_hat + sigma * tweedie_res
  mono_tweedie <- mu_hat + sigma * mono_tweedie_res
  
  list(
    tweedie       = tweedie,
    mono_tweedie  = mono_tweedie,
    beta_hat      = beta_hat,
    b             = b,
    s             = s,
    sure_val      = sure_val,
    par_selection = par_selection,
    binned        = binned,
    bc_fit        = bc_fit
  )
}