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