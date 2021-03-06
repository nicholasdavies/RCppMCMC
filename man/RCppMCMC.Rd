% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{RCppMCMC}
\alias{RCppMCMC}
\title{Fit a model with MCMC}
\usage{
RCppMCMC(
  model,
  burn_in = 500L,
  iterations = 500L,
  chains = 0L,
  threads = 1L,
  verbose = FALSE,
  seed = 0L
)
}
\arguments{
\item{model}{the model built using \code{build_model}.}

\item{burn_in}{number of iterations used for burn-in.}

\item{iterations}{number of iterations used for sampling.}

\item{chains}{number of chains used for model fitting. It is recommended to use twice as
many chains as there are parameters. The default value of 0 adopts this behaviour.}

\item{threads}{number of threads to run fitting over in parallel.}

\item{verbose}{\code{TRUE} to provide more detailed updates on the posterior during fitting.}

\item{seed}{seeds the random number generator used by the MCMC algorithm.}
}
\value{
A \code{data.frame} containing posterior samples for the model parameters,
    in addition to the log probability of the sample (\code{lp}), log likelihood (\code{ll}),
    iteration (\code{trial}) and chain number (\code{chain}).
}
\description{
Run MCMC on a model compiled using \code{build_model}.
}
\details{
Uses the Differential Evolution MCMC algorithm to sample from the posterior distribution
of a model's parameters.
}
\examples{
model = make_model("test", c(x = "N 0 1", y = "N 0 1"), "ll += -(x + y) * (x + y);")
RCppMCMC(model)

}
