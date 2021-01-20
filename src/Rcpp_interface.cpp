// Rcpp_interface.cpp

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppGSL)]]

#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <limits>
#include <omp.h>
#include <Rcpp.h>
using namespace std;

#include "randomizer.h"
#include "distribution.h"
#include "Rcpp_interface.h"
#include "mcmc.h"




//' Fit a model with MCMC
//'
//' Run MCMC on a model compiled using \code{build_model}.
//'
//' Uses the Differential Evolution MCMC algorithm to sample from the posterior distribution
//' of a model's parameters.
//'
//' @param model the model built using \code{build_model}.
//' @param burn_in number of iterations used for burn-in.
//' @param iterations number of iterations used for sampling.
//' @param chains number of chains used for model fitting. It is recommended to use twice as
//'     many chains as there are parameters. The default value of 0 adopts this behaviour.
//' @param threads number of threads to run fitting over in parallel.
//' @param verbose \code{TRUE} to provide more detailed updates on the posterior during fitting.
//' @param seed seeds the random number generator used by the MCMC algorithm.
//' @return A \code{data.frame} containing posterior samples for the model parameters,
//'     in addition to the log probability of the sample (\code{lp}), log likelihood (\code{ll}),
//'     iteration (\code{trial}) and chain number (\code{chain}).
//'
//' @examples
//' model = make_model("test", c(x = "N 0 1", y = "N 0 1"), "ll += -(x + y) * (x + y);")
//' RCppMCMC(model)
//'
//' @export
// [[Rcpp::export]]
Rcpp::DataFrame RCppMCMC(Rcpp::List model, unsigned int burn_in = 500, unsigned int iterations = 500, unsigned int chains = 0,
    unsigned int threads = 1, bool verbose = false, unsigned long int seed = 0)
{
    Rcpp::List params_priors = Rcpp::as<Rcpp::List>(model["priors"]);
    Rcpp::XPtr<cpp_model_func> func = Rcpp::as<Rcpp::XPtr<cpp_model_func>>(model["cpp_model_func"]);

    // Extract parameter names and priors
    vector<string> param_names = Rcpp::as<vector<string>>(params_priors.names());
    vector<Distribution> priors;

    for (unsigned int i = 0; i < params_priors.size(); ++i)
        priors.push_back(Distribution(Rcpp::as<string>(params_priors[i])));

    // Set parameters
    ///---
    chains = max(4L, (chains == 0 ? 2 * params_priors.size() : chains));
    bool reeval_likelihood = false;
    bool classic_gamma = false;
    bool in_parallel = threads > 1;
    ///---

    // Initialise objects
    Randomizer Rand(seed);
    Likelihood lik(*func);
    MCMCReporter rep(iterations, chains, param_names);

    // Do fitting
    DEMCMC_Priors(Rand, lik, rep, burn_in, iterations, chains, priors, verbose, param_names,
        reeval_likelihood, in_parallel, threads, classic_gamma);

    // Get data.frame as a data.table and return
    Rcpp::DataFrame df = Rcpp::DataFrame::create();
    df.push_back(Rcpp::IntegerVector::import(rep.trial.begin(), rep.trial.end()), "trial");
    df.push_back(Rcpp::NumericVector::import(rep.lp.begin(), rep.lp.end()), "lp");
    df.push_back(Rcpp::IntegerVector::import(rep.chain.begin(), rep.chain.end()), "chain");
    df.push_back(Rcpp::NumericVector::import(rep.ll.begin(), rep.ll.end()), "ll");
    for (unsigned int d = 0; d < rep.theta.size(); ++d)
        df.push_back(Rcpp::NumericVector::import(rep.theta[d].begin(), rep.theta[d].end()), rep.pnames[d]);

    return Rcpp::as<Rcpp::DataFrame>(df);
}
