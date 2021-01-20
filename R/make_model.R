#' @useDynLib RCppMCMC
#' @importFrom Rcpp sourceCpp
NULL

#' Define a model for fitting
#'
#' Compile a model as C++ code for later use with RCppMCMC.
#'
#' Prior distributions are specified via the \code{priors} parameter, which
#' should be a named vector of \strong{prior definitions}. The names of the
#' entries give the names of the parameters to fit, while the \strong{prior
#' definitions} are character strings that follow a special format:
#'
#' \code{"X 1 2 S 1 2 T 1 2 I 1 2"}
#'
#' Above, \code{"X 1 2"} defines the prior distribution for the parameter.
#' \emph{X} is a one-letter code for the kind of prior distribution (e.g.
#' uniform, normal, etc) and \emph{1} and \emph{2} are numerical parameters
#' of the prior distribution. For example, \code{"U 0 1"} defines a uniform
#' distribution from 0 to 1.
#'
#'   \tabular{llll}{
#'    \code{"U a b"} \tab uniform      \tab \code{a} = minimum  \tab \code{b} = maximum  \cr
#'    \code{"B a b"} \tab beta         \tab \code{a} = alpha    \tab \code{b} = beta     \cr
#'    \code{"N a b"} \tab normal       \tab \code{a} = mean     \tab \code{b} = std. dev \cr
#'    \code{"C a b"} \tab Cauchy       \tab \code{a} = location \tab \code{b} = scale    \cr
#'    \code{"L a b"} \tab lognormal    \tab \code{a} = mean     \tab \code{b} = std. dev (before exponentiation) \cr
#'    \code{"E a b"} \tab exponential  \tab \code{a} = rate     \tab \code{b} = (ignored, but must be specified) \cr
#'    \code{"G a b"} \tab gamma        \tab \code{a} = shape    \tab \code{b} = rate
#'  }
#'
#' The next three parts, \code{"S 1 2"}, \code{"T 1 2"}, and \code{"I 1 2"}
#' are optional and can appear in any order; they define shifting/scaling,
#' truncation, and ranges for initial values for the parameter, respectively.
#'
#' \code{"S a b"} scales (stretches) the distribution by a factor \code{a} and
#' shifts (translates) the distribution by a factor \code{b}. Shifting is applied
#' after scaling. So, for example, another way of writing "N 100 15" is
#' "N 0 1 S 15 100"; this takes a standard normal variate, multiplies it by 15,
#' and then adds 100. This is useful for distributions without a location parameter,
#' like the beta distribution: "B a b S 2 10" makes the distribution range from
#' 10 to 12 instead of from 0 to 1.
#'
#' \code{"T a b"} truncates the distribution between \code{a} and \code{b}, after
#' scaling and shifting; \code{b} must be greater than \code{a}. So for a half-normal,
#' you can do "N 0 10 T 0 1000". Unfortunately there is no way of specifying that one
#' of the bounds is at +/- infinity, so you must use a large-ish number instead.
#'
#' \code{"I a b"} specifies that, when the parameter with this prior is given
#' a random initial value, that value should be drawn from a uniform distribution
#' between \code{a} and \code{b} instead of the initial value being sampled from
#' the prior distribution. This can be useful if you are using a distribution that
#' sometimes produces extreme values, like the Cauchy distribution, and you want to
#' avoid the parameter being initialised with an extreme value; or if the sampler is
#' getting stuck at a weird local maximum, or not mixing well, and you want to
#' give it some guidance.
#'
#' @param name the name of the model; a character string
#' @param priors named vector of parameters and their \strong{prior definitions} (see below).
#' @param likelihood_code vector of lines of C++, which should increment the
#'    variable \code{ll} with log-likelihood components.
#' @return A \code{list} containing components of the compiled model, to be used with the
#' function \code{RCppMCMC}.
#'
#' @examples
#' model = make_model("test", c(x = "N 0 1", y = "N 0 1"), "ll += -(x + y) * (x + y);")
#' RCppMCMC(model)
#'
#' @export
make_model = function(name, priors, likelihood_code)
{
    # Variable definitions
    var_defs = paste(
        paste0("    double ", names(priors), " = _theta_[", seq_along(priors) - 1, "];"),
        collapse = "\n");
    cpp_code = paste0("    ", likelihood_code, collapse = '\n');

    # Function names
    model_func   = glue::glue("RCppMCMC_model_{name}");
    model_getter = glue::glue("RCppMCMC_model_{name}_get");

    # Paste together source code
    src = glue::glue(
        '#include <Rcpp.h>',
        '#include <vector>',
        '#include <cmath>',
        '#include <gsl/gsl_rng.h>',
        '#include <gsl/gsl_randist.h>',
        '#include <gsl/gsl_sf.h>',
        '#include <gsl/gsl_cdf.h>',

        'double ${model_func}(const std::vector<double>& _theta_) {',
        '    double ll = 0;',
             var_defs,
             cpp_code,
        '    return ll;',
        '}',

        'typedef double (*cpp_model_func)(const std::vector<double>&);',

        '// [[Rcpp::export]]',
        'Rcpp::XPtr<cpp_model_func> ${model_getter}() {',
        '    return Rcpp::XPtr<cpp_model_func>(new cpp_model_func(&${model_func}));',
        '}',
        .sep = "\n", .open = "${", .close = "}"
    );

    # Compile source code
    Rcpp::sourceCpp(code = src);
    cpp_model_getter = get(model_getter);

    # Return model
    list(
        name = name,
        priors = priors,
        cpp_model_func = cpp_model_getter()
    )
}
