Introduction to Gaussian processes
==================================

Consider a regression problem with a set of data

.. math::  \set{D} = \setbuilder{(\vec{x}_i, y_i), i \in 1, \dots, n} 

 which is composed of :math:`n` pairs of inputs, :math:`\vec{x}_i`,
which are vectors which describe the location of the datum in parameter
space, which are the inputs for the problem, and :math:`y_i`, the
outputs. The outputs may be noisy; in this work I will only consider
situations where the noise is additive and Gaussian, so

.. raw:: latex

   \begin{equation}
   \label{eq:gp:additive-noise}
    y_i(\vec{x}_i) = f(\vec{x}_i) + \epsilon_i, \quad \text{for} \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)
   \end{equation}

where :math:`\sigma` is the standard deviation of the noise, and
:math:`f` is the (latent) generating function of the data.

This regression problem can be addressed using *Gaussian processes*:

.. raw:: html

   <div class="definition">

A gls:gaussian-process is a collection of random variables, any finite
number of which have a joint Gaussian distribution cite:gpr.book.rw.

.. raw:: html

   </div>

Where it is more conventional to consider a prior over a set of, for
example, real values, such as a normal distribution, the Gaussian
process forms a prior over the functions, :math:`f` from equation
ref:eq:gp:additive-noise, which might form the regression fit to any
observed data. This assumes that the values of the function :math:`f`
behave as

.. raw:: latex

   \begin{equation}
   \label{eq:gp:function-values}
   p(\vec{f} | \vec{x}_1, \vec{x}_2, \dots, \vec{x}_n) = \mathcal{N}(0, \mat{K})
   \end{equation}

where :math:`\mat{K}` is the covariance matrix of :math:`\vec{x_1}` and
:math:`\vec{x_2}`, which can be calculated with reference to some
*covariance function*, :math:`k`, such that
:math:`K_{ij} = k(\vec{x}_i, \vec{x}_j)`. Note that I have assumed that
the abbr:gp is a *zero-mean* process; this assumption is frequent within
the literature. While this prior is initially untrained it still
contains information about our preconceptions of the data through the
form of the covariance function. For example, whether or not we expect
the fit to be smooth, or periodic. Covariance functions will be
discussed in greater detail in section ref:sec:gp:covariance.

By providing training data we can use Bayes theorem to update the
Gaussian process, in the same way that the posterior distribution is
updated by the addition of new data in a standard Bayesian context, and
a posterior on the set of all possible functions to fit the data is
produced. Thus, for a vector of test values of the generating function
:math:`\vec{f}_\star`, the joint posterior
:math:`p(\vec{f}, \vec{f}_* | \vec{y})`, given the observed outputs
:math:`\vec{y}` can be found by updating the abbr:gp prior on the
training and test function values :math:`p(\vec{f}, \vec{f}_*)` with the
likelihood :math:`p(\vec{y}|\vec{f})`:

.. raw:: latex

   \begin{equation}
   \label{eq:gp:bayes}
   p(\vec{f}, \vec{f}_* | \vec{y}) = \frac{p(\vec{f}, \vec{f}_*) p(\vec{y}|\vec{f})}{p(\vec{y})}.
   \end{equation}

Finally the (latent) training-set function values, :math:`\vec{f}` can
be marginalised out:

.. raw:: latex

   \begin{equation}
   p(\vec{f}_* | \vec{y}) = \int p(\vec{f}, \vec{f}_* | \vec{y}) \dd{\vec{f}} = \frac{1}{p(\vec{y})} \int p(\vec{y} | \vec{f}) p(\vec{f}, \vec{f}_*) \dd{\vec{f}}
   \end{equation}

We can take the mean of this posterior in the place of the \`\`best fit
line'' which other techniques produce, and then use the variance to
produce an estimate of the uncertainty of the prediction.

Both the prior :math:`p(\vec{f}, \vec{f}_*)` and the likelihood
:math:`p(\vec{y}|\vec{f})` are Gaussian:

.. raw:: latex

   \begin{equation}
   \label{eq:gp:prior-and-likelihood}
   p(\vec{f}, \vec{f}_*) = \mathcal{N}(\vec{0}, \mat{K}^+), \quad \text{and} \quad 
   p(\vec{y}|\vec{f}) = \mathcal{N}(\vec{f}, \sigma^2 \mat{I})
   \end{equation}

with

.. raw:: latex

   \begin{equation}
     \label{eq:blockK-plus-mat}
     \mat{K}^+ =
     \begin{bmatrix}
       \mat{K}_{\vec{f},\vec{f}} & \mat{K}_{\vec{f},\vec{f}_*} \\ \mat{K}_{\vec{f}_*,\vec{f}} & \mat{K}_{\vec{f}_*, \vec{f}_*}
     \end{bmatrix},
   \end{equation}

and :math:`\mat{I}` the identity matrix.

This leaves the form of the marginalised posterior being analytical:

.. raw:: latex

   \begin{equation}
   \label{eq:gp:posterior}
   p(\vec{f}_* | \vec{y}) = \mathcal{N} \left( 
   \mat{K}_{\vec{f}_*,\vec{f}} (\mat{K}_{\vec{f},\vec{f}} + \sigma^2 \mat{I})^{-1} \vec{y},
   \mat{K}_{\vec{f}_*, \vec{f}_*} - \mat{K}_{\vec{f},\vec{f}_*}( \mat{K}_{\vec{f},\vec{f}}+\sigma^2 \mat{I})^{-1} \mat{K}_{\vec{f},\vec{f}_*}).
   \end{equation}

Figures ref:fig:gp:training-data to ref:fig:gp:posterior-best show
visually how a one-dimensional regressor can be created using an abbr:gp
method, starting from a abbr:gp prior and (noisy) data.

The mean and variance of this posterior distribution can be used to form
a regressor for the data, :math:`\set{D}`, with the mean taking the role
of a \`\`line-of-best-fit'' in conventional regression techniques, while
the variance describes the goodness of that fit.

A graphical model of a abbr:gp is shown in figure
ref:fig:gp:chain-diagram which illustrates an important property of the
abpl:gp model: the addition (or removal) of any input point to the
abbr:gp does not change the distribution of the other variables. This
property allows outputs to be generated at arbitrary locations
throughout the parameter space.

Gaussian processes trained with :math:`N` training data require the
ability to both store and invert an :math:`N\times N` matrix of
covariances between observations; this can be a considerable
computational challenge.

Gaussian processes can be extended from the case of a single-dimensional
input predicting a single-dimensional output to the ability to predict a
multi-dimensional output from a multi-dimensional input
cite:2011arXiv1106.6251A,Alvarez2011a,Bonilla2007.

.. raw:: latex

   \begin{figure}
   \includegraphics{figures/gp/gp-training-data.pdf}
   \caption[Training data for a Gaussian process]{[Step 1] An example of raw training data (containing additive Gaussian noise) which is suitable for training a Gaussian process. In this example the input data ($x$-axis) are 1-dimensional, although GPs are also capable of handling multi-dimensional data.
   Here the generating function is plotted as a grey line.
   \label{fig:gp:training-data}}
   \end{figure}

.. raw:: latex

   \begin{figure}
   \includegraphics{figures/gp/gp-example-prior-draws.pdf}
   \caption[Draws from a Gaussian process prior]{[Step 2] We choose a covariance function for the  Gaussian process, in this case an exponential-quadratic covariance    function. The Gaussian process containing no data and this    covariance matrix forms our prior probability distribution. Here    50 draws from the prior distribution are plotted. \label{fig:gp:prior}}
   \end{figure}

.. raw:: latex

   \begin{figure}
   \includegraphics{figures/gp/gp-example-posterior-draws.pdf}
   \caption[Draws from a Gaussian process posterior]{[Step 3] The trained Gaussian process can be     sampled multiple times to produce multiple different potential     fitting functions. Here 50 draws from the Gaussian process posterior are    displayed. \label{fig:gp:covariance-matrix}}
   \end{figure}

.. raw:: latex

   \begin{figure}
   \includegraphics{figures/gp/gp-posterior-meancovar.pdf}
   \caption[The mean and variance of a Gaussian process regression prediction]{[Step 4] We can then take the mean and the covariance of the Gaussian process, and produce a single ``best-fit'' with confidence intervals.
   Again, the original generating function for the data is shown as a grey line. \label{fig:gp:posterior-best}}
   \end{figure}

Covariance Functions
====================

The covariance function defines the similarity of a pair of data points,
according to some relationship with suitable properties. The similarity
of input data is assumed to be related to the similarity of the output,
and therefore the more similar two inputs are the more likely their
outputs are to be similar.

As such, the form of the covariance function represents prior knowledge
about the data, and can encode understanding of effects such as
periodicity within the data.

.. raw:: html

   <div class="definition">

A stationary covariance function is a function
:math:`f(\vec{x} - \vec{x}')`, and which is thus invariant to
translations in the input space.

.. raw:: html

   </div>

.. raw:: html

   <div class="definition">

If a covariance function is a function of the form
:math:`f(|\vec{x} - \vec{x}'|)` then it is isotropic, and invariant
under all rigid motions.

.. raw:: html

   </div>

A covariance function which is both stationary and isotropic has the
property that it can be expressed as a function of a single variable,
:math:`r = | \vec{x} - \vec{x}' |` is known as a abbr:rbf. Functions of
the form :math:`k : (\vec{x}, \vec{x}') \to \mathbb{C}`, for two vectors
:math:`\vec{x}, \vec{x}' \in \mathcal{X}` are often known as *kernels*,
and I will frequently refer interchangably to covariance functions and
kernels where the covariance function has this form.

For a set of points :math:`\setbuilder{ \vec{x}_{i} | i = 1, \dots, n }`
a kernel, :math:`k` can be used to construct the gram matrix,
:math:`K_{i,j} = k(x_{i}, x_{j})`. If the kernel is also a covariance
function then :math:`K` is known as a *covariance matrix*.

For a kernel to be a valid covariance function for a abbr:gp it must
produce a positive semidefinite covariance matrix :math:`\mat{K}`. Such
a matrix, :math:`\mat{K} \in \mathbb{R}^{n \times n}` must satisfy
:math:`\vec{x}^{\transpose} \mat{K} \vec{x} \geq 0` for all
:math:`\vec{x} \in \mathbb{R}^{n}`.

Example covariance functions
----------------------------

One of the most frequently encountered covariance functions in the
literature is the abbr:se covariance functions cite:gpr.book.rw. Perhaps
as a result of its near-ubiquity this kernel is known under a number of
similar, but confusing names (which are often inaccurate). These include
the *exponential quadratic*, *quadratic exponential*, *squared
exponential*, and even *Gaussian* covariance function.

The reason for this is its form, which closely resembles that of the
Gaussian function:

.. raw:: latex

   \begin{equation}
      \label{eq:gp:kernels:se}
     k_{\mathrm{SE}}(r) = \exp \left( - \frac{r^2}{2 l^2} \right)
   \end{equation}

for :math:`r` the Euclidean distance of a datum from the centre of the
parameter space, and :math:`l` is a scale factor associated with the
axis along which the data are defined.

.. raw:: latex

   \begin{figure}
   \includegraphics{figures/gp/covariance-se-overview.pdf}
   \caption[The squared exponential covariance function]{The \textbf{squared exponential} covariance function (defined in equation ref:eq:gp:kernels:se). The panel on the left depicts the value of the kernel as a function of $r = (|\vec{x} - \vec{x}'|)$, at a number of different length scales ($l = 0.25, 0.5, 1.0$) while the panel on the right contains draws from Gaussian processes with \gls{se} covariance with the same length scales as the left panel.
   \label{fig:gp:covariance:overviews:se}}
   \end{figure}

The squared exponential function imposes strong smoothness constraints
on the model, as it is infinitely differentiable.

The scale factor, :math:`l` in ref:eq:gp:kernels:se, also known as its
*scale-length* defines the size of the effect within the process. This
characteristic length-scale can be understood cite:adler1976,gpr.book.rw
in terms of the number of times the abbr:gp should cross some given
level (for example, zero). Indeed, for a abbr:gp with a covariance
function :math:`k` which has well-defined first and second derivatives
the expected number of times, :math:`N_{u}` the process will cross a
value :math:`u` is

.. raw:: latex

   \begin{equation}
   \label{eq:gp:kernels:crossings}
   \mathbb{E}(Nᵤ) = \frac{1}{2 \pi} \sqrt{ - \frac{ k''(0) }{k(0)} } \exp \left( - \frac{u²}{2k(0)} \right)
   \end{equation}

A zero-mean abbr:gp which has an abbr:se covariance structure will then
cross zero :math:`1/(2 \pi l)` times on average.

Examples of the squared exponential covariance function, and of draws
from a Gaussian process prior which uses this covariance function are
plotted in figure ref:fig:gp:covariance:overviews:se for a variety of
different scale lengths.

.. raw:: latex

   \begin{figure}
   \includegraphics{figures/gp/covariance-ex-overview.pdf}
   \caption[The exponential covariance function]{The \textbf{exponential} covariance function (defined in equation ref:eq:gp:kernels:exp). The panel on the left depicts the value of the kernel as a function of $r = (|\vec{x} - \vec{x}'|)$, at a number of different length scales ($l = 0.25, 0.5, 1.0$) while the panels on the right contain draws from Gaussian processes with exponential covariance with the same length scales as the left panel.
   \label{fig:gp:covariance:overviews:ex}}
   \end{figure}

For data which is not generated by a smooth function a suitable
covariance function may be the exponential covariance function,
:math:`k_{\mathrm{EX}}`, which is defined

.. raw:: latex

   \begin{equation}
   \label{eq:gp:kernels:exp}
   k_{\mathrm{EX}} = \exp\left( - \frac{r}{l} \right),
   \end{equation}

where :math:`r` is the pairwise distance between data and :math:`l` is a
length scale, as in equation ref:eq:gp:kernels:se.

Examples of the exponential covariance function, and of draws from a
Gaussian process prior which uses this covariance function are plotted
in figure ref:fig:gp:covariance:overviews:ex for a variety of different
scale lengths.

For data generated by functions which are smooth, but not necessarily
infinitely differentiable we may turn to the Matérn family of covariance
functions, which take the form

.. raw:: latex

   \begin{equation}
   \label{eq:gp:kernels:mat}
   k_{\mathrm{Mat}}(r) = \frac{1}{2^{\nu - 1} \Gamma{\nu}} 
   \left( \frac{\sqrt{2 \nu}}{l} \right)^{\nu} K_{\nu} 
   \left( \frac{\sqrt{2 \nu}}{l} r \right),
   \end{equation}

for :math:`K_{\nu}` the modified Bessel function of the second kind, and
:math:`\Gamma` the gamma function. As with the previous two covariance
functions :math:`l` is a scale length parameter, and :math:`r` the
distance between two data. A abbr:gp which has a Matérn covariance
function will be :math:`(\lceil x \rceil - 1)`-times differentiable.

While determining an appropriate value of :math:`\nu` during the
training of the abbr:gp is possible, it is common to select a value *a
priori* for this quantity. :math:`\nu=3/2` and :math:`\nu=5/2` are
common choices as :math:`K_{\nu}` can be determined simply, and the
covariance functions are analytic.

The case with :math:`\nu=3/2`, commonly referred to as a
Matérn-\ :math:`3/2` kernel then becomes

.. raw:: latex

   \begin{equation}
   k_{\mathrm{M32}}(r) = \left(1+\frac{\sqrt{3}d}{l}\right) \exp\left( - \frac{\sqrt{3}d}{l} \right).
   \end{equation}

Examples of this covariance function, and example draws from a abbr:gp
using it as a covariance function are plotted in figure
ref:fig:gp:kernels:m32.

Similarly, the Matérn-\ :math:`5/2` is the case where :math:`\nu = 5/2`,
taking the form

.. raw:: latex

   \begin{equation}
   k_{\mathrm{M52}}(r) = 
   \left( 1+\frac{\sqrt{5}d}{l} + \frac{5d^2}{3l^2} \right) 
   \exp \left( - \frac{\sqrt{5}d}{l} \right).
   \end{equation}

Again, examples of this covariance function, and example draws from a
abbr:gp using it as a covariance function are plotted in figure
ref:fig:gp:kernels:m52.

.. raw:: latex

   \begin{figure}
   \includegraphics{figures/gp/covariance-mat32-overview.pdf}
   \caption[The Matérn-$3/2$ covariance function]{The \textbf{Matérn-$3/2$} covariance function (defined in equation ref:eq:gp:kernels:mat, with $\nu = 3/2$). The panel on the left depicts the value of the kernel as a function of $r = (|\vec{x} - \vec{x}'|)$, at a number of different length scales ($l = 0.25, 0.5, 1.0$) while the panels on the right contain draws from Gaussian processes using a Matérn-$3/2$ covariance with the same length scales as the left panel.
   \label{fig:gp:kernels:m32}}
   \end{figure}

.. raw:: latex

   \begin{figure}
   \includegraphics{figures/gp/covariance-mat52-overview.pdf}
   \caption[The Matérn-$5/2$ covariance function]{The \textbf{Mat\'{e}rn-$5/2$} covariance function (defined in equation ref:eq:gp:kernels:mat, with $\nu=5/2$). The panel on the left depicts the value of the kernel as a function of $r = (|\vec{x} - \vec{x}'|)$, at a number of different length scales ($l = 0.25, 0.5, 1.0$) while the panels on the right contain draws from Gaussian processes using Mat\'{e}rn-$5/2$ covariance functions with the same length scales as the left panel.
   \label{fig:gp:kernels:m52}}
   \end{figure}

Data may also be generated from functions with variation on multiple
scales. One approach to modelling such data is to use a abbr:gp with
**rational quadratic** covariance. This covariance function represents a
scale mixture of abbr:rbf covariance functions, each with a different
characteristic length scale. The rational quadratic covariance function
is defined as

.. raw:: latex

   \begin{equation}
   \label{eq:gp:kernels:rq}
   k_{\mathrm{RQ}}(r)  =\left( 1 + \frac{r^2}{2 \alpha l^2}^{-\alpha} \right),
   \end{equation}

where :math:`\alpha` is a parameter which controls the weighting of
small-scale compared to large-scale variations, and :math:`l` and
:math:`r` are the overall length scale of the covariance and the
distance between two data respectively. Examples of this function, at a
variety of different length scales and :math:`\alpha` values, and draws
from abpl:gp which use these functions are plotted in figure
ref:fig:gp:kernels:rq.

.. raw:: latex

   \begin{figure}
   \includegraphics{figures/gp/covariance-rq-overview.pdf}
   \caption[The rational quadratic covariance function]{The \textbf{rational quadratic} covariance function (defined in equation \ref{eq:gp:kernels:rq}). The panel on the left depicts the value of the kernel as a function of $r = (|\vec{x} - \vec{x}'|)$, at a number of different length scales ($l = 0.25, 0.5, 1.0$) while the panel on the right contains draws from Gaussian processes with rational quadratic covariance with the same length scales as the left panel.
   \label{fig:gp:kernels:rq}}
   \end{figure}

This summary of potential covariance functions for use with a abbr:gp is
far from complete (see cite:gpr.book.rw for a more detailed list).
However, these four can be used or combined to produce highly flexible
regression models, as they can be added and multiplied as normal
functions.
