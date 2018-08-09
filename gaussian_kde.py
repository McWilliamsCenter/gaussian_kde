"""
custom implementation of kde for conditional sampling
of multidimensional distributions
"""

from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.stats import gaussian_kde as scipy_gaussian_kde
from numpy.random import multivariate_normal
# The scipy KDTree does not allow for 'seuclidean' distance metric
# so we will use the BallTree class instead.
from sklearn.neighbors import BallTree
import warnings

__all__ = ('gaussian_kde')
__author__ = ('Duncan Campbell')


class gaussian_kde(scipy_gaussian_kde):
    """
    a gaussian kde class with an additional method to allow
    for approximate conditional sampling of the pdf based on
    the scipy.stats.gaussian_kde class

    Representation of a kernel-density estimate using Gaussian kernels.
    Kernel density estimation is a way to estimate the probability density
    function (PDF) of a random variable in a non-parametric way.
    `gaussian_kde` works for both uni-variate and multi-variate data.   It
    includes automatic bandwidth determination.  The estimation works best for
    a unimodal distribution; bimodal or multi-modal distributions tend to be
    oversmoothed.

    Parameters
    ----------
    dataset : array_like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2-D array with shape (# of dims, # of data).
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a scalar,
        this will be used directly as `kde.factor`.  If a callable, it should
        take a `gaussian_kde` instance as only parameter and return a scalar.
        If None (default), 'scott' is used.  See Notes for more details.

    Attributes
    ----------
    dataset : ndarray
        The dataset with which `gaussian_kde` was initialized.

    d : int
        Number of dimensions.

    n : int
        Number of datapoints.

    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`, with which
        the covariance matrix is multiplied.

    covariance : ndarray
        The covariance matrix of `dataset`, scaled by the calculated bandwidth
        (`kde.factor`).

    inv_cov : ndarray
        The inverse of `covariance`.

    Methods
    -------
    evaluate
    __call__
    integrate_gaussian
    integrate_box_1d
    integrate_box
    integrate_kde
    pdf
    logpdf
    resample
    set_bandwidth
    covariance_factor
    conditional_sample

    Notes
    -----
    Bandwidth selection strongly influences the estimate obtained from the KDE
    (much more so than the actual shape of the kernel).  Bandwidth selection
    can be done by a "rule of thumb", by cross-validation, by "plug-in
    methods" or by other means; see [3]_, [4]_ for reviews.  `gaussian_kde`
    uses a rule of thumb, the default is Scott's Rule.
    Scott's Rule [1]_, implemented as `scotts_factor`, is::
        n**(-1./(d+4)),
    with ``n`` the number of data points and ``d`` the number of dimensions.
    Silverman's Rule [2]_, implemented as `silverman_factor`, is::
        (n * (d + 2) / 4.)**(-1. / (d + 4)).
    Good general descriptions of kernel density estimation can be found in [1]_
    and [2]_, the mathematics for this multi-dimensional implementation can be
    found in [1]_.

    References
    ----------
    .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    .. [3] B.A. Turlach, "Bandwidth Selection in Kernel Density Estimation: A
           Review", CORE and Institut de Statistique, Vol. 19, pp. 1-33, 1993.
    .. [4] D.M. Bashtannyk and R.J. Hyndman, "Bandwidth selection for kernel
           conditional density estimation", Computational Statistics & Data
           Analysis, Vol. 36, pp. 279-298, 2001.

    Examples
    --------
    Generate some random two-dimensional data:

    >>> from scipy import stats
    >>> def measure(n):
    ...     "Measurement model, return two coupled measurements."
    ...     m1 = np.random.normal(size=n)
    ...     m2 = np.random.normal(scale=0.5, size=n)
    ...     return m1+m2, m1-m2
    >>> m1, m2 = measure(2000)
    >>> xmin = m1.min()
    >>> xmax = m1.max()
    >>> ymin = m2.min()
    >>> ymax = m2.max()

    Perform a kernel density estimate on the data:

    >>> X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    >>> positions = np.vstack([X.ravel(), Y.ravel()])
    >>> values = np.vstack([m1, m2])
    >>> kernel = stats.gaussian_kde(values)
    >>> Z = np.reshape(kernel(positions).T, X.shape)

    Plot the results:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
    ...           extent=[xmin, xmax, ymin, ymax])
    >>> ax.plot(m1, m2, 'k.', markersize=2)
    >>> ax.set_xlim([xmin, xmax])
    >>> ax.set_ylim([ymin, ymax])
    >>> plt.show()
    """

    def __init__(self, dataset, bw_method=None):
        """
        """
        super(gaussian_kde, self).__init__(dataset, bw_method=bw_method)

    def conditional_sample(self, x, c):
        """
        Return random samples from the distribution :math:`P(y|x)`,
        where :math:`x` and :math:`y` are subsets of the full N-dimensional
        space of the pdf obtained from KDE.  Here :math:`x` is assumed to
        have dimension q and :math:`y` N-q.

        Parameters
        ----------
        x : array_like
            an array of dependent variables of shape (q, n_samples)

        c : array_like
            boolean array defining the conditional dimensions.
            the shape must be (N,)

        Returns
        -------
        y : numpy.array
            array of sampled values of shape (N-q, n_samples)

        Notes
        -----
        This is only an approximate method.  A more robust method
        would be to calculate the full conditional distribution for
        each `x` using a different KDE method.
        """

        x = np.atleast_1d(x)
        c = np.atleast_1d(c)
        size = len(x)

        if x.ndim == 1:
            x = x[..., np.newaxis]

        # find the nearest neighbor to each x.
        V = np.diagonal(self.covariance)
        d = self.dataset.T
        tree = BallTree(d[:, c], metric='seuclidean', V=V[c])

        # add a random variable to each x.
        mask = np.ix_(c, c)
        cov_xx = np.atleast_2d(self.covariance[mask])
        dim_x = np.sum(c)
        x = x + multivariate_normal(np.zeros((dim_x,), float),
                                    cov_xx, size=size)

        # return index into training sample and the normalized distance.
        d, indices = tree.query(x, k=1, return_distance=True)

        # if the training dataset does not span the range of `x`,
        # then this may not be the best algorithm to use.
        if np.max(d) > 3.0:
            msg = ("Training dataset may not be suffucient \n"
                   "to sample over the range of `x`. \n"
                   "The maximum normalzed distance is: {0}".format(np.max(d)))
            warnings.warn(msg)

        # for each matched point in the training dataset
        # add a random variable get a sample for y.
        mask = np.ix_(~c, ~c)
        cov_yy = np.atleast_2d(self.covariance[mask])
        dim_y = self.d - np.sum(c)
        norm = multivariate_normal(np.zeros((dim_y,), float),
                                   cov_yy, size=size)
        means = self.dataset[~c, indices]

        return means + norm
