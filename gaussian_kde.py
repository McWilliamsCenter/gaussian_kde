"""
A class for (uni/multi)-variate Gaussian kernel density estimation with added methods for conditional sampling.
"""

from __future__ import division, print_function, absolute_import

from scipy.stats import gaussian_kde as scipy_gaussian_kde
from scipy.stats import multivariate_normal

import numpy as np

__all__=['gaussian_kde']
__author__ = ['Duncan Campbell']

class gaussian_kde(scipy_gaussian_kde):
    """
    """
    def __init__(self, dataset, bw_method=None):
        """
        """

        self.dataset = np.atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        super(gaussian_kde, self).__init__(dataset, bw_method=bw_method)

    def conditional_resample(self, cx, c, size=None):
        """
        Perform a conditional random sampling from the estimated pdf given a set of values to condition on.

        Parameters
        ==========
        cx : array_like
             a vector of length Nc specifying the values of the independant variables.

        c : array_like
            a boolean vector indicating the dimensions of the independant variables.

        size : int, optional
            The number of samples to draw.  If not provided, then the size is
            the same as the underlying dataset.

        Returns
        =======
        resample : (self.d, `size`) ndarray
            The sampled dataset.
        """

        if size is None:
            size = self.n

        dists = map(lambda x: multivariate_normal(x, self.covariance), self.dataset.T)  # list of mvns

        # get marginalized (over the not conditioned dimensions)
        mps = [self.marginalized_multivariate_normal(p, ~c) for p in dists]
        # evaulate each marginalized distribution at cx
        norms = np.array([mp.pdf(cx) for mp in mps])
        norms = norms/np.sum(norms)  # normalize
        #norms = norms*0.0 + 1.0/len(norms)

        # get list of conditional distributions for each point
        cdists = np.array([self._conditional_multivariate_gaussian(dist, cx, c) for dist in dists])

        # get a random number for each sampled point
        ran_num = np.random.random(size=size)

        # sort distributions by contribution at cx
        sort_inds = np.argsort(norms)
        norms = norms[sort_inds]
        cdists = cdists[sort_inds]

        self.cdists = np.copy(cdists)
        self.norms = np.copy(norms)

        # choose a distribution to sample from
        inds = np.searchsorted(norms, ran_num)-1
        #inds = range(0, len(cdists))
        self.sinds = inds

        # sample from selected distributions
        s = [dist.rvs() for dist in cdists[inds]]

        return s

    def _conditional_multivariate_gaussian(self, p, cx, c):
        """
        Given a multivariate Gaussian distribution, return a new conditional (uni/multi)-variate Gaussian.

        Parameters
        ==========
        p : scipy.stats.multivariate_normal object
            A multivariate normal random variable.

        cx : array_like
            a vector of length Nc specifying the values of the independant variables.

        c : array_like
            a boolean vector indicating the dimensions of the independant variables.

        Returns
        =======
        cp : scipy.stats.multivariate_normal
            A multivariate normal random variable.
        """

        cx = np.atleast_1d(cx)
        c = np.atleast_1d(c)

        mus = np.atleast_1d(p.mean)
        cov = np.atleast_1d(p.cov)

        # seperate components
        # means
        mu1 = mus[~c]
        mu2 = mus[c]

        # covariance matrices
        mask11 = np.ix_(~c, ~c)
        cov11 = cov[mask11]
        mask12 = np.ix_(~c, c)
        cov12 = cov[mask12]
        mask21 = np.ix_(c, ~c)
        cov21 = cov[mask21]
        mask22 = np.ix_(c, c)
        cov22 = cov[mask22]

        # number of dimensions for conditional distribution
        ndim = np.sum(~c)

        # calculate new parmaters for distribution
        mu_bar = mu1 + cov12*np.linalg.inv(cov22)*(np.matrix(cx - mu2).reshape(ndim, 1))
        cov_bar = cov11 - cov12*np.linalg.inv(cov22)*cov21

        return multivariate_normal(mean=mu_bar, cov=cov_bar)

    def evaluate_marginalized(self, points, m):
        """
        """
        dists = map(lambda x: multivariate_normal(x, self.covariance), self.dataset.T)

        # get marginalized (over the not conditioned dimensions)
        mps = [self.marginalized_multivariate_normal(p, ~c) for p in dists]

        points = np.atleast_2d(points)

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = np.reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d, self.d)
                raise ValueError(msg)

        result = np.zeros((m,), dtype=float)

        if m >= self.n:
            # there are more points than data, so loop over data
            for i in range(self.n):
                diff = self.dataset[:, i, np.newaxis] - points
                tdiff = np.dot(self.inv_cov, diff)
                energy = sum(diff*tdiff, axis=0) / 2.0
                result = result + np.exp(-energy)
        else:
            # loop over points
            for i in range(m):
                diff = self.dataset - points[:, i, np.newaxis]
                tdiff = np.dot(self.inv_cov, diff)
                energy = sum(diff * tdiff, axis=0) / 2.0
                result[i] = sum(np.exp(-energy), axis=0)

        result = result / self._norm_factor

        return result

    def marginalized_multivariate_normal(self, p, m):
        """
        marginalize over a set of dimensions of a multivariate normal
        """
        m = np.atleast_1d(m)

        mus = np.atleast_1d(p.mean)
        cov = np.atleast_1d(p.cov)

        # seperate components
        # means
        mu_bar = mus[~m]

        # covariance matrices
        mask = np.ix_(~m, ~m)
        cov_bar = cov[mask]

        return multivariate_normal(mean=mu_bar, cov=cov_bar)





