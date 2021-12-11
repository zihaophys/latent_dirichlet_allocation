import numpy as np
from scipy.special import digamma, polygamma

import numpy as np
from numpy.random import default_rng
rng = default_rng()

class LDA:

    def parameter_estimation(corpus, k, tol=1e-6, max_iter=100):

        M = len(corpus)
        V = corpus[0].shape[1]
        Nd_list = [corpus[i].shape[0] for i in range(M)]

        # Initialization
        alpha0 = rng.gamma(shape=100, scale = 0.01, size=k)
        beta0 = rng.dirichlet(np.ones(V), size=k)
        phi = [np.ones((Nd, k))/k for Nd in Nd_list]
        gam = np.array([alpha0 + phi[d].sum(axis=0) for d in range(M)])
        # gam = np.array([[alpha0 + phi[0].sum(axis=0)] * M)]

        # Iteration
        for i in range(max_iter):
            print('Iteration: {}'.format(i))
            # updat beta
            beta = np.zeros((k, V))
            for d in range(M):
                phi[d], gam[d,] = LDA.mean_field(corpus[d], alpha0, beta0, phi[d], gam[d,], tol, max_iter)
                beta += phi[d].T @ corpus[d]

            beta = beta/np.sum(beta, axis=1)[:, np.newaxis]
            # perform Newton-Ralphson method to update alpha
            g = M * (digamma(np.sum(alpha0)) - digamma(alpha0)) + np.sum(digamma(gam) - digamma(np.sum(gam, axis=1))[:, np.newaxis], axis=0)
            z = M  * polygamma(1, np.sum(alpha0))
            h = - M * polygamma(1, alpha0)
            c = np.sum(g/h) / (1/z + np.sum(1/h))
            alpha = alpha0 - (g - c)/h 

            if (np.linalg.norm(alpha - alpha0) < tol) and (np.linalg.norm(beta - beta0) < tol):
                print('Converged in {} iterations'.format(i))
                break

            alpha0 = alpha
            beta0 = beta

        return alpha, beta


    def mean_field(doc, alpha, beta, phi0, gamma0, tol=1e-6, max_iter=100):
        '''
        Given alpha and beta, for each document,
        use variational inference to compute gamma and phi
        '''
        # Initialize
        phi = phi0
        gamma = gamma0

        # Iterate
        for i in range(max_iter):
            phi = (doc@beta.T)* np.exp(digamma(gamma) - digamma(np.sum(gamma)))
            phi = phi/(np.sum(phi, axis=1)[:, np.newaxis])
            gamma = alpha + np.sum(phi, axis=0)
            if (np.linalg.norm(gamma - gamma0) < tol) and (np.linalg.norm(phi - phi0) < tol):
                break
            gamma0 = gamma
            phi0 = phi

        return phi, gamma