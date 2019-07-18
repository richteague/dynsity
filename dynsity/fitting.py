import time
import emcee
import numpy as np
import scipy.constants as sc
from scipy.interpolate import CubicSpline
from numpy.polynomial.polynomial import Polynomial

class profile(object):

    # constants

    mu = 2.37
    msun = 1.98847e30
    n0 = 1e6
    priors = {}

    def __init__(self, rvals, v_phi, zvals=None, T_gas=None, dvphi=None,
                 maskNaN=True):
        """
        Initialize the class.

        Args:
            rvals (ndarray): Array of radial positions in [au].
            v_phi (ndarray): Array of rotational velocities in [m/s].
            dv_phi (optional[ndarray]): Uncertainties on the rotational
                velocities in [m/s].
            zvals (optional[ndarray]): Array of emission height position in
                [au]. If none is specified, assume all to be 0.
            T_gas (optional[ndarray]): Array of the gas temperatures in [K].
            maskNaN (optional[bool]): If True, mask out all NaN numbers in the
                input arrays. True by default.
        """

        # Read in the arrays.
        self.rvals = rvals
        self.zvals = zvals if zvals is not None else np.zeros(self.rvals.size)
        self.v_phi = v_phi
        self.dvphi = dvphi if dvphi is not None else 0.1 * self.v_phi.copy()
        self.T_gas = T_gas if T_gas is not None else np.ones(self.rvals.size)

        # Mask out the NaN values.
        if maskNaN:
            self._mask_NaN()

        # Check everything is in order.
        self._check_input()

        # Calculate some other default values to speed the fitting.
        self.cs = self._calculate_soundspeed()
        self.rvals_m = self.rvals * sc.au
        self.zvals_m = self.zvals * sc.au
        self._fit_gamma = False
        self._fit_mstar = False
        self._fit_perts = False
        self._fit_npoly = False
        self._smoothprs = False

        # Initialize default fitting params.
        self._set_default_variables()
        self._set_default_priors()

    # -- Main Fitting Function -- #

    def fit_vphi(self, p0, fix_mstar=False, fix_nmol0=False, fix_gamma=False,
                 fix_mdisk=False, smooth=False, nwalkers=64, nburnin=1000,
                 nsteps=1000, niter=None, emcee_kwargs=None,
                 perturbations='gaussian'):
        """
        Coming soon.
        """

        # Verify the MCMC variables.
        p0 = np.atleast_1d(p0)
        nwalkers = int(max(2*p0.size, nwalkers))

        # Unpack the p0 values and assign labels.
        labels = self._get_labels(p0=p0,
                                  fix_mstar=fix_mstar,
                                  fix_nmol0=fix_nnol0,
                                  fix_gamma=fix_gamma,
                                  fix_mdisk=fix_mdisk,
                                  perturbations=perturbations.lower())
        assert p0.size == len(labels), "Wrong number of starting positions."
        ndim = len(labels)

        # Number of iterations to use, roughly one for every 3 free parameters.
        niter = max(1, int(np.ceil(ndim / 3)) if niter is None else niter)

        # Print out messages.
        print("Assuming:\n\tp0 = [%s].\n" % (',\n\t      '.join(labels)))
        print("Running {:d} iteration(s), each with ".format(niter)
              + "{:d} walkers and ".format(nwalkers)
              + "{:d} steps, of which ".format(nburnin + nsteps)
              + "{:d} are for burn-in.".format(nburnin))
        if self._fit_gauss:
            print("The model of n_mol will include "
                  + "{:d} Gaussian perturbations.".format(self._fit_gauss))
        if self._fit_npoly:
            print("The model of n_mol will include a perturbation modelled"
                  + "as a {:d}-order polynomial".format(self._fit_npoly))
        time.sleep(1.0)

        # Sample the posteriors.
        emcee_kwargs = {} if emcee_kwargs is None else emcee_kwargs
        for _ in range(niter):
            sampler = self._run_mcmc(p0=p0, nwalkers=nwalkers, nburnin=nburnin,
                                     nsteps=nsteps, **emcee_kwargs)
            samples = sampler.chain[:, -int(nsteps):]
            samples = samples.reshape(-1, samples.shape[-1])
            p0 = np.median(samples, axis=0)

        # Restore the defaults.
        self._set_default_variables()

        return

    def _set_default_variables(self):
        """Set default variables for fitting."""
        self._fix_mstar = False
        self._fix_nmol0 = False
        self._fix_gamma = False
        self._fix_mdisk = False
        self._fit_npoly = False
        self._fit_gauss = False

    def _check_input(self):
        """Check the input values are correct."""
        assert self.rvals.size == self.zvals.size, "Wrong number of zvals."
        assert self.rvals.size == self.v_phi.size, "Wrong number of v_phi."
        assert self.rvals.size == self.dvphi.size, "Wrong number of dvphi."
        assert self.rvals.size == self.T_gas.size, "Wrong number of T_gas."

    def _get_labels(self, p0, fix_mstar, fix_nmol0, fix_gamma, fix_mdisk,
                    perturbations):
        """Parse the p0 values and return the labels for the plotting."""
        labels = []
        if fix_mstar:
            self._fix_mstar = fix_mstar
        else:
            labels += [r'${\rm M_{star} \,\, (M_{sun})}$']
        if fix_nmol0:
            self._fix_nmol0 = fix_nmol0
        else:
            labels += [r'${\rm n_0 \,\, (cm^{-3})}$']
        if fix_gamma:
            self._fix_gamma = fix_gamma
        else:
            labels += [r'${\rm \gamma}$']
        if fix_mdisk:
            self._fix_mdisk = fix_mdisk
        else:
            labels += [r'${\rm M_{disk} \,\, (M_{sun})}$']
        if perturbations == 'polynomial':
            self._fit_npoly = len(p0) - len(labels)
            labels += ['a_{:d}'.format(i) for i in range(self._fit_npoly)]
        elif perturbations == 'gaussian':
            self._fit_gauss = int((len(p0) - len(labels)) / 3)
            for i in range(self._fit_gauss):
                labels += [r'${\rm r_{%d} \,\, (au)}$' % i,
                           r'${\rm \Delta r_{%d} \,\, (au)}$' % i,
                           r'${\rm log(\delta n_{%d})}$' % i]
        else:
            raise ValueError("`perturbations` must be either 'gaussian'"
                             + " or 'polynomial'.")
        return labels

    # -- MCMC Functions -- #

    def _run_mcmc(self, p0, nwalkers=16, nburnin=1000, nsteps=1000, **kwargs):
        """Explore the posteriors using MCMC."""
        p0 = self._random_p0(p0, kwargs.pop('scatter', 1e-3), nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers, p0.shape[1],
                                        self._fit_density_ln_prob,
                                        pool=kwargs.pop('pool', None))
        sampler.run_mcmc(p0, nburnin + nsteps,
                         progress=kwargs.pop('progress', True))
        return sampler

    # -- Prior Functions -- #

    def set_default_priors(self):
        """Set the default priors."""
        self.set_prior('mstar', [0., 10.], 'flat')
        self.set_prior('gamma', [-20., 0.], 'flat')
        self.set_prior('mdisk', [0., 0.1], 'flat')
        self.set_prior('r', [0.0, self.rvals.max()], 'flat')
        self.set_prior('dr', [np.diff(self.rvals).mean(), 500.], 'flat')
        self.set_prior('log_dn', [-3.0, 3.0], 'flat')

    def set_prior(self, param, args, type='flat'):
        """Set the prior for the given parameter."""
        type = type.lower()
        if type not in ['flat', 'gaussian']:
            raise ValueError("type must be 'flat' or 'gaussian'.")
        if type == 'flat':
            def prior(p):
                if not args[0] <= p <= args[1]:
                    return -np.inf
                return np.log(1.0 / (args[1] - args[0]))
        else:
            def prior(p):
                lnp = np.exp(-0.5 * ((args[0] - p) / args[1])**2)
                return np.log(lnp / np.sqrt(2. * np.pi) / args[1])
        profile.priors[param] = prior

    def _ln_prior(self, theta):
        """Log-prior functions for density profile fits."""
        mstar, gamma, mdisk, perts = self._unpack_theta(theta)
        lnp = profile.priors['mstar'](mstar)
        lnp += 0.0 if gamma is None else profile.priors['gamma'](gamma)
        lnp += 0.0 if mdisk is None else profile.priors['mdisk'](mdisk)
        for n in range(self._fit_perts):
            r, dr, dn = perts[n*3:(n+1)*3]
            lnp += profile.priors['r'](r)
            lnp += profile.priors['dr'](dr)
            lnp += profile.priors['log_dn'](dn)
        return lnp

    # -- p0 Functions -- #

    def _fit_density_ln_prob(self, theta):
        """Log-probability function for density profile fits."""
        lnp = self._ln_prior(theta)
        if np.isfinite(lnp):
            return lnp + self._fit_density_ln_like(theta)
        return -np.inf

    def _fit_density_ln_like(self, theta):
        """Log-likelihood function for density profile fits."""
        v_mod = self.calc_v_phi(theta=theta)
        lnx2 = np.power((self.v_phi - v_mod), 2)
        lnx2 = -0.5 * np.sum(lnx2 * self.dvphi**-2)
        return lnx2 if np.isfinite(lnx2) else -np.inf

    def _unpack_theta(self, theta):
        """Unpack theta."""
        theta = np.atleast_1d(theta)
        mstar, gamma, mdisk, perts = theta[0], None, None, None
        i = 1
        if self._fit_gamma:
            gamma = theta[i]
            i += 1
        if self._fit_mdisk:
            mdisk = theta[i]
            i += 1
        if self._fit_perts:
            perts = theta[i:]
        return mstar, gamma, mdisk, perts

    @staticmethod
    def _random_p0(p0, scatter, nwalkers):
        """Get the starting positions."""
        dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
        dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
        return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)

    # -- Miscellaneous Functions -- #

    def _mask_NaN(self):
        """Mask all the NaN values from all arrays."""
        mask_A = np.isfinite(self.rvals) & np.isfinite(self.zvals)
        mask_B = np.isfinite(self.v_phi) & np.isfinite(self.T_gas)
        mask = mask_A & mask_B
        self.rvals = self.rvals[mask]
        self.zvals = self.zvals[mask]
        self.v_phi = self.v_phi[mask]
        self.T_gas = self.T_gas[mask]

    def _calculate_soundspeed(self, T_gas=None, avg_mu=2.37):
        """Calculate the soundspeed for the gas in [m/s]."""
        T_gas = self.T_gas if T_gas is None else T_gas
        return np.sqrt(sc.k * T_gas / avg_mu / sc.m_p)
