import time
import emcee
import numpy as np
import scipy.constants as sc
from scipy.interpolate import CubicSpline
from numpy.polynomial.polynomial import Polynomial


class profile(object):
    """
    Class containing the radial profiles.

    Args:
        rvals (ndarray): Array of radial positions in [au].
        v_phi (ndarray): Array of rotational velocities in [m/s].
        dv_phi (Optional[ndarray]): Uncertainties on the rotational
            velocities in [m/s].
        zvals (Optional[ndarray]): Array of emission height position in
            [au]. If ``None`` is specified, assume all to be 0.
        hvals (Optional[ndarray]): Array of scale height values in [au]. If
            ``None``, will assume :math:`h/r = 0.1`.
        T_gas (Optional[ndarray]): Array of the gas temperatures in [K]. If
            ``None`` will assume 1 everywhere.
        maskNaN (Optional[bool]): If ``True``, mask out all NaN numbers in
            the input arrays. True by default.
    """

    # constants

    mu = 2.37
    msun = 1.98847e30
    n0 = 1e6
    priors = {}

    def __init__(self, rvals, v_phi, zvals=None, hvals=None, T_gas=None, dvphi=None,
                 maskNaN=True):
        """Instantiate the class."""

        # Read in the arrays.
        self.rvals = rvals
        self.zvals = np.zeros(self.rvals.size) if zvals is None else zvals
        self.hvals = 0.1 * self.rvals if hvals is None else hvals
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
        self._set_grids()

    # -- Main Fitting Function -- #

    def fit_vphi(self, p0, fix_mstar=False, fix_nmol0=False, fix_gamma=False,
                 fix_mdisk=False, smooth=False, include_pressure=True,
                 include_selfgravity=True, nwalkers=64, nburnin=1000,
                 nsteps=1000, niter=None, emcee_kwargs=None,
                 perturbations='gaussian', plot_all=True):
        """
        Coming soon.
        """

        # Verify the MCMC variables.
        p0 = np.atleast_1d(p0)
        nwalkers = int(max(2*p0.size, nwalkers))

        # Unpack the p0 values and assign labels.
        labels = self._get_labels(p0=p0,
                                  fix_mstar=fix_mstar,
                                  fix_nmol0=fix_nmol0,
                                  fix_gamma=fix_gamma,
                                  fix_mdisk=fix_mdisk,
                                  perturbations=perturbations.lower())
        assert p0.size == len(labels), "Wrong number of starting positions."
        ndim = len(labels)

        # Whether to include pressure or self-gravity terms in v_phi.
        self._incl_grv = bool(include_selfgravity)
        self._incl_prs = bool(include_pressure)

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
                  + " as a {:d}th-order polynomial".format(self._fit_npoly))
        time.sleep(1.0)

        # Sample the posteriors.
        emcee_kwargs = {} if emcee_kwargs is None else emcee_kwargs
        for _ in range(niter):
            sampler = self._run_mcmc(p0=p0, nwalkers=nwalkers, nburnin=nburnin,
                                     nsteps=nsteps, **emcee_kwargs)
            samples = sampler.chain[:, -int(nsteps):]
            samples = samples.reshape(-1, samples.shape[-1])
            p0 = np.median(samples, axis=0)

        # Diagnostic plots. Only plot walkers for [mstar, mdisk, gamma].
        if plot_all:
            idx = len(p0)
        else:
            idx = int(not fix_mstar) + int(not fix_mdisk) + int(not fix_gamma)
        self.plot_walkers(sampler.chain.T[:idx], nburnin=nburnin,
                          labels=labels[:idx])

        self.plot_v_phi(v_mod=self._calc_v_phi(np.median(samples, axis=0)))
        self.plot_n_mol(samples, N=100)

        # Restore the defaults.
        self._set_default_variables()

        return samples

    def _set_default_variables(self):
        """Set default variables for fitting."""
        self._fix_mstar = False
        self._fix_gamma = False
        self._fix_mdisk = False
        self._fit_npoly = False
        self._fit_gauss = False
        self._incl_prs = True
        self._incl_grv = True

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
        if fix_mdisk:
            self._fix_mdisk = fix_mdisk
        else:
            labels += [r'${\rm M_{disk} \,\, (M_{sun})}$']
        if fix_gamma:
            self._fix_gamma = fix_gamma
        else:
            labels += [r'${\rm \gamma}$']
        if perturbations == 'polynomial':
            self._fit_npoly = len(p0) - len(labels)
            labels += [r'${{\rm a_{:d}}}$'.format(i)
                       for i in range(self._fit_npoly)]
        elif perturbations == 'gaussian':
            self._fit_gauss = int((len(p0) - len(labels)) / 3)
            for i in range(self._fit_gauss):
                labels += [r'${\rm r_{%d} \,\, (au)}$' % i,
                           r'${\rm \Delta r_{%d} \,\, (au)}$' % i,
                           r'${\rm \delta n_{%d}}$' % i]
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

    def _fit_density_ln_prob(self, theta):
        """Log-probability function for density profile fits."""
        lnp = self._ln_prior(theta)
        if np.isfinite(lnp):
            return lnp + self._fit_density_ln_like(theta)
        return -np.inf

    def _fit_density_ln_like(self, theta):
        """Log-likelihood function for density profile fits."""
        v_mod = self._calc_v_phi(theta=theta)
        lnx2 = np.power((self.v_phi - v_mod), 2)
        lnx2 = -0.5 * np.sum(lnx2 * self.dvphi**-2)
        return lnx2 if np.isfinite(lnx2) else -np.inf

    # -- Prior Functions -- #

    def _set_default_priors(self):
        """Set the default priors."""
        self.set_prior('mstar', [0., 10.], 'flat')
        self.set_prior('gamma', [-10, 0.], 'flat')
        self.set_prior('mdisk', [0., 0.1], 'flat')
        self.set_prior('r', [0.0, self.rvals.max()], 'flat')
        self.set_prior('dr', [np.diff(self.rvals).mean(), 500.], 'flat')
        self.set_prior('dn', [-1e3, 1.0], 'flat')
        self.set_prior('a_i', [-1e4, 1e4], 'flat')

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
        mstar, mdisk, gamma, perts = self._unpack_theta(theta)
        lnp = profile.priors['mstar'](mstar)
        lnp += profile.priors['mdisk'](mdisk)
        lnp += profile.priors['gamma'](gamma)
        if perts is None:
            return lnp

        if self._fit_gauss:
            for n in range(self._fit_gauss):
                r, dr, dn = perts[n*3:(n+1)*3]
                lnp += profile.priors['r'](r)
                lnp += profile.priors['dr'](dr)
                lnp += profile.priors['dn'](dn)
        elif self._fit_npoly:
            for a_i in perts:
                lnp += profile.priors['a_i'](a_i)
        return lnp

    # -- p0 Functions -- #

    def _unpack_theta(self, theta):
        """Unpack theta based on the fixed variables."""
        theta = np.atleast_1d(theta)
        i = 0
        if self._fix_mstar:
            mstar = self._fix_mstar
        else:
            mstar = theta[i]
            i += 1
        if self._fix_mdisk:
            mdisk = self._fix_mdisk
        else:
            mdisk = theta[i]
            i += 1
        if self._fix_gamma:
            gamma = self._fix_gamma
        else:
            gamma = theta[i]
            i += 1
        if self._fit_npoly or self._fit_gauss:
            perts = theta[i:]
        else:
            perts = None
        return mstar, mdisk, gamma, perts

    @staticmethod
    def _random_p0(p0, scatter, nwalkers):
        """Get the starting positions."""
        dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
        dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
        return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)

    # -- Velocity Functions -- #

    def _calc_v_phi(self, theta):
        """Calculate the rotation velocity in [m/s]."""

        # Unpack the variables.
        mstar, mdisk, gamma, perts = self._unpack_theta(theta)

        # Keplerian rotation.
        v_kep2 = self._calc_v_kep2(mstar)
        if not self._incl_grv and not self._incl_prs:
            return np.sqrt(np.where(v_kep2 < 0.0, 0.0, v_kep2))
        sigma = self._calc_sigma(gamma, mdisk)

        # Pressure Correction.
        if self._incl_prs:
            n_mol = self._calc_n_mol(sigma, perts)
            v_prs2 = self._calc_v_prs2(n_mol)
        else:
            v_prs2 = 0.0

        # Self-gravity correction.
        if self._incl_grv:
            v_grv2 = self._calc_v_grv2(gamma, mdisk)
        else:
            v_grv2 = 0.0

        # Combination.
        v_phi2 = v_kep2 + v_prs2 + v_grv2
        return np.sqrt(np.where(v_phi2 < 0.0, 0.0, v_phi2))

    def _calc_v_prs2(self, n_mol):
        """Calculate the pressure correction term."""
        return self.rvals_m * self._dPdR(n_mol) / self._n2rho(n_mol)

    def _calc_v_kep2(self, mstar):
        """Calculate the Keplerian rotation term."""
        v_kep2 = sc.G * mstar * self.msun * self.rvals_m**2
        v_kep2 *= (self.rvals_m**2 + self.zvals_m**2)**(-3./2.)
        return v_kep2

    def _calc_v_grv2(self, gamma, mdisk):
        """Calculate the disk self-gravity correction term."""
        mass = self._calc_sig0(gamma, mdisk)
        mass *= np.power(self.rgrid / 100., gamma)
        mass = np.where(self.mask, mass * self.area, 0.0)
        phi = np.zeros(self.axis.size)
        for i in range(self.axis.size):
            phi_cell = mass.copy()
            phi_cell[i, self.idx0] = 0.0
            phi[i] = np.nansum(-sc.G * phi_cell / self.dist[i])
        phi = phi[-len(self.rvals):]
        #phi = np.interp(self.rvals, self.axis, phi[self.idxs])
        return np.gradient(phi, self.rvals_m)

    @staticmethod
    def _dphidR(sigma):
        """Gravitational potential gradient for a given sigma in [g/cm^2]."""
        return 2. * np.pi * sc.G * sigma * 1e3

    def _dPdR(self, n_mol):
        """Pressure gradient for a given n_mol in [/cm^3]."""
        P = n_mol * sc.k * self.T_gas * 1e6
        if self._smoothprs:
            return CubicSpline(self.rvals_m, P)(self.rvals_m, 1)
        return np.gradient(P, self.rvals_m)

    # -- Density Functions -- #

    def _n2rho(self, n):
        """Convert n in [/cm^3] to rho in [kg/m^3]."""
        return n * self.mu * sc.m_p * 1e6

    def _rho2n(self, rho):
        """Convert rho in [kg/m^3] to n in [/cm^3]."""
        return rho / self.mu / sc.m_p / 1e6

    def _calc_sig0(self, gamma, mdisk):
        """Return the normalization constant in [kg/m^2]."""
        if gamma <= -2.0:
            rmin, rmax = self.rvals_m[0], self.rvals_m[-1]
            sig0 = mdisk * self.msun * (2+gamma) * (100. * sc.au)**gamma
            sig0 /= 2.0 * np.pi * (rmax**(2+gamma) - rmin**(2+gamma))
        else:
            sig0 = mdisk * self.msun * (100. * sc.au)**gamma / 2.0 / np.pi
            sig0 /= np.trapz(self.rvals_m**(gamma + 1), x=self.rvals_m)
        return sig0

    def _calc_sigma(self, gamma, mdisk):
        """Return sigma in [kg/m^2] for a given disk mass in [Msun]."""
        sig0 = self._calc_sig0(gamma=gamma, mdisk=mdisk)
        return sig0 * np.power(self.rvals / 100., gamma)

    def _calc_enclosed_mass(self, gamma, mdisk):
        """Return the enclosed mass [kg] as a function of radius."""
        sig0 = self._calc_sig0(gamma=gamma, mdisk=mdisk)
        mass = self.rvals_m**(2. + gamma) - self.rvals_m[0]**(2. + gamma)
        mass *= 2. * np.pi * sig0 * (100. * sc.au)**gamma
        return mass / (2. + gamma)

    def _sample_n_mol(self, samples, N=100, percentiles=False,
                      perturbations_only=False):
        """
        Draw random samples of n_mol from the sampled posterior distributions.

        Args:
            samples (ndarray): Samples of the posterior distributions from
                ``fit_vphi`` including the ``mstar``, ``gamma`` and ``mdisk``
                values. Should be of shape ``[nsamples, ndim]``.
            N (Optional[int]): Number of random samples to draw.
            percentiles (Optional[bool]): If ``True``, take the 16th, 50th and
                84th percentiles of the ``n_mol`` samples to estimate the
                uncertainty.
            perturbations_only (Optional[bool]): If ``True``, only plot the
                perturbations rather than the full radial profile.

        Returns:
            n_mol (ndarray): Array of the ``n_mol`` samples. If
                ``percentiles=True`` values will be a ``[3, rvals.size]``
                shaped array with ``[y, -dy, +dy]`` values based on the 16th,
                50th and 84th percentiles of the ``N`` random samples.
                Otherwise will be a ``[N, rvals.size]`` array.
        """
        n_mol = []
        for i in np.random.randint(0, samples.shape[0], N):
            _, mdisk, gamma, perts = self._unpack_theta(samples[i])
            n_tmp = self._calc_n_mol(self._calc_sigma(gamma, mdisk), perts)
            if perturbations_only:
                n_tmp /= self._calc_n_mol(self._calc_sigma(gamma, mdisk), None)
            n_mol += [n_tmp]
        if percentiles:
            n_mol = np.percentile(n_mol, [16, 50, 84], axis=0)
            n_mol = np.array([n_mol[1], n_mol[1]-n_mol[0], n_mol[2]-n_mol[1]])
        return np.array(n_mol)

    # -- Perturbation Functions -- #

    def _calc_n_mol(self, sigma, perts):
        """Gas volume density in [/cm^3]."""
        n_mol = sigma * np.exp(-0.5 * (self.zvals / self.hvals)**2)
        n_mol /= np.sqrt(2. * np.pi) * self.hvals * sc.au
        n_mol = self._rho2n(n_mol)
        if perts is not None:
            if self._fit_npoly:
                dn_mol = self.polynomial_perturbation(perts)
            else:
                dn_mol = self.gaussian_perturbation_multi(perts)
        else:
            return n_mol
        return n_mol * dn_mol

    @staticmethod
    def gaussian(x, x0, dx, A):
        """Gaussian function."""
        return A * np.exp(-0.5 * np.power((x-x0)/dx, 2))

    def gaussian_perturbation(self, x0, dx, A):
        """Gaussian perturbation function."""
        return 1.0 - profile.gaussian(self.rvals, x0, dx, A)

    def gaussian_perturbation_multi(self, perts):
        """Multiple Gaussian perturbations."""
        dp = np.ones(self.rvals.size)
        if perts is not None:
            for i in range(int(len(perts) / 3)):
                dp *= self.gaussian_perturbation(perts[3*i],
                                                 perts[3*i+1],
                                                 perts[3*i+2])
        return dp

    def polynomial_perturbation(self, coeffs):
        """Polynomial pertubration vector."""
        return Polynomial(coeffs)(self.rvals)

    # -- Miscellaneous Functions -- #

    def _set_grids(self):
        """Create the grids for the calculation of self-gravity."""
        self.axis = np.linspace(-self.rvals[-1], self.rvals[-1],
                                 self.rvals.size * 2 + 1)
        xgrid, ygrid = np.meshgrid(self.axis, self.axis)
        self.rgrid = np.hypot(xgrid, ygrid)
        self.area = np.power(np.diff(self.axis).mean() * sc.au, 2.0)
        self.mask = np.logical_and(self.rgrid >= self.rvals[0],
                                    self.rgrid <= self.rvals[-1])
        self.idx0 = abs(self.axis).argmin()

        # Calcaulte the distances to each of the pixels for each pixel along
        # the x-axis to speed up the gravitational potential calculation.
        self.dist = np.array([np.hypot(xgrid[0, self.idx0] - xgrid,
                                       ygrid[j, 0] - ygrid) * sc.au
                              for j in range(self.axis.size)])

        # Sort the axis for quicker interpolation.
        self.idxs = np.argsort(abs(self.axis))
        self.axis = abs(self.axis)[self.idxs]

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

    def set_zvals(self, z0, psi, z1=0.0, phi=1.0):
        """
        Set an analytical emission surface given by,

        .. math::
            z(r) = z_0 \cdot \left( \frac{r}{100~{\rm au}} \right)^{\psi} + z_0 \cdot \left( \frac{r}{100~au} \right)^{\varphi}

        which is frequently used in the literature.

        Args:
            z0 (float): Emission surface height at 100 au in [au].
            psi (float): Flaring angle.
            z1 (Optional[float]): Correction term at 100 au in [au].
            phi (Optional[float]): Flaring angle of correction term.
        """
        self.zvals = z0 * np.power(self.rvals / 100., psi)
        self.zvals += z1 * np.power(self.rvals / 100., phi)

    def set_hvals(self, h0, psi):
        """
        Set an analytical scale height profile given by

        .. math::
            h(r) = h_0 \times \left( \frac{r}{100~{\rm au}} \right)^{\psi}

        Args:
            h0 (float): Scale height at 100 au in [au].
            psi (float): Flaring angle of the scale height profile.
        """
        self.hvals = h0 * np.power(self.rvals / 100., psi)

    # -- Plotting Functions -- #

    @staticmethod
    def plot_walkers(samples, nburnin=None, labels=None, return_fig=False,
                     plot_kwargs=None, histogram=True):
        """
        Plot the walkers to check if they are burning in.

        Args:
            samples (ndarray): Flattened array of posterior samples.
            nburnin (Optional[int]): Number of steps used to burn in. If
                provided will annotate the burn-in region.
            labels (Optional[list]): Labels for the different parameters.
            histogram (Optional[bool]): Include a histogram of the PDF samples
                at the right-hand side of the plot.
            return_fit (Optional[bool]): If True, return the figure.
            plot_kwargs (Optional[dict]): Kwargs used for ax.plot().

        Returns:
            fig (matplotlib figure): If requested.
        """

        # Imports.
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

        # Check the length of the label list.
        if labels is not None:
            if samples.shape[0] != len(labels):
                raise ValueError("Not correct number of labels.")

        # Cycle through the plots.
        if plot_kwargs is None:
            plot_kwargs = {}
        for s, sample in enumerate(samples):
            lc = plot_kwargs.get('c', plot_kwargs.get('color', 'k'))
            la = plot_kwargs.get('alpha', 0.1)
            fig, ax = plt.subplots(figsize=(4.0, 2.0))
            for walker in sample.T:
                ax.plot(walker, color=lc, alpha=la, zorder=-1, **plot_kwargs)
            ax.set_xlabel('Steps')
            ax.set_xlim(0, len(walker))
            if labels is not None:
                ax.set_ylabel(labels[s])
            y0 = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            if nburnin is not None:
                ax.axvspan(0, nburnin, facecolor='w', alpha=0.75, zorder=1)
                ax.axvline(nburnin, ls=':', color='k')
                ax.text(nburnin / 2.0, y0 + ax.get_ylim()[1], 'burn-in',
                        ha='center', va='bottom', fontsize=8)
                x0 = (len(walker) - nburnin) / 2.0 + nburnin
            else:
                x0 = len(walker) * 0.5
            ax.text(x0, y0 + ax.get_ylim()[1], 'sample PDF', ha='center',
                    va='bottom', fontsize=8)
            ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])

            # Add the aide histogram if required.
            if histogram:
                fig.set_size_inches(6.0, 2.0, forward=True)
                ax_divider = make_axes_locatable(ax)

                # Make the histogram.
                bins = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 50)
                hist, _ = np.histogram(sample[nburnin:].flatten(), bins=bins,
                                       density=True)
                bins = np.average([bins[1:], bins[:-1]], axis=0)

                # Plot it.
                ax1 = ax_divider.append_axes("right", size="25%", pad="2%")
                ax1.fill_betweenx(bins, hist, np.zeros(bins.size), step='mid',
                                  color='darkgray', lw=0.0)
                ax1.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
                ax1.set_xlim(0, ax1.get_xlim()[1])

                # Gentrification.
                ax1.set_yticklabels([])
                ax1.set_xticklabels([])
                ax1.tick_params(which='both', left=0, bottom=0, right=0, top=0)
                ax1.spines['right'].set_visible(False)
                ax1.spines['bottom'].set_visible(False)
                ax1.spines['top'].set_visible(False)

        # Return the figure.
        if return_fig:
            return fig

    def plot_v_phi(self, v_mod=None, plot_residual=True, return_fig=False):
        """
        Plot the rotation profile.

        Args:
            v_mod (Optional[ndarray]): Model v_phi profile to compared to the
                data.
            plot_residual (Optional[bool]): If a model is provided, plot the
                residual at the bottom of the axis.
            return_fig (Optional[bool]): Return the figure.
        """

        # Imports.
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

        # Basic plot.
        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        ax.errorbar(self.rvals, self.v_phi, self.dvphi, color='k', fmt=' ',
                    linewidth=0.8, capsize=1.0, capthick=1.0)
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
        if v_mod is not None:
            ax.plot(self.rvals, v_mod, color='orangered')
        ax.set_ylabel(r'${\rm v_{\phi} \quad (m\,s^{-1})}$')
        ax.set_xlim(self.rvals[0], self.rvals[-1])

        # Include a residual box.
        if v_mod is not None and plot_residual:
            fig.set_size_inches(5.5, 5.0, forward=True)
            ax_divider = make_axes_locatable(ax)
            ax1 = ax_divider.append_axes("bottom", size="40%", pad="2%")
            ax1.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
            ax.set_xticks([])
            ax1.set_xlabel('Radius (au)')
            ax1.axhline(0.0, color='orangered', lw=1.0, zorder=-10)
            ax1.errorbar(self.rvals, self.v_phi - v_mod, self.dvphi,
                         color='k', fmt=' ', linewidth=0.8, capsize=1.0,
                         capthick=1.0)
            ax1.set_ylabel(r'${\rm v_{\phi} - v_{\rm mod} \quad (m\,s^{-1})}$')
        else:
            ax.set_xlabel('Radius (au)')
        if return_fig:
            return fig

    def plot_n_mol(self, samples, N=100, log_axes=True, percentiles=False,
                   perturbations_only=False, return_fig=False):
        """
        Plot the molecular number density including perturbations.

        Args:
            samples (ndarray): Samples of the posterior distributions for which
                has shape ``[nsamples, ndim]``.
            N (Optional[int]): Number of random samples to draw.
            log_axes (Optional[bool]): If ``True``, plot logarithmic y-axis.
            percentiles (Optional[bool]): If ``True``, take the 16th, 50th and
                84th percentiles of the ``n_mol`` samples to estimate the
                uncertainty.
            perturbations_only (Optional[bool]): If ``True``, only plot the
                perturbations rather than the full radial profile.
            return_fig (Optional[bool]): If ``True``, return the figure.
        """

        # Get the samples.
        n_mol = self._sample_n_mol(samples=samples, N=N,
                                   percentiles=percentiles,
                                   perturbations_only=perturbations_only)

        # Plot the figure.
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
        if percentiles:
            ax.errorbar(self.rvals, n_mol[0], n_mol[1:], color='k', lw=0.8,
                        capsize=1.0, capthick=1.0, fmt=' ')
        else:
            ax.plot(self.rvals, n_mol.T, color='k', lw=1.0, alpha=float(5/N))
        if log_axes:
            ax.set_yscale('log')
        ax.set_xlabel('Radius (au)')
        if perturbations_only:
            ax.set_ylabel(r'${\rm n(H_2) \, / \, n_0(H_2)}$')
        else:
            ax.set_ylabel(r'${\rm n(H_2) \quad (cm^{-3})}$')
        if return_fig:
            return fig
