.. methodology:

Methodology
===========

In this page we give a brief overview of the modelling process used in ``dynsity``. We refer the interested reader to `Takeuchi & Lin (2002) <https://ui.adsabs.harvard.edu/abs/2002ApJ...581.1344T>`_ for more thorough details.

Background
----------

The gas in a protoplanetary disk will rotate at a slightly sub-Keplerian velocity due to the support from the radial pressure gradient. In addition, the mass of the disk will contribute to the gravitational potential and speed the rotation of the disk, such that the total rotation is given by,

.. math::
    \frac{v_{\phi}^2}{r} = \frac{GM_{\rm star}r}{(r^2 + z^2)^{3/2}} + \frac{1}{\rho_{\rm gas}} \frac{\partial P_{\rm gas}}{\partial r} + \frac{\partial \phi_{\rm gas}}{\partial r},

where, for an ideal gas, :math:`P_{\rm gas} = n_{\rm gas} \, k \, T_{\rm gas}`, and :math:`\phi_{\rm gas}` is the gravitational potential of the gas which satisfies

.. math::
    \nabla^2 \phi_{\rm gas} = 4 \pi G \rho_{\rm gas}.

If we are able to measure :math:`v_{\phi}` precisely, and are able to constrain both :math:`z(r)` and :math:`T_{\rm gas}(r)` observationally, we hope to

1) Place tight constraints on the dynamical mass of the star, :math:`M_{\rm sun}`.

2) Infer local changes in :math:`P_{\rm gas}` due to local deviations in :math:`v_{\phi}`, such as due to gaps in the gas surface density.

3) Constrain the dynamical mass of the disk after making some assumptions about how :math:`\Sigma_{\rm gas}(r)` varies.

The Model
---------

Inputs
^^^^^^

We assume that the user has been able to measure:

* :math:`v_{\phi}(r)` - The deprojected rotational velocity of the gas as a function of radius in :math:`[{\rm m\,s^{-1}}]`.

* :math:`z(r)` - The height of the emission surface as a function of radius in :math:`[{\rm au}]`.

* :math:`h_p(r)` - The gas pressure scale height as a function of radius in :math:`[{\rm au}]`.

* :math:`T_{\rm gas}(r)` - The gas temperature as a function of radius in :math:`[{\rm K}]`.

The emission surface can be derived either from fitting the rotation map, as in `Keppler et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019A%26A...625A.118K>`_, or following the method in `Pinte et al. (2018) <https://ui.adsabs.harvard.edu/abs/2018A%26A...609A..47P>`_. The gas temperature is harder to measure, but optically thick lines like CO are useful as :math:`T_{\rm B} = T_{\rm ex}`. Optically thin lines may pose more of a challenge.

Gas Surface Density
^^^^^^^^^^^^^^^^^^^

We make the assumption that the gas surface density is well described by

.. math::
    \Sigma_{\rm gas} = \Sigma_0 \times \left( \frac{r}{100~{\rm au}}\right)^{\gamma},

where we have neglected the often used exponential edge. The normalization of this term is given in terms of :math:`M_{\rm disk}` such that,

.. math::
    \Sigma_0 = \frac{M_{\rm disk} \, r_0^{\gamma} \, (2 + \gamma)}{2 \pi \, (r_{\rm max}^2 - r_{\rm min}^2)},

which means that :math:`\gamma > -2`. This means that for :math:`\gamma \leq 2` we have to calculate this numerically.

.. note::
    With this approach, the inner radius of the fit is considered :math:`r_{\rm min}` and any disk mass inside this term will result in a slightly inflated :math:`M_{\rm star}`.


Gas Volume Density
^^^^^^^^^^^^^^^^^^

To relate :math:`\Sigma_{\rm gas}` to :math:`n({\rm H_2})` we assume an isothermal vertical density profile,

.. math::
    n(r,\, z) = \frac{\Sigma_{\rm gas}(r)}{\sqrt{2 \pi} h_p(r)} \cdot \exp\left(-\frac{1}{2}\frac{z^2}{h_p(r)^2} \right),

where :math:`h_p` is the gas scale height. Unless there is some other observational constrain on :math:`h_p`, it is typically taken to be :math:`h_p \, / \, r = 0.1`. While this provides some degree of self-consistency between the models, significant changes in :math:`z(r)` can result in large deviation ins :math:`n({\rm H_2})`.

.. note::
    There are different definitions of the :math:`h_p` which can vary by a factor of :math:`\sqrt{2}`. While this shouldn't introduce a significant difference relative to the other uncertainties involved, it's good to check.


Disk Self-Gravity
^^^^^^^^^^^^^^^^^

Currently to calculate the self-gravity of the disk we take the simplification of

.. math::
    \left. \frac{\partial \phi_{\rm gas}}{\partial r} \right|_{r = r^{\prime}} = 2 \pi G \Sigma(r^{\prime}),

which is appropriate when :math:`\Sigma_{\rm gas} = \Sigma_0 \cdot (r \, / \, r_0)^{-1}`. If :math:`\gamma \sim -1` this holds, however we need to think about this for other :math:`\gamma` values.


Perturbations in :math:`n({\rm H_2})` Profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In ``dynsity`` we have two options for modelling :math:`\delta n`: either the product of multiple Gaussian perturbations,

.. math::
    \delta n = \prod_{i}^N \mathcal{G}_i,

where

.. math::
    \mathcal{G}_i (r,\, r_0, \Delta r,\, \Delta n) = 1 - \Delta n \cdot \exp\left(-\frac{(r - r_0)^2}{2\Delta r^2}\right),

or a :math:`N^{\rm th}`-order polynomial. Any number of perturbation terms can be added to the model for :math:`n({\rm H_2})`, however note that no perturbations will be added to the attached :math:`T_{\rm gas}`.

.. warning::
    Currently we have no good way to bounding the coefficients for the polynomial perturbations so these should be ignored.
