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

We assume that the user has been able to measure:

* :math:`v_{\phi}(r)` - The deprojected rotational velocity of the gas as a function of radius in :math:`[{\rm m\,s^{-1}}]`.

* :math:`z(r)` - The height of the emission surface as a function of radius in :math:`[{\rm au}]`.

* :math:`T_{\rm gas}(r)` - The gas temperature as a function of radius in :math:`[{\rm K}]`.

The emission surface can be derived either from fitting the rotation map, as in [citation], or following the method in `Pinte et al. (2018) <www.google.com>`_. The gas temperature is harder to measure, but optically thick lines like CO are useful as :math:`T_{\rm B} = T_{\rm ex}`.

Fitting :math:`M_{\rm star}` Only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the most simple case where only :math:`v_{\phi}(r)` and :math:`z(r)` are needed. With the inclusion of a non-zero :math:`z(r)` note that :math:`M_{\rm star}` will be substantially larger than for an assumed 2D disk.

Inferring a :math:`n({\rm H_2})` Profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we make the assumption that the gas volume density *at the emission surface* is well described by,

.. math::
    n({\rm H_2}) = n_0 \times \left( \frac{r}{100~{\rm au}}\right)^{\gamma} \times \delta n

where :math:`\delta n` is some perturbation term and includes all local deviations. In ``dynsity`` we have two options for modelling :math:`\delta n`: either the product of multiple Gaussian perturbations,

.. math::
    \delta n = \prod_{i}^N \mathcal{G}_i,

where

.. math::
    \mathcal{G}_i (r,\, r_0, \Delta r,\, \Delta n) = 1 - \Delta n \cdot \exp\left(-\frac{(r - r_0)^2}{2\Delta r^2}\right),

or a :math:`N^{\rm th}`-order polynomial. Any number of perturbation terms can be added to the model for :math:`n({\rm H_2})`, however note that no perturbations will be added to the attached :math:`T_{\rm gas}`.

Including :math:`M_{\rm disk}` Corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To make the connection betweem :math:`n({\rm H_2})` and :math:`\Sigma_{\rm gas}` needed to calculate :math:`\phi_{\rm gas}`, we make the assumption that the gas surface densitiy follows the profile of

.. math::
    \Sigma_{\rm gas}(r) = \Sigma_0 \times \left( \frac{r}{100~{\rm au}} \right),

so the same as :math:`n({\rm H_2})`, but without the perturbations.


In practice, this is somewhat more complicated as certain disk configurations may result in a radially *increasing* :math:`n({\rm H_2})` due to the emission surface dropping deeper into the disk (i.e. having :math:`\gamma > 0`). This needs some more thought.
