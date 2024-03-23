# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

"""Common docstring snippets for model."""

# Used by exSpy

FIT_PARAMETERS_ARG = """optimizer : str or None, default None
            The optimization algorithm used to perform the fitting.

            * ``"lm"`` performs least-squares optimization using the
              Levenberg-Marquardt algorithm, and supports bounds
              on parameters.
            * ``"trf"`` performs least-squares optimization using the
              Trust Region Reflective algorithm, and supports
              bounds on parameters.
            * ``"dogbox"`` performs least-squares optimization using the
              dogleg algorithm with rectangular trust regions, and
              supports bounds on parameters.
            * ``"odr"`` performs the optimization using the orthogonal
              distance regression (ODR) algorithm. It does not support
              bounds on parameters. See :mod:`scipy.odr` for more details.
            * All of the available methods for :func:`scipy.optimize.minimize`
              can be used here. See the :ref:`User Guide <model.fitting>`
              documentation for more details.
            * ``"Differential Evolution"`` is a global optimization method.
              It does support bounds on parameters. See
              :func:`scipy.optimize.differential_evolution` for more
              details on available options.
            * ``"Dual Annealing"`` is a global optimization method.
              It does support bounds on parameters. See
              :func:`scipy.optimize.dual_annealing` for more
              details on available options. Requires ``scipy >= 1.2.0``.
            * ``"SHGO"`` (simplicial homology global optimization) is a global
              optimization method. It does support bounds on parameters. See
              :func:`scipy.optimize.shgo` for more details on available
              options. Requires ``scipy >= 1.2.0``.

        loss_function : {``"ls"``, ``"ML-poisson"``, ``"huber"``, callable}, default ``"ls"``
            The loss function to use for minimization. Only ``"ls"`` is available
            if ``optimizer`` is one of ``"lm"``, ``"trf"``, ``"dogbox"`` or ``"odr"``.

            * ``"ls"`` minimizes the least-squares loss function.
            * ``"ML-poisson"`` minimizes the negative log-likelihood for
              Poisson-distributed data. Also known as Poisson maximum
              likelihood estimation (MLE).
            * ``"huber"`` minimize the Huber loss function. The delta value
              of the Huber function is controlled by the ``huber_delta``
              keyword argument (the default value is 1.0).
            * callable supports passing your own minimization function.

        grad : {``"fd"``, ``"analytical"``, callable, None}, default ``"fd"``
            Whether to use information about the gradient of the loss function
            as part of the optimization. This parameter has no effect if
            ``optimizer`` is a derivative-free or global optimization method.

            * ``"fd"`` uses a finite difference scheme (if available) for numerical
              estimation of the gradient. The scheme can be further controlled
              with the ``fd_scheme`` keyword argument.
            * ``"analytical"`` uses the analytical gradient (if available) to speed
              up the optimization, since the gradient does not need to be estimated.
            * callable should be a function that returns the gradient vector.
            * None means that no gradient information is used or estimated. Not
              available if ``optimizer`` is one of ``"lm"``, ``"trf"`` or ``"dogbox"``.

        bounded : bool, default False
            If True, performs bounded parameter optimization if
            supported by ``optimizer``.
        update_plot : bool, default False
            If True, the plot is updated during the optimization
            process. It slows down the optimization, but it enables
            visualization of the optimization progress.
        print_info : bool, default False
            If True, print information about the fitting results, which
            are also stored in ``model.fit_output`` in the form of
            a :class:`scipy.optimize.OptimizeResult` object.
        return_info : bool, default True
            If True, returns the fitting results in the form of
            a :class:`scipy.optimize.OptimizeResult` object.
        fd_scheme : str {``"2-point"``, ``"3-point"``, ``"cs"``}, default ``"2-point"``
            If ``grad='fd'``, selects the finite difference scheme to use.
            See :func:`scipy.optimize.minimize` for details. Ignored if
            ``optimizer`` is one of ``"lm"``, ``"trf"`` or ``"dogbox"``.
        **kwargs : dict
            Any extra keyword argument will be passed to the chosen
            optimizer. For more information, read the docstring of the
            optimizer of your choice in :mod:`scipy.optimize`.
        """
