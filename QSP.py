import numpy as np
from numpy.polynomial import Polynomial


def compute_angles(P): 
    r"""
    This is adapted from the pennylane implementation. -- Add citation here
    Computes the Quantum Signal Processing (QSP) angles given the coefficients of a polynomial P.
    The coefficients of P must be given in the Chebyshev basis!

    The method for computing the QSP angles is adapted from the approach described in [`arXiv:2406.04246 <https://arxiv.org/abs/2406.04246>`_] for Generalized-QSP.

    Args:
        P (tensor-like): Chebyshev basis coefficients of the polynomial P with degree d
        N (tensor-like): size of the FFT, N >= (d+1). For example N = 40*d

    Returns:
        (tensor-like): QSP angles corresponding to the input polynomial P

    .. details::
        :title: Implementation details

        Based on the appendix A in `arXiv:2406.04246 <https://arxiv.org/abs/2406.04246>`_, the target polynomial :math:`P`
        is transformed into a new polynomial :math:`P'` by following the steps below:

        0. The input to the function are the coefficients of :math:`P` in the Chebyshev basis, e.g. :math:`[a_0, 0, a_1, 0, a_2]`.
        2. We generate :math:`P'` by reordering the array, moving the zeros to the initial positions.
           :math:`P' = [0, 0, a_0, a_1, a_2]`.

        The polynomial :math:`P` can now be used in Algorithm 1 of [`arXiv:2308.01501 <https://arxiv.org/abs/2308.01501>`_]
        in order to find the desired angles.

        The above algorithm is specific to Generalized-QSP so an adaptation has been made to return the required angles:

            - The :math:`R(\theta, \phi, \lambda)` gate, is simplified into a :math:`R_Y(\theta)` gate.

            - The calculation of :math:`\theta_d` is updated to :math:`\theta_d = \tan^{-1}(\frac{a_d}{b_d})`.
              In this way, the sign introduced by :math:`\phi_d` and :math:`\lambda_d` is absorbed
              in the :math:`\theta_d` value.
    """
    parity = (len(P)-1) % 2
    P = np.concatenate([P[-parity::-2], P[parity::2]])/2
    Q = _complementary_poly(P)
    polynomial_matrix = np.array([P, Q])
    num_terms = polynomial_matrix.shape[1]
    rotation_angles = np.zeros(num_terms)

    # Adaptation of Algorithm 1 of [arXiv:2308.01501]
    for idx in range(num_terms - 1, -1, -1):

        poly_a, poly_b = polynomial_matrix[:, idx]
        rotation_angles[idx] = np.arctan2(poly_b.real, poly_a.real)

        #rotation_op = ops.RY.compute_matrix(-2 * rotation_angles[idx])

        phi = -1 * rotation_angles[idx]
        rotation_op = [[np.cos(phi), -1 * np.sin(phi)], [np.sin(phi), np.cos(phi)]]

        updated_poly_matrix = rotation_op @ polynomial_matrix
        polynomial_matrix = np.array(
            [updated_poly_matrix[0][1 : idx + 1], updated_poly_matrix[1][0:idx]]
        )

    return rotation_angles

def _complementary_poly(poly_coeffs):
    r"""
    Computes the complementary polynomial Q given a polynomial P.

    The polynomial Q is complementary to P if it satisfies the following equation:

    .. math:

        |P(e^{i\theta})|^2 + |Q(e^{i\theta})|^2 = 1, \quad \forall \quad \theta \in \left[0, 2\pi\right]

    The method is based on computing an auxiliary polynomial R, finding its roots,
    and reconstructing Q by using information extracted from the roots.
    For more details see `arXiv:2308.01501 <https://arxiv.org/abs/2308.01501>`_.

    Args:
        poly_coeffs (tensor-like): coefficients of the complex polynomial P

    Returns:
        tensor-like: coefficients of the complementary polynomial Q
    """
    poly_degree = len(poly_coeffs) - 1

    # Build the polynomial R(z) = z^degree * (1 - conj(P(1/z)) * P(z)), deduced from (eq.33) and (eq.34) of
    # `arXiv:2308.01501 <https://arxiv.org/abs/2308.01501>`_.
    # Note that conj(P(1/z)) * P(z) could be expressed as z^-degree * conj(P(z)[::-1]) * P(z)
    R = Polynomial.basis(poly_degree) - Polynomial(poly_coeffs) * Polynomial(
        np.conj(poly_coeffs[::-1])
    )
   
    r_roots = R.roots()

    inside_circle = [root for root in r_roots if np.abs(root) <= 1]
    outside_circle = [root for root in r_roots if np.abs(root) > 1]

    scale_factor = np.sqrt(np.abs(R.coef[-1] * np.prod(outside_circle)))

    Q_poly = scale_factor * Polynomial.fromroots(inside_circle)

    return Q_poly.coef