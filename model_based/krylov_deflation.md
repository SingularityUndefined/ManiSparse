# Krylov Reduced Deflation for MultiSignal

## Problem

Let

$$
A = I + \lambda_L L + \lambda_\theta \Theta .
$$

The first E-step solves

$$
A x_1 = y .
$$

The deflated modes in `deflation.tex` solve, for \(k \ge 2\),

$$
P_{k-1} A P_{k-1} z_k = P_{k-1} y,
\qquad
x_k = P_{k-1} z_k,
$$

where

$$
P_{k-1} = I - Q_{k-1} Q_{k-1}^T,
\qquad
Q_{k-1} = [q_1,\ldots,q_{k-1}],
\qquad
q_i = x_i / \|x_i\|_2 .
$$

This means each mode is solved in the Euclidean orthogonal complement of the previously generated modes.

## Current Bottleneck

The direct implementation applies \(P_{k-1} A P_{k-1}\) inside CG for every mode. Even if \(P_{k-1}\) is implemented without materializing the projection matrix, each CG iteration still needs:

- one application of \(A\);
- projection before applying \(A\);
- projection after applying \(A\).

For \(K\) modes and \(t_k\) CG iterations per mode, the dominant cost is approximately

$$
\sum_{k=2}^{K} t_k \cdot \text{cost}(A\text{-apply}).
$$

## Krylov Reduction

Construct an \(m\)-dimensional Krylov subspace

$$
\mathcal K_m(A,y)
= \mathrm{span}\{y, Ay, A^2y,\ldots,A^{m-1}y\}.
$$

Using Lanczos for symmetric \(A\), obtain an orthonormal basis \(V_m\) and a small tridiagonal matrix \(T_m\):

$$
V_m^T V_m = I,
\qquad
V_m^T A V_m = T_m,
\qquad
V_m^T y = \beta e_1,
$$

where

$$
\beta = \|y\|_2 .
$$

Then approximate every mode by

$$
x_k \approx V_m s_k .
$$

If the already generated normalized modes satisfy

$$
q_i = V_m h_i,
$$

then the original-space projection \(P_{k-1}\) becomes a small-space projection

$$
P^H_{k-1} = I - H_{k-1} H_{k-1}^T,
\qquad
H_{k-1} = [h_1,\ldots,h_{k-1}].
$$

The large projected system is replaced by the small projected system

$$
P^H_{k-1} T_m P^H_{k-1} r_k
=
P^H_{k-1} \beta e_1,
\qquad
s_k = P^H_{k-1} r_k,
\qquad
x_k = V_m s_k .
$$

## Reduced Algorithm

1. Define the matrix-vector operator

   $$
   A(v) = v + \lambda_L L(v) + \lambda_\theta \Theta(v).
   $$

2. Run \(m\)-step Lanczos on \(A\) with initial vector \(y\), producing \(V_m\), \(T_m\), and \(\beta\).

3. Solve the first reduced E-step:

   $$
   T_m s_1 = \beta e_1,
   \qquad
   x_1 = V_m s_1.
   $$

   If the full E-step \(x_1\) has already been computed, it can be used directly and projected into the reduced space:

   $$
   s_1 = V_m^T x_1.
   $$

4. Normalize:

   $$
   h_1 = s_1 / \|s_1\|_2 .
   $$

5. For \(k = 2,\ldots,K\):

   $$
   P^H_{k-1} = I - H_{k-1} H_{k-1}^T.
   $$

   Solve the small projected system:

   $$
   P^H_{k-1} T_m P^H_{k-1} r_k
   =
   P^H_{k-1} \beta e_1.
   $$

   Then set

   $$
   s_k = P^H_{k-1} r_k,
   \qquad
   x_k = V_m s_k,
   \qquad
   h_k = s_k / \|s_k\|_2.
   $$

6. Return

   $$
   X = [x_1,\ldots,x_K].
   $$

## Complexity

The direct projected CG method costs roughly

$$
O\left(\sum_{k=2}^{K} t_k \cdot \text{cost}(A\text{-apply})\right).
$$

The Krylov reduced method costs roughly

$$
O\left(m \cdot \text{cost}(A\text{-apply})\right)
+ O(Km^3)
+ O(Kmn),
$$

where:

- \(m\) is the Krylov dimension;
- \(K\) is the number of modes;
- \(n\) is the number of graph nodes;
- \(O(Km^3)\) is for small projected solves;
- \(O(Kmn)\) is for reconstructing \(x_k = V_m s_k\).

When \(m \ll n\) and \(K\) is not tiny, this can be much faster because large \(A\)-applications are paid once during Lanczos instead of repeatedly inside every deflated CG solve.

## Practical Notes

- The method is approximate unless \(m=n\).
- A larger \(m\) improves accuracy but increases memory and small-system cost.
- Since \(A\) is symmetric positive definite under the usual GEM regularization, Lanczos is the natural Krylov method.
- If \(A\) is ill-conditioned, preconditioned Lanczos or preconditioned CG in the reduced solve may be needed.
- The projection \(P^H_{k-1}\) is cheap because it lives in \(m\)-dimensional space.
- The large projection matrix \(P_{k-1}\) is never formed.
