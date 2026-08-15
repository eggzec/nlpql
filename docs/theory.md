# Theory

## The optimization problem

All codes of the NLPQL family solve the smooth nonlinear programming
problem

$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & f(x) \\
\text{s.t.} \quad & g_j(x) = 0, && j = 1,\dots,m_e \\
                  & g_j(x) \ge 0, && j = m_e+1,\dots,m \\
                  & x_l \le x \le x_u &&
\end{aligned}
$$

It is assumed that the objective function $f$ and all constraint
functions $g_j$ are continuously differentiable on the whole
$\mathbb{R}^n$. Upper and lower bounds are handled separately, they are
never inserted as rows of the constraint matrix and they are satisfied
by every iterate.

## Sequential quadratic programming

Sequential quadratic programming proceeds from a quadratic approximation
of the Lagrangian function

$$
L(x,u) := f(x) - \sum_{j=1}^{m} u_j\, g_j(x)
$$

and a linearization of the constraints. Given an iterate
$x_k \in \mathbb{R}^n$, a multiplier estimate $v_k \in \mathbb{R}^m$ and
a positive definite approximation $C_k \in \mathbb{R}^{n \times n}$ of
the Hessian of the Lagrangian, the quadratic programming subproblem

$$
\begin{aligned}
\min_{d \in \mathbb{R}^n,\ \delta \in \mathbb{R}} \quad
  & \tfrac12 d^{T} C_k d + \nabla f(x_k)^{T} d
    + \tfrac12 \sigma_k \delta^{2} \\
\text{s.t.} \quad
  & \nabla g_j(x_k)^{T} d + (1-\delta)\, g_j(x_k) = 0,
    && j = 1,\dots,m_e \\
  & \nabla g_j(x_k)^{T} d + (1-\delta)\, g_j(x_k) \ge 0,
    && j = m_e+1,\dots,m \\
  & x_l - x_k \le d \le x_u - x_k, \quad 0 \le \delta \le 1 &&
\end{aligned}
$$

is formulated and solved. The additional variable $\delta$ prevents
inconsistent linearized constraints. As long as the linearization
possesses a feasible solution, $\delta$ is fixed at zero and the
subproblem coincides with the classical one. This is why the multiplier
vector has $m + 2n + 2$ components and why the row dimension `NMAX` has
to be greater than `N`.

Let $d_k$ be the solution and $u_k$ the corresponding multiplier. A new
iterate is obtained by

$$
\begin{pmatrix} x_{k+1} \\ v_{k+1} \end{pmatrix} :=
\begin{pmatrix} x_k \\ v_k \end{pmatrix} +
\alpha_k \begin{pmatrix} d_k \\ u_k - v_k \end{pmatrix}
$$

with a steplength $\alpha_k \in (0,1]$.

## The augmented Lagrangian merit function

Global convergence is enforced by a line search with respect to the
augmented Lagrangian merit function

$$
\psi_r(x,v) := f(x)
  - \sum_{j \in J} \Bigl( v_j g_j(x) - \tfrac12 r_j g_j(x)^2 \Bigr)
  - \tfrac12 \sum_{j \in K} \frac{v_j^2}{r_j}
$$

with the index sets

$$
J := \{1,\dots,m_e\} \cup \{\, j : m_e < j \le m,\ g_j(x) \le v_j/r_j \,\},
\qquad K := \{1,\dots,m\} \setminus J .
$$

The objective function is penalized as soon as an iterate leaves the
feasible domain. The penalty parameters $r_j$ have to be chosen so that

$$
\phi_{r_k}'(0) = \nabla \psi_{r_k}(x_k,v_k)^{T}
\begin{pmatrix} d_k \\ u_k - v_k \end{pmatrix} < 0 ,
$$

where $\phi_r(\alpha) := \psi_r\bigl((x,v)^{T} + \alpha (d, u-v)^{T}\bigr)$.
The implemented update is

$$
r_j := \max\left\{
  \frac{2m\,(u_j - v_j)^2}{(1-\delta_k)\, d_k^{T} C_k d_k},\;
  \tfrac12 \bigl(r_j + \hat r_j\bigr)
\right\},
$$

which never lets a penalty parameter fall below the value required for a
sufficient descent property, but allows a slow decrease again.

## Line search

### Serial line search

The steplength satisfies an Armijo type sufficient decrease condition

$$
\phi_r(\sigma \beta^i) \le \phi_r(0) + \sigma \beta^i \mu\, \phi_r'(0)
$$

with $0 < \mu < \tfrac12$ and $0 < \beta < 1$. A pure bisection is
inefficient, therefore a quadratic interpolation is applied first and
the Armijo condition is only used as a stopping criterion:

> **Algorithm.** Let $\beta$, $\mu$ with $0 < \beta < 1$,
> $0 < \mu < 0.5$ be given. Start with $\alpha_0 := 1$.
> For $i = 0,1,2,\dots$ do
>
> 1. If $\phi_r(\alpha_i) < \phi_r(0) + \mu\,\alpha_i\,\phi_r'(0)$, stop.
> 2. Compute
>    $\bar\alpha_i := \dfrac{0.5\,\alpha_i^{2}\,\phi_r'(0)}
>    {\alpha_i \phi_r'(0) - \phi_r(\alpha_i) + \phi_r(0)}$.
> 3. Let $\alpha_{i+1} := \max(\beta\,\alpha_i,\ \bar\alpha_i)$.

Step 3 prevents the minimizer of the quadratic interpolation from
leaving the interval $(0,1]$.

### Distributed line search

If model functions can be computed simultaneously on $l$ machines, the
$l$ test values $\alpha_i = \beta^{i-1}$, $i = 1,\dots,l$, with
$\beta = \tau^{1/(l-1)}$ are evaluated in parallel and the first one
satisfying the sufficient decrease condition is accepted. The parameter
$\tau$ is the input variable `STPMIN`. Numerical experience reported in
the user's guide shows that at least five and at most ten simultaneous
evaluations are useful.

### Non-monotone line search

If the line search cannot be terminated within `MAXFUN` steps, the
algorithm proceeds from a descent direction whose directional derivative
is extremely small, typically because of inaccurate function or gradient
values. Instead of the monotone test, a steplength is then accepted as
soon as

$$
\phi_{r_k}(\alpha_k) \le
  \max_{k-p(k) \le j \le k} \phi_{r_j}(0)
  + \alpha_k \mu\, \phi_{r_k}'(0)
$$

holds, where $p(k) = \min\{k,p\}$ and $p$ is the queue length `MAXNM`.
A monotone line search is applied as long as it terminates successfully,
the non-monotone one is only used in this special error situation.

## Quasi-Newton update

The matrix $C_k$ is updated by the BFGS formula

$$
C_{k+1} := C_k + \frac{q_k q_k^{T}}{p_k^{T} q_k}
  - \frac{C_k p_k p_k^{T} C_k}{p_k^{T} C_k p_k}
$$

with $p_k := x_{k+1} - x_k$ and
$q_k := \nabla_x L(x_{k+1},u_k) - \nabla_x L(x_k,u_k)$. The
modification of Powell guarantees that all matrices stay positive
definite: whenever $p^{T} q < 0.2\, p^{T} C p$, the vector $q$ is
replaced by $\theta q + (1-\theta) C p$ with
$\theta := 0.8\, p^{T} C p / (p^{T} C p - p^{T} q)$.

Scaling and restarts are controlled by `MODE`. The scaling factor of the
Oren-Luenberger procedure is $\gamma_k = p_k^{T} q_k / p_k^{T} p_k$, and
$C_k$ may be replaced by $\gamma_k I$ initially, adaptively or
periodically. In error situations, for instance an uphill search
direction caused by inaccurate derivatives, the matrix is reset to
$\rho I$ with the input parameter `RHO`.

## Termination

The Karush-Kuhn-Tucker criterion reported by the codes is

$$
\mathrm{KKT} = \bigl| \nabla f(x)^{T} d \bigr|
  + \sum_{j=1}^{m} \bigl| u_j\, g_j(x) \bigr|
  + \sum_{i=1}^{n} \bigl| u_i^{l} (x_i - x_{l,i}) \bigr|
  + \sum_{i=1}^{n} \bigl| u_i^{u} (x_{u,i} - x_i) \bigr| ,
$$

i.e. the stationarity of the quadratic subproblem plus the
complementarity of all constraints and bounds. Together with the sum of
constraint violations

$$
\mathrm{SCV} = \sum_{j=1}^{m_e} |g_j(x)|
  + \sum_{j=m_e+1}^{m} \max\{0, -g_j(x)\}
$$

the iteration is stopped as soon as $\mathrm{KKT} < \varepsilon$ and
$\mathrm{SCV} < \sqrt{\varepsilon}$, where $\varepsilon$ is the input
parameter `ACC`. In addition the algorithm stops when the search
direction vanishes, and when the predicted decrease of the merit
function drops below the accuracy by which the merit function itself can
be evaluated, since no further progress is possible in that situation.

## The quadratic programming subproblem

The subproblem is solved by `QL`, an implementation of the primal-dual
method of Goldfarb and Idnani. Starting from the unconstrained minimum
$x = -C^{-1} d$, violated constraints are successively added to an
active set. In every step the minimizer subject to the current active
set is computed, which keeps the objective function values strictly
increasing, so that the method terminates after finitely many steps. All
matrix manipulations are performed by orthogonal Givens rotations
applied to $J = R^{-1}$, where $C = R^{T} R$ is the Cholesky
factorization, and to the triangular factor of $J^{T} N$ with $N$ the
matrix of the active constraint normals. A dual method needs no phase I,
i.e. no feasible starting point has to be computed.

## Very many constraints

`NLPQLB` extends the method to problems where $m$ is very large compared
to $n$ and where the Jacobian possesses no exploitable sparsity. The
user provides a bound $m_w$ with $n \le m_w \le m$ for the number of
expected active constraints. Quadratic subproblems are generated with
$m_w$ linear constraints only, the working set

$$
W_k := J_k^{\star} \cup \overline{K}_k^{\star}, \qquad |W_k| = m_w ,
$$

which always contains the set of active constraints

$$
J_k^{\star} := \{1,\dots,m_e\} \cup
  \{\, j : m_e < j \le m,\ g_j(x_k) < \varepsilon
  \ \text{or}\ v_j^{(k)} > 0 \,\} .
$$

New gradients are only required for the constraints of the working set.
Since every constraint outside the working set satisfies
$g_j(x) > \varepsilon$, the convergence conditions of the reduced
problem are applicable for the original problem as well. The line search
additionally guarantees that no intermediate iterate violates more than
$m_w$ constraints, the steplength is reduced until this condition holds.

## Successive restarts

`NLPQLG` executes `NLPQLP` for several starting points and returns the
best local solution. The starting points of the cycles
$k = 2,3,\dots$ are generated by the deterministic Kronecker sequence

$$
t_i^{(k)} = \operatorname{frac}\bigl( (k-1)\sqrt{p_i} \bigr), \qquad
x_i = x_{l,i} + t_i^{(k)} (x_{u,i} - x_{l,i}) ,
$$

with $p_i$ the $i$-th odd number. No random number generator is
involved, so that all runs are exactly reproducible.

## Data fitting and min-max problems

Objective functions built from $L$ individual functions $f_1,\dots,f_L$
are either non-differentiable or possess a structure that should not be
handed to a general purpose solver directly. They are therefore
transformed into smooth nonlinear programs by additional variables and
constraints.

For a sum of squares, $L$ additional variables $z$ and $L$ additional
equality constraints are introduced,

$$
\min_{x,z} \ \sum_{i=1}^{L} z_i^{2} \quad\text{s.t.}\quad
f_i(x) - z_i = 0 ,
$$

which retains the typical features of a Gauss-Newton type method. For a
sum of absolute values, $L$ variables and $2L$ inequalities are needed,

$$
\min_{x,z} \ \sum_{i=1}^{L} z_i \quad\text{s.t.}\quad
-f_i(x) + z_i \ge 0 ,\quad f_i(x) + z_i \ge 0 ,\quad z \ge 0 .
$$

For the maximum norm one single variable and $2L$ inequalities suffice,

$$
\min_{x,z} \ z \quad\text{s.t.}\quad
-f_i(x) + z \ge 0 ,\quad f_i(x) + z \ge 0 ,\quad z \ge 0 ,
$$

and for a min-max problem one variable and $L$ inequalities,

$$
\min_{x,z} \ z \quad\text{s.t.}\quad -f_i(x) + z \ge 0 .
$$

If the number of terms is very large, the additional variables of the
least squares transformation become prohibitive. `NLPLSX` therefore
minimizes the sum of squares directly and assembles the gradient from
the individual terms,
$\nabla f = 2\sum_i f_i(x)\,\nabla f_i(x)$.

## Multicriteria problems

A vector of criteria $(f_1,\dots,f_L)$ is reduced to a scalar objective
by one of sixteen transformations. All of them can be written as a
residual vector

$$
r_i(x) := a_i \bigl( f_i(x) - b_i \bigr) , \qquad i = 1,\dots,L ,
$$

combined by a plain sum, by the $L_1$, the $L_2$ or the maximum norm,
plus at most $L-1$ additional inequality constraints for the
hierarchical and the trade-off method. The coefficients $a_i$ and $b_i$
follow from the weights, from the goal values and from the individual
minima $f_i^{\star}$, which are obtained beforehand by minimizing each
criterion separately. Once the residuals are formed, the very same
transformations as for data fitting apply.

## Functions defined on a subset only

If the objective function and some of the constraints can be evaluated
only on the convex set

$$
F := \{\, x \in \mathbb{R}^n : e_i(x) \ge 0 ,\ i = 1,\dots,m \,\}
$$

described by concave feasibility constraints $e_i$, every argument at
which they are requested has to belong to $F$. Proceeding from a
feasible starting point, the feasibility constraints are evaluated first
at every test point of the line search. They are assumed to be much
cheaper than the objective. If one of them is violated, the steplength
is halved and a new test point is generated. Since $F$ is convex and
contains the actual iterate, the reduction terminates, and the objective
is never evaluated outside of $F$.
