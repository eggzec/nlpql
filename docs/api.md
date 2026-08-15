# API Reference

## `nlpql.minimize`

```python
minimize(
    fun,
    x0,
    args=(),
    method="NLPQLP",
    jac=None,
    bounds=None,
    constraints=(),
    tol=None,
    callback=None,
    options=None,
)
```

Minimize a smooth function subject to nonlinear equality and inequality
constraints and bounds on the variables.

| argument      | meaning                                                    |
| ------------- | ---------------------------------------------------------- |
| `fun`         | objective `f(x, *args)` returning a scalar                  |
| `x0`          | starting point                                              |
| `args`        | extra arguments of `fun`, `jac` and the constraints         |
| `method`      | `NLPQLP`, `NLPQL`, `NLPQLY`, `NLPQLB`, `NLPQLG`, `NLPQLF`   |
| `jac`         | gradient of `fun`, forward differences if `None`            |
| `bounds`      | sequence of `(lower, upper)` pairs, or an object `lb`/`ub`  |
| `constraints` | dictionary or sequence of dictionaries, see below           |
| `tol`         | desired accuracy, overrides `options["acc"]`                |
| `callback`    | called with the current iterate after every iteration       |
| `options`     | solver options, see [Quickstart](quickstart.md#options)     |

### Result

`minimize` returns an `OptimizeResult`, a dictionary whose entries are
also accessible as attributes.

| field         | meaning                                                    |
| ------------- | ---------------------------------------------------------- |
| `x`           | final iterate                                               |
| `fun`         | objective function value at `x`                             |
| `jac`         | gradient of the objective at `x`                            |
| `constr`      | constraint values at `x`, equalities first                  |
| `multipliers` | multipliers of the constraints, then of the bounds          |
| `active`      | boolean mask of the constraints considered active           |
| `working_set` | indices of the working set, `NLPQLB` only                   |
| `residuals`   | individual function values, `fit` only                      |
| `objectives`  | values of all criteria, `minimize_multi` only               |
| `feasibility` | feasibility constraint values, `NLPQLF` only                |
| `status`      | termination flag `IFAIL` of the Fortran routine             |
| `success`     | `True` if `status == 0`                                     |
| `message`     | description of the termination reason                       |
| `nit`         | number of iterations                                        |
| `nfev`        | number of objective and constraint evaluations              |
| `njev`        | number of gradient evaluations                              |
| `nqp`         | number of quadratic programs solved                         |
| `method`      | name of the code that has been used                         |

The multiplier vector holds the `m` multipliers of the constraints
first, then `n` multipliers of the lower and `n` multipliers of the
upper bounds. At an optimal solution all multipliers belonging to
inequality constraints are non-negative.

### Termination flags

| `status` | meaning                                                        |
| -------: | -------------------------------------------------------------- |
|      `0` | optimality conditions satisfied                                 |
|      `1` | maximum number of iterations exceeded                           |
|      `2` | uphill search direction                                         |
|      `3` | underflow when computing the new BFGS update matrix             |
|      `4` | line search exceeded the maximum number of function calls       |
|      `5` | length of a working array too short                             |
|      `6` | false dimensions                                                |
|      `7` | search direction close to zero at an infeasible iterate         |
|      `8` | starting point violates a lower or upper bound                  |
|      `9` | wrong input parameter                                           |
|     `10` | inconsistency in the quadratic programming subproblem           |
|     `11` | too many active constraints, or non-evaluable function calls    |
|  `> 100` | error `status - 100` of the quadratic programming solver        |

## `nlpql.fit`

```python
fit(
    fun,
    x0,
    args=(),
    method="NLPLSQ",
    jac=None,
    bounds=None,
    constraints=(),
    tol=None,
    callback=None,
    options=None,
)
```

Solve a constrained data fitting problem, where `fun` returns the vector
of individual functions `f_1(x), ..., f_L(x)`.  The norm by which they
are combined follows from `method`:

| method   | objective                       |
| -------- | ------------------------------- |
| `NLPLSQ` | `sum f_i(x)^2`                  |
| `NLPLSX` | `sum f_i(x)^2`, very many terms |
| `NLPL1`  | `sum abs(f_i(x))`               |
| `NLPINF` | `max abs(f_i(x))`               |
| `NLPMMX` | `max f_i(x)`                    |

The result carries the additional field `residuals`.

## `nlpql.minimize_multi`

```python
minimize_multi(
    fun,
    x0,
    args=(),
    model=1,
    imin=1,
    weights=None,
    fk=None,
    jac=None,
    bounds=None,
    constraints=(),
    tol=None,
    callback=None,
    options=None,
)
```

Solve a multicriteria problem with `NLPJOB`, where `fun` returns the
vector of criteria.  `model` selects one of the sixteen scalar
transformations listed in `nlpql.JOB_MODELS`, `imin` the criterion used
by the models 0, 2 and 3, `weights` the weights, bounds or goal values
and `fk` the individual minima or goal values.  The result carries the
additional field `objectives`.

## `nlpql.solve_qp`

```python
solve_qp(c, d, a=None, b=None, me=0, xl=None, xu=None, eps=1.0e-12)
```

Solve the strictly convex quadratic program

```text
min  1/2 x^T C x + d^T x
     a_j^T x + b_j  = 0 ,  j = 1, ..., me
     a_j^T x + b_j >= 0 ,  j = me+1, ..., m
     xl <= x <= xu
```

by the primal-dual method of Goldfarb and Idnani. Returns an
`OptimizeResult` with the fields `x`, `fun`, `multipliers`, `status`,
`success` and `message`.

## Workspace helpers

The Fortran routines never allocate memory, they check the length of
every working array instead. The helpers below reproduce the
partitioning of the sources, which is useful when the routines are
driven directly.

```python
nlpql.nlpqlp_sizes(n, m, nproc=1)  # NLPQLP
nlpql.nlpql_sizes(n, m)  # NLPQL
nlpql.nlpqly_sizes(n, m)  # NLPQLY
nlpql.nlpqlb_sizes(n, m, mw)  # NLPQLB
nlpql.nlpqlg_sizes(n, m, nproc=1)  # NLPQLG
nlpql.nlpqlf_sizes(n, mf, m)  # NLPQLF
nlpql.fit_sizes(method, n, m, ell)  # data fitting codes
nlpql.nlpjob_sizes(n, m, ell)  # NLPJOB
```

Each returns a dictionary with the required array dimensions, for
example `nmax`, `mmax`, `mnn2`, `lwa`, `lkwa` and `lact`.

## Fortran routines

`nlpql.raw` exposes the compiled routines directly. They use the reverse
communication protocol of the original codes.

| routine    | description                                                 |
| ---------- | ----------------------------------------------------------- |
| `nlpqlpq`  | `NLPQLP` with the QP solver `QL` already bound               |
| `nlpql`    | the original `NLPQL`                                         |
| `nlpqly`   | the easy-to-use version `NLPQLY`                             |
| `nlpqlbq`  | `NLPQLB` with the QP solver `QL` already bound               |
| `nlpqlgq`  | `NLPQLG` with the QP solver `QL` already bound               |
| `nlpqlfq`  | `NLPQLF` with the QP solver `QL` already bound               |
| `nlplsqq`  | `NLPLSQ`, constrained nonlinear least squares                 |
| `nlplsxq`  | `NLPLSX`, least squares with very many terms                 |
| `nlpl1q`   | `NLPL1`, sum of absolute values                              |
| `nlpinfq`  | `NLPINF`, maximum norm data fitting                          |
| `nlpmmxq`  | `NLPMMX`, min-max optimization                               |
| `nlpjob`   | `NLPJOB`, multicriteria optimization                         |
| `ql`       | the convex quadratic programming code `QL`                   |

The complete calling sequences of the underlying Fortran subroutines,
including the variants that accept an external QP solver through the
argument `QPSLVE`, are documented in the headers of the sources in
`src/`:

```text
src/ql.f        QL      convex quadratic programming
src/nlpqlp.f    NLPQLP  distributed and non-monotone line search
src/nlpql.f     NLPQL   the original SQP code
src/nlpqly.f    NLPQLY  easy-to-use version
src/nlpqlb.f    NLPQLB  active set strategy for many constraints
src/nlpqlg.f    NLPQLG  successive restarts
src/nlpqlf.f    NLPQLF  feasible SQP method
src/nlpjob.f    NLPJOB  multicriteria optimization
src/nlplsq.f    NLPLSQ  constrained nonlinear least squares
src/nlplsx.f    NLPLSX  least squares with very many terms
src/nlpl1.f     NLPL1   sum of absolute values
src/nlpinf.f    NLPINF  maximum norm data fitting
src/nlpmmx.f    NLPMMX  min-max optimization
src/nldftw.f    NLDFTW  common transformation of the fitting codes
```

Every file holds exactly one program unit whose name is the name
of the file, so that any routine can be located directly.

### Reverse communication protocol

1. Choose starting values and store them in `X`.
2. Compute the objective and all constraint values, store them in `F`
   and `G`.
3. Compute the gradients and store them in `DF` and `DG`.
4. Set `IFAIL = 0` and call the routine.
5. If the routine returns with `IFAIL = -1`, compute new function
   values at `X` and call it again.
6. If it returns with `IFAIL = -2`, compute new gradient values at `X`
   and call it again. Only derivatives of the constraints marked in
   `ACT` are needed.
7. `IFAIL = 0` indicates a successful return, `IFAIL > 0` an error.

`NLPQLF` uses one additional request, `IFAIL = -3`, which asks for
the values of the feasibility constraints only.  They are evaluated
before the objective function, so that the latter is never called
outside of the set on which it is defined.

If the model functions cannot be evaluated at a trial point, call
`NLPQLP` with `IFAIL = -10`. The steplength is then reduced by the
factor `0.5` and the routine returns immediately with `IFAIL = -1`.
