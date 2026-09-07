# QOCO.jl

Julia wrapper for the [QOCO](https://github.com/qoco-org/qoco) solver — a
Quadratic Objective Conic Optimizer.

QOCO solves second-order cone programs (SOCPs) with quadratic objectives of the
form:

```text
min   (1/2)x'Px + c'x
s.t.  Ax = b
      Gx <=_C h
```

where `C = R_+^l × SOC(q₁) × … × SOC(qₙ)` is a product of the nonnegative
orthant and second-order cones.

## Installation

`QOCO.jl` is not currently registered in the Julia General registry, so install
it directly from GitHub:

```julia
import Pkg
Pkg.add(url = "https://github.com/qoco-org/QOCO.jl.git")
```

This also installs the bundled `QOCO_jll` binaries, so you do not need to
install QOCO separately.

## Usage with JuMP

```julia
using JuMP, QOCO

model = Model(QOCO.Optimizer)
set_silent(model)

@variable(model, x[1:2] >= 0)
@constraint(model, x[1] + x[2] == 1)
@objective(model, Min, x[1]^2 + x[2]^2 - x[1] - x[2])

optimize!(model)

value.(x)  # [0.5, 0.5]
```

## Usage with MathOptInterface

```julia
import MathOptInterface as MOI

optimizer = QOCO.Optimizer()
MOI.set(optimizer, MOI.Silent(), true)
```

## MathOptInterface API

The QOCO optimizer supports the following native MathOptInterface objective
functions and constraint types.

List of supported objective functions:

 * `MOI.ObjectiveFunction{MOI.VariableIndex}`
 * `MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}`
 * `MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}`

List of supported constraint types:

 * `MOI.VectorAffineFunction{Float64}` in `MOI.Zeros`
 * `MOI.VectorAffineFunction{Float64}` in `MOI.Nonnegatives`
 * `MOI.VectorAffineFunction{Float64}` in `MOI.SecondOrderCone`

JuMP and `MOI.Bridges.full_bridge_optimizer` can bridge many scalar linear
constraints and bound constraints into these native vector-affine forms.

## Limitations

QOCO is exposed as a one-shot, `copy_to`-based MOI optimizer. Incremental model
construction is handled by JuMP or an MOI caching optimizer rather than by the
native solver object itself.

The native interface supports quadratic objectives, but not quadratic
constraints.

QOCO does not currently provide reliable infeasibility or unboundedness
certificates through the wrapper, so such models may terminate with iteration or
numerical-error statuses instead of `MOI.INFEASIBLE` or
`MOI.DUAL_INFEASIBLE`.

QOCO does not support integer or mixed-integer optimization.

## Solver Settings

Settings can be passed as keyword arguments to the optimizer or set via
`MOI.RawOptimizerAttribute`:

```julia
# Via constructor
optimizer = QOCO.Optimizer(; max_iters = 500, abstol = 1e-8)

# Via MOI
MOI.set(optimizer, MOI.RawOptimizerAttribute("max_iters"), 500)
```

| Parameter           | Type    | Default  | Description                                   |
|---------------------|---------|----------|-----------------------------------------------|
| `max_iters`         | Int     | 200      | Maximum number of iterations                  |
| `ruiz_iters`        | Int     | 0        | Ruiz equilibration iterations                 |
| `max_ir_iters`      | Int     | 5        | Maximum iterative refinement iterations       |
| `ir_tol`            | Float64 | 1e-6     | Iterative refinement stopping tolerance       |
| `kkt_static_reg_P`  | Float64 | 1e-8 †   | KKT static regularization, (1,1) `P` block    |
| `kkt_static_reg_A`  | Float64 | 1e-8     | KKT static regularization, (2,2) `A` block    |
| `kkt_static_reg_G`  | Float64 | 1e-13    | KKT static regularization, (3,3) `G` block    |
| `kkt_dynamic_reg`   | Float64 | 1e-11    | KKT dynamic regularization                    |
| `abstol`            | Float64 | 1e-7     | Absolute tolerance                            |
| `reltol`            | Float64 | 1e-7     | Relative tolerance                            |
| `abstol_inacc`      | Float64 | 1e-5     | Absolute tolerance (inaccurate)               |
| `reltol_inacc`      | Float64 | 1e-5     | Relative tolerance (inaccurate)               |
| `verbose`           | Bool    | false    | Print solver output                           |

† `kkt_static_reg_P` is the one setting where QOCO.jl does not use QOCO's own
default, which is `1e-13`. When the objective has no quadratic term, the (1,1)
block of the KKT system is exactly this regularization and nothing else, so on
problems whose constraints are also rank deficient the KKT matrix becomes
numerically singular: the primal residual stops improving while the duality gap
keeps shrinking, and the solve returns `MOI.NUMERICAL_ERROR` even though the
objective is correct. QOCO.jl restores the pre-0.3.0 value of `1e-8`, which
repairs those cases and leaves well-conditioned problems unchanged. Set the
attribute explicitly to override this, or read `QOCO.c_default_settings()` for
QOCO's unmodified values.

`MOI.Silent()` takes precedence over the raw `verbose` setting. By default, the
wrapper uses QOCO's compiled default (`verbose = false`). To enable solver
output, set `MOI.RawOptimizerAttribute("verbose")` to `true`, or pass
`verbose = true` to `QOCO.Optimizer`.
