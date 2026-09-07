using QOCO
using Test
import MathOptInterface as MOI
using JuMP

@testset "MOI Wrapper" begin
    @testset "Solver attributes" begin
        opt = QOCO.Optimizer()
        @test MOI.get(opt, MOI.SolverName()) == "QOCO"
        @test MOI.get(opt, MOI.SolverVersion()) == "0.3.2"
        @test MOI.is_empty(opt)
    end

    @testset "Settings" begin
        opt = QOCO.Optimizer(; max_iters = 100, abstol = 1e-6)
        @test MOI.get(opt, MOI.RawOptimizerAttribute("max_iters")) == 100
        @test MOI.get(opt, MOI.RawOptimizerAttribute("abstol")) == 1e-6
        MOI.set(opt, MOI.RawOptimizerAttribute("reltol"), 1e-8)
        @test MOI.get(opt, MOI.RawOptimizerAttribute("reltol")) == 1e-8

        # Settings introduced in QOCO 0.3.
        for name in (
            "max_ir_iters", "ir_tol",
            "kkt_static_reg_P", "kkt_static_reg_A", "kkt_static_reg_G",
        )
            @test MOI.supports(opt, MOI.RawOptimizerAttribute(name))
        end
        @test MOI.get(opt, MOI.RawOptimizerAttribute("max_ir_iters")) == 5

        # The wrapper's overridden default is what gets reported, and an
        # explicit value still takes precedence over it.
        @test MOI.get(opt, MOI.RawOptimizerAttribute("kkt_static_reg_P")) ==
              QOCO.SETTING_OVERRIDES.kkt_static_reg_P
        MOI.set(opt, MOI.RawOptimizerAttribute("kkt_static_reg_P"), 1e-12)
        @test MOI.get(opt, MOI.RawOptimizerAttribute("kkt_static_reg_P")) == 1e-12

        MOI.set(opt, MOI.RawOptimizerAttribute("ir_tol"), 1e-9)
        @test MOI.get(opt, MOI.RawOptimizerAttribute("ir_tol")) == 1e-9
        MOI.set(opt, MOI.RawOptimizerAttribute("kkt_static_reg_A"), 1e-7)
        @test MOI.get(opt, MOI.RawOptimizerAttribute("kkt_static_reg_A")) == 1e-7

        # Settings that QOCO 0.1.6 had but 0.3 removed.
        for name in ("bisect_iters", "iter_ref_iters", "kkt_static_reg")
            @test !MOI.supports(opt, MOI.RawOptimizerAttribute(name))
        end
    end

    @testset "Silent" begin
        opt = QOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)
        @test MOI.get(opt, MOI.Silent()) == true
    end

    @testset "Query surfaces" begin
        opt = QOCO.Optimizer()
        model = MOI.Utilities.Model{Float64}()
        MOI.set(model, MOI.Name(), "query-model")
        x = MOI.add_variables(model, 2)
        MOI.set(model, MOI.VariableName(), x[1], "x1")
        MOI.set(model, MOI.VariableName(), x[2], "x2")

        obj = MOI.ScalarAffineFunction(
            [
                MOI.ScalarAffineTerm(1.0, x[1]),
                MOI.ScalarAffineTerm(2.0, x[2]),
            ],
            3.0,
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [-1.0],
        )
        ci = MOI.add_constraint(model, eq_func, MOI.Zeros(1))
        MOI.set(model, MOI.ConstraintName(), ci, "eq")

        idxmap = MOI.copy_to(opt, model)
        mapped_ci = idxmap[ci]

        @test MOI.get(opt, MOI.Name()) == "query-model"
        @test MOI.Name() in MOI.get(opt, MOI.ListOfModelAttributesSet())
        @test MOI.ObjectiveSense() in MOI.get(opt, MOI.ListOfModelAttributesSet())
        @test MOI.ObjectiveFunction{typeof(obj)}() in MOI.get(opt, MOI.ListOfModelAttributesSet())
        @test MOI.get(opt, MOI.ObjectiveSense()) == MOI.MIN_SENSE
        @test MOI.get(opt, MOI.ObjectiveFunctionType()) == typeof(obj)
        @test isapprox(
            MOI.get(opt, MOI.ObjectiveFunction{typeof(obj)}()),
            MOI.Utilities.canonical(obj),
        )
        @test MOI.get(opt, MOI.VariableName(), idxmap[x[1]]) == "x1"
        @test MOI.get(opt, MOI.VariableName(), idxmap[x[2]]) == "x2"
        @test MOI.get(opt, MOI.ConstraintName(), mapped_ci) == "eq"
        @test MOI.get(opt, MOI.ConstraintSet(), mapped_ci) == MOI.Zeros(1)
        @test isapprox(
            MOI.get(opt, MOI.ConstraintFunction(), mapped_ci),
            MOI.Utilities.canonical(eq_func),
        )
        @test MOI.get(opt, MOI.VariableIndex, "x1") == idxmap[x[1]]
        @test MOI.get(opt, typeof(mapped_ci), "eq") == mapped_ci
        @test MOI.get(opt, MOI.ConstraintIndex, "eq") == mapped_ci
        @test MOI.get(opt, MOI.NumberOfVariables()) == 2
        @test MOI.get(opt, MOI.ListOfVariableIndices()) == [idxmap[x[1]], idxmap[x[2]]]
        @test MOI.get(opt, MOI.NumberOfConstraints{typeof(eq_func), MOI.Zeros}()) == 1
        @test MOI.get(opt, MOI.ListOfConstraintIndices{typeof(eq_func), MOI.Zeros}()) == [mapped_ci]
    end

    @testset "Unsupported direct constraints throw" begin
        opt = QOCO.Optimizer()
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variable(model)
        func = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0)
        MOI.add_constraint(model, func, MOI.EqualTo(1.0))
        err = MOI.UnsupportedConstraint{
            MOI.ScalarAffineFunction{Float64},
            MOI.EqualTo{Float64},
        }()
        @test_throws err MOI.copy_to(opt, model)
    end

    @testset "Simple QP via MOI" begin
        # min (1/2)(2x1^2 + 2x2^2) - x1 - x2
        # s.t. x1 + x2 = 1
        #      x1, x2 >= 0
        # Solution: x = [0.5, 0.5], obj = -0.5

        opt = QOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)

        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 2)

        # Objective: (1/2)(2x1^2 + 2x2^2) - x1 - x2
        quad_terms = [
            MOI.ScalarQuadraticTerm(2.0, x[1], x[1]),
            MOI.ScalarQuadraticTerm(2.0, x[2], x[2]),
        ]
        aff_terms = [
            MOI.ScalarAffineTerm(-1.0, x[1]),
            MOI.ScalarAffineTerm(-1.0, x[2]),
        ]
        obj = MOI.ScalarQuadraticFunction(quad_terms, aff_terms, 0.0)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        # x1 + x2 = 1
        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [-1.0],
        )
        MOI.add_constraint(model, eq_func, MOI.Zeros(1))

        # x1, x2 >= 0
        nn_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [0.0, 0.0],
        )
        MOI.add_constraint(model, nn_func, MOI.Nonnegatives(2))

        idxmap = MOI.copy_to(opt, model)
        MOI.optimize!(opt)

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        @test MOI.get(opt, MOI.ResultCount()) == 1

        x1_val = MOI.get(opt, MOI.VariablePrimal(), idxmap[x[1]])
        x2_val = MOI.get(opt, MOI.VariablePrimal(), idxmap[x[2]])
        @test x1_val ≈ 0.5 atol = 1e-5
        @test x2_val ≈ 0.5 atol = 1e-5
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ -0.5 atol = 1e-5

        # Statistics that QOCO 0.3 added.
        @test MOI.get(opt, QOCO.IterativeRefinementIterations()) >= 0
        @test MOI.get(opt, QOCO.AnalysisTimeSec()) >= 0.0
    end

    @testset "Warm start via VariablePrimalStart" begin
        # Same QP as "Simple QP via MOI", solved from a supplied starting point.
        opt = QOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)
        @test MOI.supports(opt, MOI.VariablePrimalStart(), MOI.VariableIndex)

        # UniversalFallback so the source model can hold VariablePrimalStart.
        model = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
        x = MOI.add_variables(model, 2)
        obj = MOI.ScalarQuadraticFunction(
            [
                MOI.ScalarQuadraticTerm(2.0, x[1], x[1]),
                MOI.ScalarQuadraticTerm(2.0, x[2], x[2]),
            ],
            [
                MOI.ScalarAffineTerm(-1.0, x[1]),
                MOI.ScalarAffineTerm(-1.0, x[2]),
            ],
            0.0,
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)
        MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(
                [
                    MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                    MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
                ],
                [-1.0],
            ),
            MOI.Zeros(1),
        )
        MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(
                [
                    MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                    MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[2])),
                ],
                [0.0, 0.0],
            ),
            MOI.Nonnegatives(2),
        )

        # Start only x[1]; x[2] falls back to zero.
        MOI.set(model, MOI.VariablePrimalStart(), x[1], 0.4)

        idxmap = MOI.copy_to(opt, model)
        @test MOI.get(opt, MOI.VariablePrimalStart(), idxmap[x[1]]) == 0.4
        @test MOI.get(opt, MOI.VariablePrimalStart(), idxmap[x[2]]) === nothing
        @test MOI.VariablePrimalStart() in
              MOI.get(opt, MOI.ListOfVariableAttributesSet())
        @test MOI.get(
            opt,
            MOI.ListOfVariablesWithAttributeSet(MOI.VariablePrimalStart()),
        ) == [idxmap[x[1]]]

        MOI.optimize!(opt)

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.VariablePrimal(), idxmap[x[1]]) ≈ 0.5 atol = 1e-5
        @test MOI.get(opt, MOI.VariablePrimal(), idxmap[x[2]]) ≈ 0.5 atol = 1e-5
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ -0.5 atol = 1e-5

        # Clearing the start returns the optimizer to default initialisation.
        MOI.set(opt, MOI.VariablePrimalStart(), idxmap[x[1]], nothing)
        @test MOI.get(opt, MOI.VariablePrimalStart(), idxmap[x[1]]) === nothing
        @test isempty(MOI.get(opt, MOI.ListOfVariableAttributesSet()))
        MOI.optimize!(opt)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
    end

    @testset "Linear objective with SOC via MOI" begin
        # min x1
        # s.t. (x1, x2, x3) ∈ SOC(3)
        #      x2 = 1, x3 = 0
        # Solution: x = [1, 1, 0]

        opt = QOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)

        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 3)

        # Objective: min x1
        obj = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, x[1])],
            0.0,
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        # (x1, x2, x3) ∈ SOC(3)
        soc_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[2])),
                MOI.VectorAffineTerm(3, MOI.ScalarAffineTerm(1.0, x[3])),
            ],
            [0.0, 0.0, 0.0],
        )
        MOI.add_constraint(model, soc_func, MOI.SecondOrderCone(3))

        # x2 = 1, x3 = 0
        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[3])),
            ],
            [-1.0, 0.0],
        )
        MOI.add_constraint(model, eq_func, MOI.Zeros(2))

        idxmap = MOI.copy_to(opt, model)
        MOI.optimize!(opt)

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        x1_val = MOI.get(opt, MOI.VariablePrimal(), idxmap[x[1]])
        x2_val = MOI.get(opt, MOI.VariablePrimal(), idxmap[x[2]])
        x3_val = MOI.get(opt, MOI.VariablePrimal(), idxmap[x[3]])
        @test x1_val ≈ 1.0 atol = 1e-4
        @test x2_val ≈ 1.0 atol = 1e-4
        @test abs(x3_val) < 1e-4
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 1.0 atol = 1e-4
    end

    @testset "Maximization" begin
        # max -x1^2 - x2^2 + x1 + x2
        # s.t. x1 + x2 = 1, x1,x2 >= 0
        # Equivalent to min x1^2 + x2^2 - x1 - x2 s.t. same
        # Solution: x = [0.5, 0.5], obj = 0.5

        opt = QOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)

        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 2)

        quad_terms = [
            MOI.ScalarQuadraticTerm(-2.0, x[1], x[1]),
            MOI.ScalarQuadraticTerm(-2.0, x[2], x[2]),
        ]
        aff_terms = [
            MOI.ScalarAffineTerm(1.0, x[1]),
            MOI.ScalarAffineTerm(1.0, x[2]),
        ]
        obj = MOI.ScalarQuadraticFunction(quad_terms, aff_terms, 0.0)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [-1.0],
        )
        MOI.add_constraint(model, eq_func, MOI.Zeros(1))

        nn_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [0.0, 0.0],
        )
        MOI.add_constraint(model, nn_func, MOI.Nonnegatives(2))

        idxmap = MOI.copy_to(opt, model)
        MOI.optimize!(opt)

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 0.5 atol = 1e-4
    end

    @testset "Empty model" begin
        opt = QOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)
        model = MOI.Utilities.Model{Float64}()
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        idxmap = MOI.copy_to(opt, model)
        MOI.optimize!(opt)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 0.0
    end

    @testset "Iteration limit" begin
        # Use very few iterations so the solver cannot converge
        opt = QOCO.Optimizer(; max_iters = 1)
        MOI.set(opt, MOI.Silent(), true)

        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 2)

        quad_terms = [
            MOI.ScalarQuadraticTerm(2.0, x[1], x[1]),
            MOI.ScalarQuadraticTerm(2.0, x[2], x[2]),
        ]
        aff_terms = [
            MOI.ScalarAffineTerm(-1.0, x[1]),
            MOI.ScalarAffineTerm(-1.0, x[2]),
        ]
        obj = MOI.ScalarQuadraticFunction(quad_terms, aff_terms, 0.0)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [-1.0],
        )
        MOI.add_constraint(model, eq_func, MOI.Zeros(1))

        nn_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[2])),
            ],
            [0.0, 0.0],
        )
        MOI.add_constraint(model, nn_func, MOI.Nonnegatives(2))

        idxmap = MOI.copy_to(opt, model)
        MOI.optimize!(opt)

        status = MOI.get(opt, MOI.TerminationStatus())
        @test status in (MOI.ITERATION_LIMIT, MOI.ALMOST_OPTIMAL)
    end

    @testset "Infeasible problem" begin
        # x1 = 1, x1 = 2 (contradictory equality constraints)
        opt = QOCO.Optimizer()
        MOI.set(opt, MOI.Silent(), true)

        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 1)

        obj = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, x[1])],
            0.0,
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

        # P = 0
        # x1 = 1 AND x1 = 2  (infeasible)
        eq_func = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[1])),
            ],
            [-1.0, -2.0],
        )
        MOI.add_constraint(model, eq_func, MOI.Zeros(2))

        idxmap = MOI.copy_to(opt, model)
        MOI.optimize!(opt)

        # QOCO doesn't have a dedicated infeasible status; it will report
        # either iteration limit, numerical error, or inaccurate solution
        status = MOI.get(opt, MOI.TerminationStatus())
        @test status in (
            MOI.ITERATION_LIMIT,
            MOI.NUMERICAL_ERROR,
            MOI.ALMOST_OPTIMAL,
            MOI.OTHER_ERROR,
        )
    end
end

@testset "MOI.Test" begin
    optimizer = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            QOCO.Optimizer(),
        ),
        Float64,
    )
    MOI.set(optimizer, MOI.Silent(), true)

    MOI.Test.runtests(
        optimizer,
        MOI.Test.Config(
            atol = 1e-3,
            rtol = 1e-3,
            optimal_status = MOI.OPTIMAL,
            exclude = Any[
                MOI.ConstraintBasisStatus,
                MOI.VariableBasisStatus,
                MOI.ObjectiveBound,
                MOI.TimeLimitSec,
            ],
        );
        exclude = Any[
            # QOCO does not support these cone types
            "test_conic_ExponentialCone",
            "test_conic_DualExponentialCone",
            "test_conic_PowerCone",
            "test_conic_DualPowerCone",
            "test_conic_GeometricMeanCone",
            r"test_conic_RootDetConeTriangle.*",
            r"test_conic_RootDetConeSquare.*",
            r"test_conic_LogDetConeTriangle.*",
            r"test_conic_LogDetConeSquare.*",
            r"test_conic_PositiveSemidefiniteConeTriangle.*",
            r"test_conic_PositiveSemidefiniteConeSquare.*",
            "test_conic_RelativeEntropyCone",
            r"test_conic_NormSpectralCone.*",
            r"test_conic_NormNuclearCone.*",
            "test_conic_NormInfinityCone",
            r"test_conic_NormOneCone.*",
            "test_conic_HermitianPositiveSemidefiniteConeTriangle",
            "test_conic_NormCone",
            # No integer support
            "test_solve_SOS",
            "test_constraint_ZeroOne",
            "test_constraint_Integer",
            # QOCO doesn't detect infeasibility or unboundedness
            r"test_conic_linear_INFEASIBLE.*",
            "test_conic_NonnegToNonworking",
            "test_conic_SecondOrderCone_INFEASIBLE",
            "test_conic_SecondOrderCone_negative_post_bound_2",
            "test_conic_SecondOrderCone_negative_post_bound_3",
            "test_conic_SecondOrderCone_no_initial_bound",
            r"test_conic_RotatedSecondOrderCone_INFEASIBLE.*",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE",
            "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            "test_infeasible",
            r"test_linear_INFEASIBLE.*",
            r"test_linear_DUAL_INFEASIBLE.*",
            "test_unbounded",
            # Quadratic constraints not supported (only quadratic objectives)
            "test_quadratic_constraint_",
        ],
    )
end

@testset "JuMP smoke test" begin
    model = Model(QOCO.Optimizer)
    set_silent(model)

    @variable(model, x[1:2] >= 0)
    @constraint(model, x[1] + x[2] == 1)
    @objective(model, Min, x[1]^2 + x[2]^2 - x[1] - x[2])

    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test value(x[1]) ≈ 0.5 atol = 1e-4
    @test value(x[2]) ≈ 0.5 atol = 1e-4
    @test objective_value(model) ≈ -0.5 atol = 1e-4
end
