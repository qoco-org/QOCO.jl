using QOCO
using Test

@testset "C Wrapper" begin
    @testset "Struct layouts match QOCO 0.3.2" begin
        # A mismatched QOCOSolver layout makes qoco_setup write past the end of
        # the allocation, so pin the widths that the C headers define.
        @test sizeof(QOCO.QOCOSolver) == 5 * sizeof(Ptr{Cvoid})
        @test sizeof(QOCO.QOCOSettings) == 96
        @test fieldnames(QOCO.QOCOSettings) == (
            :max_iters, :ruiz_iters, :max_ir_iters, :ir_tol,
            :kkt_static_reg_P, :kkt_static_reg_A, :kkt_static_reg_G,
            :kkt_dynamic_reg, :abstol, :reltol, :abstol_inacc, :reltol_inacc,
            :verbose,
        )
        @test fieldnames(QOCO.QOCOSolution) == (
            :x, :s, :y, :z, :iters, :ir_iters, :setup_time_sec,
            :solve_time_sec, :analysis_time_sec, :obj, :pres, :dres, :gap,
            :status,
        )
    end

    @testset "QOCO's own defaults" begin
        settings = QOCO.c_default_settings()
        @test settings.max_iters == 200
        @test settings.ruiz_iters == 0
        @test settings.max_ir_iters == 5
        @test settings.ir_tol ≈ 1e-6
        @test settings.kkt_static_reg_A ≈ 1e-8
        @test settings.kkt_dynamic_reg ≈ 1e-11
        @test settings.abstol ≈ 1e-7
        @test settings.reltol ≈ 1e-7
        @test settings.abstol_inacc ≈ 1e-5
        @test settings.reltol_inacc ≈ 1e-5
        @test settings.verbose == 0x00
    end

    @testset "Default settings" begin
        settings = QOCO.default_settings()

        # QOCO.jl raises the P-block static regularization; see
        # QOCO.SETTING_OVERRIDES for why.
        @test QOCO.SETTING_OVERRIDES.kkt_static_reg_P ≈ 1e-8
        @test settings.kkt_static_reg_P ≈ QOCO.SETTING_OVERRIDES.kkt_static_reg_P

        # Every field the wrapper does not override must still match QOCO's own
        # default, so an upstream change is never silently masked.
        c_settings = QOCO.c_default_settings()
        for name in fieldnames(QOCO.QOCOSettings)
            haskey(QOCO.SETTING_OVERRIDES, name) && continue
            @test getfield(settings, name) == getfield(c_settings, name)
        end

        @test settings.max_iters == 200
        @test settings.abstol ≈ 1e-7
        @test settings.reltol ≈ 1e-7
        @test settings.verbose == 0x00
    end

    @testset "Overridden default repairs the degenerate LP" begin
        # min x1 + x2  s.t.  x1 + x2 >= 1,  x1 + x2 >= 2.
        # No quadratic term and a rank-deficient G, so the KKT (1,1) block is
        # exactly kkt_static_reg_P * I. QOCO 0.3.2's own 1e-13 stalls the primal
        # residual at 1e-4 and exits QOCO_NUMERICAL_ERROR here.
        function solve_degenerate_lp(settings)
            n, p, m, l, nsoc = 2, 0, 2, 2, 0
            q = QOCO.QOCOInt[0]

            Px, Pp, Pi = QOCO.QOCOFloat[0.0], QOCO.QOCOInt[0, 0, 0], QOCO.QOCOInt[0]
            P = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
            QOCO.qoco_set_csc!(P, n, n, 0, Px, Pp, Pi)
            c_vec = QOCO.QOCOFloat[1.0, 1.0]

            Ax, Ap, Ai = QOCO.QOCOFloat[0.0], QOCO.QOCOInt[0, 0, 0], QOCO.QOCOInt[0]
            A = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
            QOCO.qoco_set_csc!(A, p, n, 0, Ax, Ap, Ai)
            b_vec = QOCO.QOCOFloat[0.0]

            Gx = QOCO.QOCOFloat[-1.0, -1.0, -1.0, -1.0]
            Gp, Gi = QOCO.QOCOInt[0, 2, 4], QOCO.QOCOInt[0, 1, 0, 1]
            G = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
            QOCO.qoco_set_csc!(G, m, n, 4, Gx, Gp, Gi)
            h_vec = QOCO.QOCOFloat[-1.0, -2.0]

            settings.verbose = 0x00
            solver_ptr = QOCO.qoco_solver_alloc()
            err = GC.@preserve Px Pp Pi Ax Ap Ai Gx Gp Gi QOCO.qoco_setup!(
                solver_ptr, n, m, p, P, c_vec, A, b_vec, G, h_vec, l, nsoc, q,
                settings,
            )
            err == QOCO.QOCO_NO_ERROR || error("setup failed with code $err")
            QOCO.qoco_solve!(solver_ptr)
            sol = QOCO.get_solution(solver_ptr)
            result = (status = sol.status, pres = sol.pres, obj = sol.obj)
            QOCO.qoco_cleanup!(solver_ptr)
            return result
        end

        result = solve_degenerate_lp(QOCO.default_settings())
        @test result.status == QOCO.QOCO_SOLVED
        @test result.pres < 1e-6
        @test result.obj ≈ 2.0 atol = 1e-5
    end

    @testset "Small QP via C API" begin
        # min (1/2) x'Px + c'x
        # s.t. Gx <=_C h (nonneg cone: x >= 0)
        #      Ax = b
        #
        # P = [2 0; 0 2], c = [-1; -1]
        # G = [-1 0; 0 -1], h = [0; 0] (x >= 0 ⟺ -x ≤ 0)
        # A = [1 1], b = [1] (x1 + x2 = 1)
        #
        # Solution: x = [0.5, 0.5], obj = -0.75

        n = 2   # variables
        m = 2   # conic constraints (nonneg)
        p = 1   # equality constraints
        l = 2   # nonneg cone dimension
        nsoc = 0
        q = QOCO.QOCOInt[0]

        # P (upper triangular, CSC, 0-indexed)
        P = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
        Px = QOCO.QOCOFloat[2.0, 2.0]
        Pp = QOCO.QOCOInt[0, 1, 2]
        Pi = QOCO.QOCOInt[0, 1]
        QOCO.qoco_set_csc!(P, n, n, 2, Px, Pp, Pi)

        c_vec = QOCO.QOCOFloat[-1.0, -1.0]

        # A (CSC, 0-indexed)
        A = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
        Ax = QOCO.QOCOFloat[1.0, 1.0]
        Ap = QOCO.QOCOInt[0, 1, 2]
        Ai = QOCO.QOCOInt[0, 0]
        QOCO.qoco_set_csc!(A, p, n, 2, Ax, Ap, Ai)

        b_vec = QOCO.QOCOFloat[1.0]

        # G (CSC, 0-indexed)
        G = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
        Gx = QOCO.QOCOFloat[-1.0, -1.0]
        Gp = QOCO.QOCOInt[0, 1, 2]
        Gi = QOCO.QOCOInt[0, 1]
        QOCO.qoco_set_csc!(G, m, n, 2, Gx, Gp, Gi)

        h_vec = QOCO.QOCOFloat[0.0, 0.0]

        settings = QOCO.default_settings()
        settings.verbose = 0x00

        solver_ptr = QOCO.qoco_solver_alloc()
        err = QOCO.qoco_setup!(solver_ptr, n, m, p, P, c_vec, A, b_vec, G, h_vec, l, nsoc, q, settings)
        @test err == QOCO.QOCO_NO_ERROR

        QOCO.qoco_solve!(solver_ptr)
        sol = QOCO.get_solution(solver_ptr)

        @test sol.status == QOCO.QOCO_SOLVED
        x = unsafe_wrap(Array, sol.x, n)
        @test x[1] ≈ 0.5 atol = 1e-5
        @test x[2] ≈ 0.5 atol = 1e-5
        @test sol.obj ≈ -0.5 atol = 1e-5

        QOCO.qoco_cleanup!(solver_ptr)
    end

    @testset "SOC constraint via C API" begin
        # min x1
        # s.t. ||[x2; x3]|| <= x1 (SOC)
        #      x2 = 1, x3 = 0
        # Solution: x = [1, 1, 0]

        n = 3
        p = 2   # x2=1, x3=0
        m = 3   # SOC of dim 3
        l = 0
        nsoc = 1
        q = QOCO.QOCOInt[3]

        # P = 0 (linear objective)
        P = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
        Px = QOCO.QOCOFloat[0.0]
        Pp = QOCO.QOCOInt[0, 0, 0, 0]
        Pi = QOCO.QOCOInt[0]
        QOCO.qoco_set_csc!(P, n, n, 0, Px, Pp, Pi)

        c_vec = QOCO.QOCOFloat[1.0, 0.0, 0.0]

        # A: x2 = 1, x3 = 0
        A = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
        A_x = QOCO.QOCOFloat[1.0, 1.0]
        A_p = QOCO.QOCOInt[0, 0, 1, 2]
        A_i = QOCO.QOCOInt[0, 1]
        QOCO.qoco_set_csc!(A, p, n, 2, A_x, A_p, A_i)

        b_vec = QOCO.QOCOFloat[1.0, 0.0]

        # G: SOC ||[x2;x3]|| <= x1
        # Written as: (x1, x2, x3) ∈ SOC(3)
        # QOCO: h - Gx ∈ SOC → G = -I (rows for SOC variables), h = 0
        G = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
        G_x = QOCO.QOCOFloat[-1.0, -1.0, -1.0]
        G_p = QOCO.QOCOInt[0, 1, 2, 3]
        G_i = QOCO.QOCOInt[0, 1, 2]
        QOCO.qoco_set_csc!(G, m, n, 3, G_x, G_p, G_i)

        h_vec = QOCO.QOCOFloat[0.0, 0.0, 0.0]

        settings = QOCO.default_settings()
        settings.verbose = 0x00

        solver_ptr = QOCO.qoco_solver_alloc()
        err = QOCO.qoco_setup!(solver_ptr, n, m, p, P, c_vec, A, b_vec, G, h_vec, l, nsoc, q, settings)
        @test err == QOCO.QOCO_NO_ERROR

        QOCO.qoco_solve!(solver_ptr)
        sol = QOCO.get_solution(solver_ptr)

        @test sol.status == QOCO.QOCO_SOLVED
        x = unsafe_wrap(Array, sol.x, n)
        @test x[1] ≈ 1.0 atol = 1e-5
        @test x[2] ≈ 1.0 atol = 1e-5
        @test abs(x[3]) < 1e-5

        QOCO.qoco_cleanup!(solver_ptr)
    end

    @testset "Custom starting point via qoco_set_x0!" begin
        # Same QP as "Small QP via C API"; solving from a supplied x0 must
        # reach the same optimum.
        n, m, p, l, nsoc = 2, 2, 1, 2, 0
        q = QOCO.QOCOInt[0]

        Px, Pp, Pi = QOCO.QOCOFloat[2.0, 2.0], QOCO.QOCOInt[0, 1, 2], QOCO.QOCOInt[0, 1]
        P = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
        QOCO.qoco_set_csc!(P, n, n, 2, Px, Pp, Pi)
        c_vec = QOCO.QOCOFloat[-1.0, -1.0]

        Ax, Ap, Ai = QOCO.QOCOFloat[1.0, 1.0], QOCO.QOCOInt[0, 1, 2], QOCO.QOCOInt[0, 0]
        A = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
        QOCO.qoco_set_csc!(A, p, n, 2, Ax, Ap, Ai)
        b_vec = QOCO.QOCOFloat[1.0]

        Gx, Gp, Gi = QOCO.QOCOFloat[-1.0, -1.0], QOCO.QOCOInt[0, 1, 2], QOCO.QOCOInt[0, 1]
        G = QOCO.QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
        QOCO.qoco_set_csc!(G, m, n, 2, Gx, Gp, Gi)
        h_vec = QOCO.QOCOFloat[0.0, 0.0]

        settings = QOCO.default_settings()
        settings.verbose = 0x00

        solver_ptr = QOCO.qoco_solver_alloc()
        # The CSC structs only hold pointers into these arrays, so root them
        # across the setup call.
        err = GC.@preserve Px Pp Pi Ax Ap Ai Gx Gp Gi QOCO.qoco_setup!(
            solver_ptr, n, m, p, P, c_vec, A, b_vec, G, h_vec, l, nsoc, q, settings,
        )
        @test err == QOCO.QOCO_NO_ERROR

        QOCO.qoco_set_x0!(solver_ptr, QOCO.QOCOFloat[0.5, 0.5])
        QOCO.qoco_solve!(solver_ptr)
        sol = QOCO.get_solution(solver_ptr)

        @test sol.status == QOCO.QOCO_SOLVED
        x = unsafe_wrap(Array, sol.x, n)
        @test x[1] ≈ 0.5 atol = 1e-5
        @test x[2] ≈ 0.5 atol = 1e-5

        # Fields new in QOCO 0.3: these are only meaningful if the layout is
        # right, so sanity-check that they are non-negative and finite.
        @test sol.ir_iters >= 0
        @test isfinite(sol.analysis_time_sec) && sol.analysis_time_sec >= 0.0
        @test isfinite(sol.setup_time_sec) && sol.setup_time_sec >= 0.0
        @test isfinite(sol.solve_time_sec) && sol.solve_time_sec >= 0.0

        # Clearing the starting point must be accepted too.
        QOCO.qoco_set_x0!(solver_ptr, C_NULL)
        QOCO.qoco_solve!(solver_ptr)
        @test QOCO.get_solution(solver_ptr).status == QOCO.QOCO_SOLVED

        QOCO.qoco_cleanup!(solver_ptr)
    end
end
