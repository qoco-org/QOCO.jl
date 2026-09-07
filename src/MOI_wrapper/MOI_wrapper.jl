import MathOptInterface as MOI

const SupportedObjectiveFunction = Union{
    MOI.VariableIndex,
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
}

const _SUPPORTED_CONSTRAINT_FUNCTION = MOI.VectorAffineFunction{Float64}

_zero_objective() = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], 0.0)

mutable struct Optimizer <: MOI.AbstractOptimizer
    silent::Bool
    settings::Dict{Symbol, Any}
    name::String
    name_set::Bool
    objective_sense_set::Bool
    objective_set::Bool
    variable_names_set::Bool
    eq_names_set::Bool
    nn_names_set::Bool
    soc_names_set::Bool

    n::Int
    variable_names::Vector{String}
    x0::Vector{Union{Nothing, QOCOFloat}}
    x0_set::Bool
    objective::SupportedObjectiveFunction
    sense::MOI.OptimizationSense
    P_colptr::Vector{QOCOInt}
    P_rowval::Vector{QOCOInt}
    P_nzval::Vector{QOCOFloat}
    c::Vector{QOCOFloat}

    p::Int
    A_colptr::Vector{QOCOInt}
    A_rowval::Vector{QOCOInt}
    A_nzval::Vector{QOCOFloat}
    b::Vector{QOCOFloat}

    m::Int
    l::Int
    nsoc::Int
    q::Vector{QOCOInt}
    G_colptr::Vector{QOCOInt}
    G_rowval::Vector{QOCOInt}
    G_nzval::Vector{QOCOFloat}
    h::Vector{QOCOFloat}

    A_coo_i::Vector{QOCOInt}
    A_coo_j::Vector{QOCOInt}
    A_coo_v::Vector{QOCOFloat}
    G_coo_i::Vector{QOCOInt}
    G_coo_j::Vector{QOCOInt}
    G_coo_v::Vector{QOCOFloat}

    eq_constraints::Vector{Tuple{Int64, UnitRange{Int}}}
    eq_functions::Vector{MOI.VectorAffineFunction{Float64}}
    eq_sets::Vector{MOI.Zeros}
    eq_names::Vector{String}
    nn_constraints::Vector{Tuple{Int64, UnitRange{Int}}}
    nn_functions::Vector{MOI.VectorAffineFunction{Float64}}
    nn_sets::Vector{MOI.Nonnegatives}
    nn_names::Vector{String}
    soc_constraints::Vector{Tuple{Int64, UnitRange{Int}}}
    soc_functions::Vector{MOI.VectorAffineFunction{Float64}}
    soc_sets::Vector{MOI.SecondOrderCone}
    soc_names::Vector{String}
    constraint_offset::Dict{Int64, Tuple{Symbol, Int}}
    next_constraint_id::Int64

    has_result::Bool
    solve_status::QOCOInt
    termination_status::MOI.TerminationStatusCode
    raw_status::String
    primal_status::MOI.ResultStatusCode
    dual_status::MOI.ResultStatusCode
    result_count::Int
    primal::Vector{Float64}
    dual_eq::Vector{Float64}
    dual_cone::Vector{Float64}
    slack_cone::Vector{Float64}
    obj_val::Float64
    dual_obj_val::Float64
    solve_time::Float64
    setup_time::Float64
    analysis_time::Float64
    iterations::Int
    ir_iterations::Int
    pres::Float64
    dres::Float64
    gap::Float64
end

function Optimizer(; kwargs...)
    opt = Optimizer(
        false,
        Dict{Symbol, Any}(),
        "",
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        0,
        String[],
        Union{Nothing, QOCOFloat}[],
        false,
        _zero_objective(),
        MOI.FEASIBILITY_SENSE,
        QOCOInt[],
        QOCOInt[],
        QOCOFloat[],
        QOCOFloat[],
        0,
        QOCOInt[],
        QOCOInt[],
        QOCOFloat[],
        QOCOFloat[],
        0,
        0,
        0,
        QOCOInt[],
        QOCOInt[],
        QOCOInt[],
        QOCOFloat[],
        QOCOFloat[],
        QOCOInt[],
        QOCOInt[],
        QOCOFloat[],
        QOCOInt[],
        QOCOInt[],
        QOCOFloat[],
        Tuple{Int64, UnitRange{Int}}[],
        MOI.VectorAffineFunction{Float64}[],
        MOI.Zeros[],
        String[],
        Tuple{Int64, UnitRange{Int}}[],
        MOI.VectorAffineFunction{Float64}[],
        MOI.Nonnegatives[],
        String[],
        Tuple{Int64, UnitRange{Int}}[],
        MOI.VectorAffineFunction{Float64}[],
        MOI.SecondOrderCone[],
        String[],
        Dict{Int64, Tuple{Symbol, Int}}(),
        0,
        false,
        QOCO_UNSOLVED,
        MOI.OPTIMIZE_NOT_CALLED,
        "OPTIMIZE_NOT_CALLED",
        MOI.NO_SOLUTION,
        MOI.NO_SOLUTION,
        0,
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        NaN,   # obj_val
        NaN,   # dual_obj_val
        NaN,   # solve_time
        NaN,   # setup_time
        NaN,   # analysis_time
        0,     # iterations
        0,     # ir_iterations
        NaN,   # pres
        NaN,   # dres
        NaN,   # gap
    )
    for (key, value) in kwargs
        MOI.set(opt, MOI.RawOptimizerAttribute(String(key)), value)
    end
    return opt
end

Base.summary(io::IO, ::Optimizer) = print(io, "QOCO optimizer")

function MOI.default_cache(::Optimizer, ::Type{Float64})
    return MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
end

MOI.supports_incremental_interface(::Optimizer) = false

function MOI.is_empty(opt::Optimizer)
    return opt.n == 0 && opt.p == 0 && opt.m == 0
end

function _reset_results!(opt::Optimizer)
    opt.has_result = false
    opt.solve_status = QOCO_UNSOLVED
    opt.termination_status = MOI.OPTIMIZE_NOT_CALLED
    opt.raw_status = "OPTIMIZE_NOT_CALLED"
    opt.primal_status = MOI.NO_SOLUTION
    opt.dual_status = MOI.NO_SOLUTION
    opt.result_count = 0
    opt.primal = Float64[]
    opt.dual_eq = Float64[]
    opt.dual_cone = Float64[]
    opt.slack_cone = Float64[]
    opt.obj_val = NaN
    opt.dual_obj_val = NaN
    opt.solve_time = NaN
    opt.setup_time = NaN
    opt.analysis_time = NaN
    opt.iterations = 0
    opt.ir_iterations = 0
    opt.pres = NaN
    opt.dres = NaN
    opt.gap = NaN
    return
end

function MOI.empty!(opt::Optimizer)
    opt.name = ""
    opt.name_set = false
    opt.objective_sense_set = false
    opt.objective_set = false
    opt.variable_names_set = false
    opt.eq_names_set = false
    opt.nn_names_set = false
    opt.soc_names_set = false

    opt.n = 0
    opt.variable_names = String[]
    opt.x0 = Union{Nothing, QOCOFloat}[]
    opt.x0_set = false
    opt.objective = _zero_objective()
    opt.sense = MOI.FEASIBILITY_SENSE
    opt.P_colptr = QOCOInt[]
    opt.P_rowval = QOCOInt[]
    opt.P_nzval = QOCOFloat[]
    opt.c = QOCOFloat[]

    opt.p = 0
    opt.A_colptr = QOCOInt[]
    opt.A_rowval = QOCOInt[]
    opt.A_nzval = QOCOFloat[]
    opt.b = QOCOFloat[]

    opt.m = 0
    opt.l = 0
    opt.nsoc = 0
    opt.q = QOCOInt[]
    opt.G_colptr = QOCOInt[]
    opt.G_rowval = QOCOInt[]
    opt.G_nzval = QOCOFloat[]
    opt.h = QOCOFloat[]

    empty!(opt.A_coo_i)
    empty!(opt.A_coo_j)
    empty!(opt.A_coo_v)
    empty!(opt.G_coo_i)
    empty!(opt.G_coo_j)
    empty!(opt.G_coo_v)

    empty!(opt.eq_constraints)
    empty!(opt.eq_functions)
    empty!(opt.eq_sets)
    empty!(opt.eq_names)
    empty!(opt.nn_constraints)
    empty!(opt.nn_functions)
    empty!(opt.nn_sets)
    empty!(opt.nn_names)
    empty!(opt.soc_constraints)
    empty!(opt.soc_functions)
    empty!(opt.soc_sets)
    empty!(opt.soc_names)
    empty!(opt.constraint_offset)
    opt.next_constraint_id = 0

    _reset_results!(opt)
    return
end

MOI.get(::Optimizer, ::MOI.SolverName) = "QOCO"
MOI.get(::Optimizer, ::MOI.SolverVersion) = "0.3.2"
MOI.get(opt::Optimizer, ::MOI.RawSolver) = opt

MOI.supports(::Optimizer, ::MOI.Name) = true
MOI.get(opt::Optimizer, ::MOI.Name) = opt.name

function MOI.set(opt::Optimizer, ::MOI.Name, value::String)
    opt.name = value
    opt.name_set = true
    return
end

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.get(opt::Optimizer, ::MOI.Silent) = opt.silent

function MOI.set(opt::Optimizer, ::MOI.Silent, value::Bool)
    opt.silent = value
    return
end

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = false

const _SETTINGS_FIELDS = Dict{String, Tuple{Symbol, Type}}(
    "max_iters"         => (:max_iters, Int),
    "ruiz_iters"        => (:ruiz_iters, Int),
    "max_ir_iters"      => (:max_ir_iters, Int),
    "ir_tol"            => (:ir_tol, Float64),
    "kkt_static_reg_P"  => (:kkt_static_reg_P, Float64),
    "kkt_static_reg_A"  => (:kkt_static_reg_A, Float64),
    "kkt_static_reg_G"  => (:kkt_static_reg_G, Float64),
    "kkt_dynamic_reg"   => (:kkt_dynamic_reg, Float64),
    "abstol"            => (:abstol, Float64),
    "reltol"            => (:reltol, Float64),
    "abstol_inacc"      => (:abstol_inacc, Float64),
    "reltol_inacc"      => (:reltol_inacc, Float64),
    "verbose"           => (:verbose, Bool),
)

function _default_setting_value(key::Symbol)
    settings = default_settings()
    value = getfield(settings, key)
    return key == :verbose ? value != 0x00 : value
end

function MOI.supports(::Optimizer, attr::MOI.RawOptimizerAttribute)
    return haskey(_SETTINGS_FIELDS, attr.name)
end

function MOI.get(opt::Optimizer, attr::MOI.RawOptimizerAttribute)
    haskey(_SETTINGS_FIELDS, attr.name) || throw(MOI.UnsupportedAttribute(attr))
    key, _ = _SETTINGS_FIELDS[attr.name]
    return get(opt.settings, key, _default_setting_value(key))
end

function MOI.set(opt::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    haskey(_SETTINGS_FIELDS, attr.name) || throw(MOI.UnsupportedAttribute(attr))
    key, T = _SETTINGS_FIELDS[attr.name]
    opt.settings[key] = convert(T, value)
    return
end

function _build_settings(opt::Optimizer)
    settings = default_settings()
    for (key, value) in opt.settings
        if key == :verbose
            settings.verbose = UInt8(value ? 1 : 0)
        else
            setfield!(settings, key, convert(fieldtype(QOCOSettings, key), value))
        end
    end
    if opt.silent
        settings.verbose = UInt8(0)
    end
    return settings
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.get(opt::Optimizer, ::MOI.ObjectiveSense) = opt.sense
MOI.get(opt::Optimizer, ::MOI.ObjectiveFunctionType) = typeof(opt.objective)

function MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{F}) where {F<:SupportedObjectiveFunction}
    return true
end

MOI.supports(::Optimizer, ::MOI.VariableName, ::Type{MOI.VariableIndex}) = true
MOI.supports(::Optimizer, ::MOI.ConstraintName, ::Type{<:MOI.ConstraintIndex}) = true

# QOCO 0.3 exposes `qoco_set_x0`, which takes a full-length primal starting
# point. Variables left unset default to zero.
MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex}) = true

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorAffineFunction{Float64}},
    ::Type{<:Union{MOI.Zeros, MOI.Nonnegatives, MOI.SecondOrderCone}},
)
    return true
end

function _lookup_unique_name(names::Vector{String}, name::String)
    isempty(name) && return nothing
    matches = findall(==(name), names)
    isempty(matches) && return nothing
    length(matches) == 1 || error("Duplicate name detected: $name")
    return only(matches)
end

_constraint_entries(opt::Optimizer, ::Type{MOI.Zeros}) = opt.eq_constraints
_constraint_entries(opt::Optimizer, ::Type{MOI.Nonnegatives}) = opt.nn_constraints
_constraint_entries(opt::Optimizer, ::Type{MOI.SecondOrderCone}) = opt.soc_constraints

_constraint_functions(opt::Optimizer, ::Type{MOI.Zeros}) = opt.eq_functions
_constraint_functions(opt::Optimizer, ::Type{MOI.Nonnegatives}) = opt.nn_functions
_constraint_functions(opt::Optimizer, ::Type{MOI.SecondOrderCone}) = opt.soc_functions

_constraint_sets(opt::Optimizer, ::Type{MOI.Zeros}) = opt.eq_sets
_constraint_sets(opt::Optimizer, ::Type{MOI.Nonnegatives}) = opt.nn_sets
_constraint_sets(opt::Optimizer, ::Type{MOI.SecondOrderCone}) = opt.soc_sets

_constraint_names(opt::Optimizer, ::Type{MOI.Zeros}) = opt.eq_names
_constraint_names(opt::Optimizer, ::Type{MOI.Nonnegatives}) = opt.nn_names
_constraint_names(opt::Optimizer, ::Type{MOI.SecondOrderCone}) = opt.soc_names

_constraint_kind(::Type{MOI.Zeros}) = :eq
_constraint_kind(::Type{MOI.Nonnegatives}) = :nn
_constraint_kind(::Type{MOI.SecondOrderCone}) = :soc

function _constraint_position(opt::Optimizer, ci::MOI.ConstraintIndex, kind::Symbol)
    data = get(opt.constraint_offset, ci.value, nothing)
    data === nothing && throw(MOI.InvalidIndex(ci))
    stored_kind, position = data
    stored_kind == kind || throw(MOI.InvalidIndex(ci))
    return position
end

MOI.is_valid(opt::Optimizer, vi::MOI.VariableIndex) = 1 <= vi.value <= opt.n
MOI.is_valid(::Optimizer, ::MOI.ConstraintIndex) = false

function MOI.is_valid(
    opt::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros},
)
    data = get(opt.constraint_offset, ci.value, nothing)
    return data !== nothing && first(data) == :eq
end

function MOI.is_valid(
    opt::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives},
)
    data = get(opt.constraint_offset, ci.value, nothing)
    return data !== nothing && first(data) == :nn
end

function MOI.is_valid(
    opt::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.SecondOrderCone},
)
    data = get(opt.constraint_offset, ci.value, nothing)
    return data !== nothing && first(data) == :soc
end

function _check_valid(opt::Optimizer, vi::MOI.VariableIndex)
    MOI.is_valid(opt, vi) || throw(MOI.InvalidIndex(vi))
    return
end

function _check_valid(
    opt::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros},
)
    MOI.is_valid(opt, ci) || throw(MOI.InvalidIndex(ci))
    return
end

function _check_valid(
    opt::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives},
)
    MOI.is_valid(opt, ci) || throw(MOI.InvalidIndex(ci))
    return
end

function _check_valid(
    opt::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.SecondOrderCone},
)
    MOI.is_valid(opt, ci) || throw(MOI.InvalidIndex(ci))
    return
end

MOI.get(opt::Optimizer, ::MOI.NumberOfVariables) = opt.n

function MOI.get(opt::Optimizer, ::MOI.ListOfVariableIndices)
    return [MOI.VariableIndex(i) for i in 1:opt.n]
end

function MOI.get(opt::Optimizer, ::MOI.ListOfVariableAttributesSet)
    attrs = MOI.AbstractVariableAttribute[]
    opt.variable_names_set && push!(attrs, MOI.VariableName())
    opt.x0_set && push!(attrs, MOI.VariablePrimalStart())
    return attrs
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ListOfVariablesWithAttributeSet{MOI.VariableName},
)
    return [
        MOI.VariableIndex(i) for i in eachindex(opt.variable_names)
        if !isempty(opt.variable_names[i])
    ]
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ListOfVariablesWithAttributeSet{MOI.VariablePrimalStart},
)
    return [
        MOI.VariableIndex(i) for i in eachindex(opt.x0)
        if opt.x0[i] !== nothing
    ]
end

function MOI.get(opt::Optimizer, ::MOI.ListOfModelAttributesSet)
    attrs = MOI.AbstractModelAttribute[]
    opt.name_set && push!(attrs, MOI.Name())
    opt.objective_sense_set && push!(attrs, MOI.ObjectiveSense())
    if opt.objective_set
        push!(attrs, MOI.ObjectiveFunction{typeof(opt.objective)}())
    end
    return attrs
end

function MOI.get(opt::Optimizer, ::MOI.NumberOfConstraints{F, S}) where {F, S}
    if F == MOI.VectorAffineFunction{Float64} && S == MOI.Zeros
        return length(opt.eq_constraints)
    elseif F == MOI.VectorAffineFunction{Float64} && S == MOI.Nonnegatives
        return length(opt.nn_constraints)
    elseif F == MOI.VectorAffineFunction{Float64} && S == MOI.SecondOrderCone
        return length(opt.soc_constraints)
    end
    return 0
end

function MOI.get(opt::Optimizer, ::MOI.ListOfConstraintTypesPresent)
    types = Tuple{Type, Type}[]
    !isempty(opt.eq_constraints) &&
        push!(types, (MOI.VectorAffineFunction{Float64}, MOI.Zeros))
    !isempty(opt.nn_constraints) &&
        push!(types, (MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives))
    !isempty(opt.soc_constraints) &&
        push!(types, (MOI.VectorAffineFunction{Float64}, MOI.SecondOrderCone))
    return types
end

function MOI.get(opt::Optimizer, ::MOI.ListOfConstraintIndices{F, S}) where {F, S}
    if F == MOI.VectorAffineFunction{Float64} && S == MOI.Zeros
        return MOI.ConstraintIndex{F, S}[
            MOI.ConstraintIndex{F, S}(id) for (id, _) in opt.eq_constraints
        ]
    elseif F == MOI.VectorAffineFunction{Float64} && S == MOI.Nonnegatives
        return MOI.ConstraintIndex{F, S}[
            MOI.ConstraintIndex{F, S}(id) for (id, _) in opt.nn_constraints
        ]
    elseif F == MOI.VectorAffineFunction{Float64} && S == MOI.SecondOrderCone
        return MOI.ConstraintIndex{F, S}[
            MOI.ConstraintIndex{F, S}(id) for (id, _) in opt.soc_constraints
        ]
    end
    return MOI.ConstraintIndex{F, S}[]
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ListOfConstraintAttributesSet{MOI.VectorAffineFunction{Float64}, MOI.Zeros},
)
    if opt.eq_names_set
        return MOI.AbstractConstraintAttribute[MOI.ConstraintName()]
    end
    return MOI.AbstractConstraintAttribute[]
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ListOfConstraintAttributesSet{MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives},
)
    if opt.nn_names_set
        return MOI.AbstractConstraintAttribute[MOI.ConstraintName()]
    end
    return MOI.AbstractConstraintAttribute[]
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ListOfConstraintAttributesSet{MOI.VectorAffineFunction{Float64}, MOI.SecondOrderCone},
)
    if opt.soc_names_set
        return MOI.AbstractConstraintAttribute[MOI.ConstraintName()]
    end
    return MOI.AbstractConstraintAttribute[]
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ListOfConstraintsWithAttributeSet{
        MOI.VectorAffineFunction{Float64},
        MOI.Zeros,
        MOI.ConstraintName,
    },
)
    return MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros}[
        MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros}(id)
        for ((id, _), name) in zip(opt.eq_constraints, opt.eq_names)
        if !isempty(name)
    ]
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ListOfConstraintsWithAttributeSet{
        MOI.VectorAffineFunction{Float64},
        MOI.Nonnegatives,
        MOI.ConstraintName,
    },
)
    return MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives}[
        MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives}(id)
        for ((id, _), name) in zip(opt.nn_constraints, opt.nn_names)
        if !isempty(name)
    ]
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ListOfConstraintsWithAttributeSet{
        MOI.VectorAffineFunction{Float64},
        MOI.SecondOrderCone,
        MOI.ConstraintName,
    },
)
    return MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.SecondOrderCone}[
        MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.SecondOrderCone}(id)
        for ((id, _), name) in zip(opt.soc_constraints, opt.soc_names)
        if !isempty(name)
    ]
end

function MOI.get(opt::Optimizer, ::MOI.VariableName, vi::MOI.VariableIndex)
    _check_valid(opt, vi)
    return opt.variable_names[vi.value]
end

function MOI.set(
    opt::Optimizer,
    ::MOI.VariableName,
    vi::MOI.VariableIndex,
    name::String,
)
    _check_valid(opt, vi)
    opt.variable_names[vi.value] = name
    opt.variable_names_set = true
    return
end

function MOI.get(opt::Optimizer, ::Type{MOI.VariableIndex}, name::String)
    idx = _lookup_unique_name(opt.variable_names, name)
    return idx === nothing ? nothing : MOI.VariableIndex(idx)
end

function MOI.get(
    opt::Optimizer,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
)
    _check_valid(opt, vi)
    value = opt.x0[vi.value]
    return value === nothing ? nothing : Float64(value)
end

function MOI.set(
    opt::Optimizer,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value::Union{Nothing, Real},
)
    _check_valid(opt, vi)
    opt.x0[vi.value] = value === nothing ? nothing : convert(QOCOFloat, value)
    opt.x0_set = any(!isnothing, opt.x0)
    return
end

"""
    _primal_start(opt) -> Union{Vector{QOCOFloat}, Nothing}

Assemble the length-`n` starting point to hand to `qoco_set_x0!`, or `nothing`
if no `MOI.VariablePrimalStart` was set. Variables without a start contribute
`0.0`, since QOCO only accepts a full vector.
"""
function _primal_start(opt::Optimizer)
    opt.x0_set || return nothing
    length(opt.x0) == opt.n || return nothing
    return QOCOFloat[value === nothing ? 0.0 : value for value in opt.x0]
end

function _constraint_name(
    opt::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, S},
) where {S}
    idx = _constraint_position(opt, ci, _constraint_kind(S))
    return _constraint_names(opt, S)[idx]
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ConstraintName,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros},
)
    return _constraint_name(opt, ci)
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ConstraintName,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives},
)
    return _constraint_name(opt, ci)
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ConstraintName,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.SecondOrderCone},
)
    return _constraint_name(opt, ci)
end

function _set_constraint_name!(
    opt::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, S},
    name::String,
) where {S}
    idx = _constraint_position(opt, ci, _constraint_kind(S))
    _constraint_names(opt, S)[idx] = name
    if S == MOI.Zeros
        opt.eq_names_set = true
    elseif S == MOI.Nonnegatives
        opt.nn_names_set = true
    else
        opt.soc_names_set = true
    end
    return
end

function MOI.set(
    opt::Optimizer,
    ::MOI.ConstraintName,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros},
    name::String,
)
    _set_constraint_name!(opt, ci, name)
    return
end

function MOI.set(
    opt::Optimizer,
    ::MOI.ConstraintName,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives},
    name::String,
)
    _set_constraint_name!(opt, ci, name)
    return
end

function MOI.set(
    opt::Optimizer,
    ::MOI.ConstraintName,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.SecondOrderCone},
    name::String,
)
    _set_constraint_name!(opt, ci, name)
    return
end

function _named_constraint(
    opt::Optimizer,
    name::String,
    ::Type{MOI.VectorAffineFunction{Float64}},
    ::Type{S},
) where {S}
    idx = _lookup_unique_name(_constraint_names(opt, S), name)
    idx === nothing && return nothing
    id, _ = _constraint_entries(opt, S)[idx]
    return MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, S}(id)
end

function MOI.get(
    opt::Optimizer,
    ::Type{MOI.ConstraintIndex{F, S}},
    name::String,
) where {F, S}
    if F == MOI.VectorAffineFunction{Float64} && S == MOI.Zeros
        return _named_constraint(opt, name, F, MOI.Zeros)
    elseif F == MOI.VectorAffineFunction{Float64} && S == MOI.Nonnegatives
        return _named_constraint(opt, name, F, MOI.Nonnegatives)
    elseif F == MOI.VectorAffineFunction{Float64} && S == MOI.SecondOrderCone
        return _named_constraint(opt, name, F, MOI.SecondOrderCone)
    end
    return nothing
end

function MOI.get(opt::Optimizer, ::Type{MOI.ConstraintIndex}, name::String)
    isempty(name) && return nothing
    matches = MOI.ConstraintIndex[]
    for S in (MOI.Zeros, MOI.Nonnegatives, MOI.SecondOrderCone)
        ci = _named_constraint(opt, name, MOI.VectorAffineFunction{Float64}, S)
        ci !== nothing && push!(matches, ci)
    end
    isempty(matches) && return nothing
    length(matches) == 1 || error("Duplicate name detected: $name")
    return only(matches)
end

function MOI.get(opt::Optimizer, ::MOI.ObjectiveFunction{MOI.VariableIndex})
    obj = opt.objective
    obj isa MOI.VariableIndex || throw(MOI.UnsupportedAttribute(MOI.ObjectiveFunction{MOI.VariableIndex}()))
    return obj
end

function MOI.get(opt::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}})
    obj = opt.objective
    obj isa MOI.ScalarAffineFunction{Float64} ||
        throw(MOI.UnsupportedAttribute(MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()))
    return copy(obj)
end

function MOI.get(opt::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}})
    obj = opt.objective
    obj isa MOI.ScalarQuadraticFunction{Float64} ||
        throw(MOI.UnsupportedAttribute(MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}()))
    return copy(obj)
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ConstraintFunction,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros},
)
    idx = _constraint_position(opt, ci, :eq)
    return copy(opt.eq_functions[idx])
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ConstraintFunction,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives},
)
    idx = _constraint_position(opt, ci, :nn)
    return copy(opt.nn_functions[idx])
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ConstraintFunction,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.SecondOrderCone},
)
    idx = _constraint_position(opt, ci, :soc)
    return copy(opt.soc_functions[idx])
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros},
)
    idx = _constraint_position(opt, ci, :eq)
    return opt.eq_sets[idx]
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives},
)
    idx = _constraint_position(opt, ci, :nn)
    return opt.nn_sets[idx]
end

function MOI.get(
    opt::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.SecondOrderCone},
)
    idx = _constraint_position(opt, ci, :soc)
    return opt.soc_sets[idx]
end

function _objective_constant(::MOI.VariableIndex)
    return 0.0
end

function _objective_constant(obj::MOI.ScalarAffineFunction{Float64})
    return obj.constant
end

function _objective_constant(obj::MOI.ScalarQuadraticFunction{Float64})
    return obj.constant
end

function _evaluate_objective(obj::MOI.VariableIndex, x::Vector{Float64})
    return x[obj.value]
end

function _evaluate_objective(obj::MOI.ScalarAffineFunction{Float64}, x::Vector{Float64})
    value = obj.constant
    for term in obj.terms
        value += term.coefficient * x[term.variable.value]
    end
    return value
end

function _evaluate_objective(obj::MOI.ScalarQuadraticFunction{Float64}, x::Vector{Float64})
    value = obj.constant
    for term in obj.affine_terms
        value += term.coefficient * x[term.variable.value]
    end
    for term in obj.quadratic_terms
        i = term.variable_1.value
        j = term.variable_2.value
        if i == j
            value += 0.5 * term.coefficient * x[i] * x[j]
        else
            value += term.coefficient * x[i] * x[j]
        end
    end
    return value
end

function _evaluate_vector_affine(
    func::MOI.VectorAffineFunction{Float64},
    x::Vector{Float64},
)
    values = copy(func.constants)
    for term in func.terms
        values[term.output_index] +=
            term.scalar_term.coefficient * x[term.scalar_term.variable.value]
    end
    return values
end

function _append_coo!(
    opt::Optimizer,
    matrix::Symbol,
    I::Vector{QOCOInt},
    J::Vector{QOCOInt},
    V::Vector{QOCOFloat},
)
    if matrix == :A
        append!(opt.A_coo_i, I)
        append!(opt.A_coo_j, J)
        append!(opt.A_coo_v, V)
    else
        append!(opt.G_coo_i, I)
        append!(opt.G_coo_j, J)
        append!(opt.G_coo_v, V)
    end
    return
end

function _set_zero_internal_objective!(opt::Optimizer)
    opt.c = zeros(QOCOFloat, opt.n)
    opt.P_colptr = zeros(QOCOInt, opt.n + 1)
    opt.P_rowval = QOCOInt[]
    opt.P_nzval = QOCOFloat[]
    return
end

function _apply_objective_sense!(opt::Optimizer)
    if opt.sense == MOI.FEASIBILITY_SENSE
        _set_zero_internal_objective!(opt)
    elseif opt.sense == MOI.MAX_SENSE
        opt.c .*= -1
        opt.P_nzval .*= -1
    end
    return
end

function _copy_model_attributes!(opt::Optimizer, src::MOI.ModelLike)
    for attr in MOI.get(src, MOI.ListOfModelAttributesSet())
        if attr == MOI.Name()
            opt.name = MOI.get(src, MOI.Name())
            opt.name_set = true
        elseif attr == MOI.ObjectiveSense()
            opt.objective_sense_set = true
        elseif attr isa MOI.ObjectiveFunction{MOI.VariableIndex} ||
               attr isa MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}} ||
               attr isa MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}
            opt.objective_set = true
        elseif attr isa MOI.ObjectiveFunction
            throw(MOI.UnsupportedAttribute(attr))
        else
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    opt.sense = MOI.get(src, MOI.ObjectiveSense())
    return
end

function _copy_variable_attributes!(
    opt::Optimizer,
    src::MOI.ModelLike,
    vis,
    idxmap::MOI.Utilities.IndexMap,
)
    copy_names = false
    copy_start = false
    for attr in MOI.get(src, MOI.ListOfVariableAttributesSet())
        if attr == MOI.VariableName()
            copy_names = true
        elseif attr == MOI.VariablePrimalStart()
            copy_start = true
        else
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    if copy_names
        opt.variable_names_set = true
        for vi in vis
            mapped = idxmap[vi]
            opt.variable_names[mapped.value] = MOI.get(src, MOI.VariableName(), vi)
        end
    end
    if copy_start
        for vi in vis
            mapped = idxmap[vi]
            value = MOI.get(src, MOI.VariablePrimalStart(), vi)
            opt.x0[mapped.value] =
                value === nothing ? nothing : convert(QOCOFloat, value)
        end
        opt.x0_set = any(!isnothing, opt.x0)
    end
    return
end

function _process_objective!(
    opt::Optimizer,
    src::MOI.ModelLike,
    idxmap::MOI.Utilities.IndexMap,
)
    obj_type = MOI.get(src, MOI.ObjectiveFunctionType())
    if obj_type == MOI.VariableIndex
        obj = MOI.Utilities.map_indices(
            idxmap,
            MOI.get(src, MOI.ObjectiveFunction{MOI.VariableIndex}()),
        )
        opt.objective = obj
        opt.c = zeros(QOCOFloat, opt.n)
        opt.c[obj.value] = 1.0
        opt.P_colptr = zeros(QOCOInt, opt.n + 1)
        opt.P_rowval = QOCOInt[]
        opt.P_nzval = QOCOFloat[]
    elseif obj_type == MOI.ScalarAffineFunction{Float64}
        obj = MOI.Utilities.canonical(
            MOI.Utilities.map_indices(
                idxmap,
                MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()),
            ),
        )
        opt.objective = obj
        opt.c = zeros(QOCOFloat, opt.n)
        for term in obj.terms
            opt.c[term.variable.value] += term.coefficient
        end
        opt.P_colptr = zeros(QOCOInt, opt.n + 1)
        opt.P_rowval = QOCOInt[]
        opt.P_nzval = QOCOFloat[]
    elseif obj_type == MOI.ScalarQuadraticFunction{Float64}
        obj = MOI.Utilities.canonical(
            MOI.Utilities.map_indices(
                idxmap,
                MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}()),
            ),
        )
        opt.objective = obj
        opt.c = zeros(QOCOFloat, opt.n)
        for term in obj.affine_terms
            opt.c[term.variable.value] += term.coefficient
        end
        I = QOCOInt[]
        J = QOCOInt[]
        V = QOCOFloat[]
        for term in obj.quadratic_terms
            i = min(term.variable_1.value, term.variable_2.value)
            j = max(term.variable_1.value, term.variable_2.value)
            push!(I, QOCOInt(i))
            push!(J, QOCOInt(j))
            push!(V, QOCOFloat(term.coefficient))
        end
        if isempty(I)
            opt.P_colptr = zeros(QOCOInt, opt.n + 1)
            opt.P_rowval = QOCOInt[]
            opt.P_nzval = QOCOFloat[]
        else
            P = triu(sparse(I, J, V, opt.n, opt.n))
            opt.P_colptr = QOCOInt.(P.colptr .- 1)
            opt.P_rowval = QOCOInt.(P.rowval .- 1)
            opt.P_nzval = QOCOFloat.(P.nzval)
        end
    else
        throw(MOI.UnsupportedAttribute(MOI.ObjectiveFunction{obj_type}()))
    end
    _apply_objective_sense!(opt)
    return
end

function _constraint_attributes_supported!(
    opt::Optimizer,
    src::MOI.ModelLike,
    ::Type{S},
) where {S<:Union{MOI.Zeros, MOI.Nonnegatives, MOI.SecondOrderCone}}
    attrs = MOI.get(src, MOI.ListOfConstraintAttributesSet{_SUPPORTED_CONSTRAINT_FUNCTION, S}())
    has_names = false
    for attr in attrs
        attr == MOI.ConstraintName() || throw(MOI.UnsupportedAttribute(attr))
        has_names = true
    end
    if S == MOI.Zeros
        opt.eq_names_set = has_names
    elseif S == MOI.Nonnegatives
        opt.nn_names_set = has_names
    else
        opt.soc_names_set = has_names
    end
    return has_names
end

function _process_constraints!(
    opt::Optimizer,
    src::MOI.ModelLike,
    idxmap::MOI.Utilities.IndexMap,
    ::Type{MOI.Zeros},
)
    has_names = _constraint_attributes_supported!(opt, src, MOI.Zeros)
    cis = MOI.get(src, MOI.ListOfConstraintIndices{_SUPPORTED_CONSTRAINT_FUNCTION, MOI.Zeros}())
    I = QOCOInt[]
    J = QOCOInt[]
    V = QOCOFloat[]
    row_offset = opt.p
    for ci in cis
        func = MOI.Utilities.canonical(
            MOI.Utilities.map_indices(idxmap, MOI.get(src, MOI.ConstraintFunction(), ci)),
        )
        set = MOI.get(src, MOI.ConstraintSet(), ci)
        dim = set.dimension

        new_id = (opt.next_constraint_id += 1)
        idxmap[ci] = MOI.ConstraintIndex{_SUPPORTED_CONSTRAINT_FUNCTION, MOI.Zeros}(new_id)
        rows = (row_offset + 1):(row_offset + dim)
        push!(opt.eq_constraints, (new_id, rows))
        opt.constraint_offset[new_id] = (:eq, length(opt.eq_constraints))
        push!(opt.eq_functions, func)
        push!(opt.eq_sets, set)
        push!(opt.eq_names, has_names ? MOI.get(src, MOI.ConstraintName(), ci) : "")

        append!(opt.b, QOCOFloat.(-func.constants))
        for term in func.terms
            push!(I, QOCOInt(row_offset + term.output_index))
            push!(J, QOCOInt(term.scalar_term.variable.value))
            push!(V, QOCOFloat(term.scalar_term.coefficient))
        end
        row_offset += dim
    end
    opt.p = row_offset
    _append_coo!(opt, :A, I, J, V)
    return
end

function _process_constraints!(
    opt::Optimizer,
    src::MOI.ModelLike,
    idxmap::MOI.Utilities.IndexMap,
    ::Type{MOI.Nonnegatives},
)
    has_names = _constraint_attributes_supported!(opt, src, MOI.Nonnegatives)
    cis = MOI.get(
        src,
        MOI.ListOfConstraintIndices{_SUPPORTED_CONSTRAINT_FUNCTION, MOI.Nonnegatives}(),
    )
    I = QOCOInt[]
    J = QOCOInt[]
    V = QOCOFloat[]
    row_offset = opt.m
    for ci in cis
        func = MOI.Utilities.canonical(
            MOI.Utilities.map_indices(idxmap, MOI.get(src, MOI.ConstraintFunction(), ci)),
        )
        set = MOI.get(src, MOI.ConstraintSet(), ci)
        dim = set.dimension

        new_id = (opt.next_constraint_id += 1)
        idxmap[ci] = MOI.ConstraintIndex{_SUPPORTED_CONSTRAINT_FUNCTION, MOI.Nonnegatives}(new_id)
        rows = (row_offset + 1):(row_offset + dim)
        push!(opt.nn_constraints, (new_id, rows))
        opt.constraint_offset[new_id] = (:nn, length(opt.nn_constraints))
        push!(opt.nn_functions, func)
        push!(opt.nn_sets, set)
        push!(opt.nn_names, has_names ? MOI.get(src, MOI.ConstraintName(), ci) : "")

        append!(opt.h, QOCOFloat.(func.constants))
        for term in func.terms
            push!(I, QOCOInt(row_offset + term.output_index))
            push!(J, QOCOInt(term.scalar_term.variable.value))
            push!(V, QOCOFloat(-term.scalar_term.coefficient))
        end
        row_offset += dim
        opt.l += dim
    end
    opt.m = row_offset
    _append_coo!(opt, :G, I, J, V)
    return
end

function _process_constraints!(
    opt::Optimizer,
    src::MOI.ModelLike,
    idxmap::MOI.Utilities.IndexMap,
    ::Type{MOI.SecondOrderCone},
)
    has_names = _constraint_attributes_supported!(opt, src, MOI.SecondOrderCone)
    cis = MOI.get(
        src,
        MOI.ListOfConstraintIndices{_SUPPORTED_CONSTRAINT_FUNCTION, MOI.SecondOrderCone}(),
    )
    I = QOCOInt[]
    J = QOCOInt[]
    V = QOCOFloat[]
    row_offset = opt.m
    for ci in cis
        func = MOI.Utilities.canonical(
            MOI.Utilities.map_indices(idxmap, MOI.get(src, MOI.ConstraintFunction(), ci)),
        )
        set = MOI.get(src, MOI.ConstraintSet(), ci)
        dim = set.dimension

        new_id = (opt.next_constraint_id += 1)
        idxmap[ci] = MOI.ConstraintIndex{_SUPPORTED_CONSTRAINT_FUNCTION, MOI.SecondOrderCone}(new_id)
        rows = (row_offset + 1):(row_offset + dim)
        push!(opt.soc_constraints, (new_id, rows))
        opt.constraint_offset[new_id] = (:soc, length(opt.soc_constraints))
        push!(opt.soc_functions, func)
        push!(opt.soc_sets, set)
        push!(opt.soc_names, has_names ? MOI.get(src, MOI.ConstraintName(), ci) : "")

        append!(opt.h, QOCOFloat.(func.constants))
        for term in func.terms
            push!(I, QOCOInt(row_offset + term.output_index))
            push!(J, QOCOInt(term.scalar_term.variable.value))
            push!(V, QOCOFloat(-term.scalar_term.coefficient))
        end
        row_offset += dim
        opt.nsoc += 1
        push!(opt.q, QOCOInt(dim))
    end
    opt.m = row_offset
    _append_coo!(opt, :G, I, J, V)
    return
end

function _finalize_data!(opt::Optimizer)
    if isempty(opt.P_colptr)
        opt.P_colptr = zeros(QOCOInt, opt.n + 1)
    end
    if isempty(opt.A_coo_i)
        opt.A_colptr = zeros(QOCOInt, opt.n + 1)
        opt.A_rowval = QOCOInt[]
        opt.A_nzval = QOCOFloat[]
    else
        A = sparse(opt.A_coo_i, opt.A_coo_j, opt.A_coo_v, opt.p, opt.n)
        opt.A_colptr = QOCOInt.(A.colptr .- 1)
        opt.A_rowval = QOCOInt.(A.rowval .- 1)
        opt.A_nzval = QOCOFloat.(A.nzval)
    end
    if isempty(opt.G_coo_i)
        opt.G_colptr = zeros(QOCOInt, opt.n + 1)
        opt.G_rowval = QOCOInt[]
        opt.G_nzval = QOCOFloat[]
    else
        G = sparse(opt.G_coo_i, opt.G_coo_j, opt.G_coo_v, opt.m, opt.n)
        opt.G_colptr = QOCOInt.(G.colptr .- 1)
        opt.G_rowval = QOCOInt.(G.rowval .- 1)
        opt.G_nzval = QOCOFloat.(G.nzval)
    end
    empty!(opt.A_coo_i)
    empty!(opt.A_coo_j)
    empty!(opt.A_coo_v)
    empty!(opt.G_coo_i)
    empty!(opt.G_coo_j)
    empty!(opt.G_coo_v)
    return
end

function MOI.copy_to(opt::Optimizer, src::MOI.ModelLike)
    MOI.empty!(opt)
    idxmap = MOI.Utilities.IndexMap()

    _copy_model_attributes!(opt, src)

    vis = MOI.get(src, MOI.ListOfVariableIndices())
    opt.n = length(vis)
    opt.variable_names = fill("", opt.n)
    opt.x0 = Union{Nothing, QOCOFloat}[nothing for _ in 1:opt.n]
    for (i, vi) in enumerate(vis)
        idxmap[vi] = MOI.VariableIndex(i)
    end
    _copy_variable_attributes!(opt, src, vis, idxmap)

    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        MOI.supports_constraint(opt, F, S) || throw(MOI.UnsupportedConstraint{F, S}())
    end

    _process_objective!(opt, src, idxmap)
    _process_constraints!(opt, src, idxmap, MOI.Zeros)
    _process_constraints!(opt, src, idxmap, MOI.Nonnegatives)
    _process_constraints!(opt, src, idxmap, MOI.SecondOrderCone)
    _finalize_data!(opt)

    return idxmap
end

function _set_result_status!(
    opt::Optimizer,
    termination::MOI.TerminationStatusCode,
    raw_status::String,
    primal_status::MOI.ResultStatusCode,
    dual_status::MOI.ResultStatusCode,
    result_count::Int,
)
    opt.has_result = true
    opt.termination_status = termination
    opt.raw_status = raw_status
    opt.primal_status = primal_status
    opt.dual_status = dual_status
    opt.result_count = result_count
    return
end

function _setup_error_string(code::QOCOInt)
    if code == QOCO_DATA_VALIDATION_ERROR
        return "QOCO_DATA_VALIDATION_ERROR"
    elseif code == QOCO_SETTINGS_VALIDATION_ERROR
        return "QOCO_SETTINGS_VALIDATION_ERROR"
    elseif code == QOCO_SETUP_ERROR
        return "QOCO_SETUP_ERROR"
    elseif code == QOCO_AMD_ERROR
        return "QOCO_AMD_ERROR"
    elseif code == QOCO_MALLOC_ERROR
        return "QOCO_MALLOC_ERROR"
    end
    return "QOCO_UNKNOWN_SETUP_ERROR"
end

function _set_setup_error!(opt::Optimizer, code::QOCOInt)
    status = code == QOCO_DATA_VALIDATION_ERROR ? MOI.INVALID_MODEL : MOI.OTHER_ERROR
    opt.solve_status = QOCO_UNSOLVED
    _set_result_status!(opt, status, _setup_error_string(code), MOI.NO_SOLUTION, MOI.NO_SOLUTION, 0)
    opt.primal = Float64[]
    opt.dual_eq = Float64[]
    opt.dual_cone = Float64[]
    opt.slack_cone = Float64[]
    opt.obj_val = NaN
    opt.dual_obj_val = NaN
    opt.solve_time = 0.0
    opt.setup_time = 0.0
    opt.analysis_time = 0.0
    opt.iterations = 0
    opt.ir_iterations = 0
    opt.pres = NaN
    opt.dres = NaN
    opt.gap = NaN
    return
end

function _set_qoco_status!(opt::Optimizer, status::QOCOInt)
    opt.solve_status = status
    if status == QOCO_SOLVED
        _set_result_status!(opt, MOI.OPTIMAL, "QOCO_SOLVED", MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT, 1)
    elseif status == QOCO_SOLVED_INACCURATE
        _set_result_status!(
            opt,
            MOI.ALMOST_OPTIMAL,
            "QOCO_SOLVED_INACCURATE",
            MOI.NEARLY_FEASIBLE_POINT,
            MOI.NEARLY_FEASIBLE_POINT,
            1,
        )
    elseif status == QOCO_MAX_ITER
        _set_result_status!(
            opt,
            MOI.ITERATION_LIMIT,
            "QOCO_MAX_ITER",
            MOI.NEARLY_FEASIBLE_POINT,
            MOI.NEARLY_FEASIBLE_POINT,
            1,
        )
    elseif status == QOCO_NUMERICAL_ERROR
        _set_result_status!(
            opt,
            MOI.NUMERICAL_ERROR,
            "QOCO_NUMERICAL_ERROR",
            MOI.NO_SOLUTION,
            MOI.NO_SOLUTION,
            0,
        )
    else
        _set_result_status!(opt, MOI.OTHER_ERROR, "QOCO_UNSOLVED", MOI.NO_SOLUTION, MOI.NO_SOLUTION, 0)
    end
    return
end

function _dual_objective_value(opt::Optimizer)
    dual_obj = _objective_constant(opt.objective)
    sign = opt.sense == MOI.MAX_SENSE ? 1.0 : -1.0
    for i in eachindex(opt.eq_functions)
        _, rows = opt.eq_constraints[i]
        dual_obj += sign * dot(opt.eq_functions[i].constants, -opt.dual_eq[rows])
    end
    for i in eachindex(opt.nn_functions)
        _, rows = opt.nn_constraints[i]
        dual_obj += sign * dot(opt.nn_functions[i].constants, opt.dual_cone[rows])
    end
    for i in eachindex(opt.soc_functions)
        _, rows = opt.soc_constraints[i]
        dual_obj += sign * dot(opt.soc_functions[i].constants, opt.dual_cone[rows])
    end
    return dual_obj
end

const _CONSTANT_MODEL_TOL = 1e-9

function _constant_model_feasible(opt::Optimizer)
    for func in opt.eq_functions
        any(value -> abs(value) > _CONSTANT_MODEL_TOL, func.constants) && return false
    end
    for func in opt.nn_functions
        any(value -> value < -_CONSTANT_MODEL_TOL, func.constants) && return false
    end
    for func in opt.soc_functions
        rhs = length(func.constants) == 1 ? 0.0 : norm(@view func.constants[2:end])
        func.constants[1] < rhs - _CONSTANT_MODEL_TOL && return false
    end
    return true
end

function _solve_constant_model!(opt::Optimizer)
    feasible = _constant_model_feasible(opt)
    opt.solve_time = 0.0
    opt.setup_time = 0.0
    opt.analysis_time = 0.0
    opt.iterations = 0
    opt.ir_iterations = 0
    opt.pres = 0.0
    opt.dres = 0.0
    opt.gap = 0.0
    if feasible
        opt.solve_status = QOCO_SOLVED
        _set_result_status!(
            opt,
            MOI.OPTIMAL,
            "CONSTANT_MODEL_FEASIBLE",
            MOI.FEASIBLE_POINT,
            MOI.FEASIBLE_POINT,
            1,
        )
        opt.primal = Float64[]
        opt.dual_eq = zeros(Float64, opt.p)
        opt.dual_cone = zeros(Float64, opt.m)
        opt.slack_cone = copy(opt.h)
        opt.obj_val = _evaluate_objective(opt.objective, opt.primal)
        opt.dual_obj_val = _dual_objective_value(opt)
    else
        opt.solve_status = QOCO_UNSOLVED
        _set_result_status!(
            opt,
            MOI.INFEASIBLE,
            "CONSTANT_MODEL_INFEASIBLE",
            MOI.NO_SOLUTION,
            MOI.NO_SOLUTION,
            0,
        )
        opt.primal = Float64[]
        opt.dual_eq = Float64[]
        opt.dual_cone = Float64[]
        opt.slack_cone = Float64[]
        opt.obj_val = NaN
        opt.dual_obj_val = NaN
    end
    return
end

function _store_solution!(opt::Optimizer, sol::QOCOSolution)
    opt.primal = opt.n > 0 ? copy(unsafe_wrap(Array, sol.x, opt.n)) : Float64[]
    opt.dual_eq = opt.p > 0 ? copy(unsafe_wrap(Array, sol.y, opt.p)) : Float64[]
    opt.dual_cone = opt.m > 0 ? copy(unsafe_wrap(Array, sol.z, opt.m)) : Float64[]
    opt.slack_cone = opt.m > 0 ? copy(unsafe_wrap(Array, sol.s, opt.m)) : Float64[]
    opt.solve_time = sol.solve_time_sec
    opt.setup_time = sol.setup_time_sec
    opt.analysis_time = sol.analysis_time_sec
    opt.iterations = Int(sol.iters)
    opt.ir_iterations = Int(sol.ir_iters)
    opt.pres = sol.pres
    opt.dres = sol.dres
    opt.gap = sol.gap
    _set_qoco_status!(opt, sol.status)
    if opt.result_count > 0
        opt.obj_val = _evaluate_objective(opt.objective, opt.primal)
        opt.dual_obj_val = _dual_objective_value(opt)
    else
        opt.obj_val = NaN
        opt.dual_obj_val = NaN
    end
    return
end

function MOI.optimize!(opt::Optimizer, src::MOI.ModelLike)
    index_map = MOI.copy_to(opt, src)
    MOI.optimize!(opt)
    return index_map, false
end

function MOI.optimize!(opt::Optimizer)
    _reset_results!(opt)

    if opt.n == 0
        _solve_constant_model!(opt)
        return
    end

    settings = _build_settings(opt)

    P_csc = QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
    A_csc = QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)
    G_csc = QOCOCscMatrix(0, 0, 0, C_NULL, C_NULL, C_NULL)

    P_values = isempty(opt.P_nzval) ? QOCOFloat[0.0] : opt.P_nzval
    P_rows = isempty(opt.P_rowval) ? QOCOInt[0] : opt.P_rowval
    qoco_set_csc!(P_csc, opt.n, opt.n, length(opt.P_nzval), P_values, opt.P_colptr, P_rows)

    A_values = isempty(opt.A_nzval) ? QOCOFloat[0.0] : opt.A_nzval
    A_rows = isempty(opt.A_rowval) ? QOCOInt[0] : opt.A_rowval
    qoco_set_csc!(A_csc, opt.p, opt.n, length(opt.A_nzval), A_values, opt.A_colptr, A_rows)

    G_values = isempty(opt.G_nzval) ? QOCOFloat[0.0] : opt.G_nzval
    G_rows = isempty(opt.G_rowval) ? QOCOInt[0] : opt.G_rowval
    qoco_set_csc!(G_csc, opt.m, opt.n, length(opt.G_nzval), G_values, opt.G_colptr, G_rows)

    b_vec = isempty(opt.b) ? QOCOFloat[0.0] : opt.b
    h_vec = isempty(opt.h) ? QOCOFloat[0.0] : opt.h
    q_vec = isempty(opt.q) ? QOCOInt[0] : opt.q

    solver_ptr = qoco_solver_alloc()
    # The CSC structs hold bare pointers into these arrays, so they are not
    # reachable from the `qoco_setup` call itself and must be rooted explicitly
    # for its duration.
    err = GC.@preserve P_values P_rows A_values A_rows G_values G_rows qoco_setup!(
        solver_ptr,
        opt.n,
        opt.m,
        opt.p,
        P_csc,
        opt.c,
        A_csc,
        b_vec,
        G_csc,
        h_vec,
        opt.l,
        opt.nsoc,
        q_vec,
        settings,
    )
    if err != QOCO_NO_ERROR
        _set_setup_error!(opt, err)
        Libc.free(solver_ptr)
        return
    end

    try
        x0 = _primal_start(opt)
        if x0 !== nothing
            qoco_set_x0!(solver_ptr, x0)
        end
        qoco_solve!(solver_ptr)
        _store_solution!(opt, get_solution(solver_ptr))
    finally
        qoco_cleanup!(solver_ptr)
    end
    return
end

function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    return opt.termination_status
end

function MOI.get(opt::Optimizer, ::MOI.RawStatusString)
    return opt.raw_status
end

function MOI.get(opt::Optimizer, attr::MOI.PrimalStatus)
    if !opt.has_result || attr.result_index != 1 || opt.result_count == 0
        return MOI.NO_SOLUTION
    end
    return opt.primal_status
end

function MOI.get(opt::Optimizer, attr::MOI.DualStatus)
    if !opt.has_result || attr.result_index != 1 || opt.result_count == 0
        return MOI.NO_SOLUTION
    end
    return opt.dual_status
end

function MOI.get(opt::Optimizer, ::MOI.ResultCount)
    return opt.result_count
end

function MOI.get(opt::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(opt, attr)
    return opt.obj_val
end

function MOI.get(opt::Optimizer, attr::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(opt, attr)
    return opt.dual_obj_val
end

function MOI.get(opt::Optimizer, ::MOI.SolveTimeSec)
    if !opt.has_result
        return 0.0
    end
    return opt.solve_time + opt.setup_time
end

function MOI.get(opt::Optimizer, ::MOI.BarrierIterations)
    return Int64(opt.iterations)
end

"""
    QOCO.AnalysisTimeSec()

Model attribute for the time, in seconds, that the linear system backend spent
in its symbolic analysis phase. Reported by QOCO 0.3 and later; it is part of
the setup time rather than additional to it.
"""
struct AnalysisTimeSec <: MOI.AbstractModelAttribute end

MOI.is_set_by_optimize(::AnalysisTimeSec) = true

function MOI.get(opt::Optimizer, ::AnalysisTimeSec)
    return opt.has_result ? opt.analysis_time : 0.0
end

"""
    QOCO.IterativeRefinementIterations()

Model attribute for the total number of iterative refinement iterations taken
across all interior point steps. Reported by QOCO 0.3 and later.
"""
struct IterativeRefinementIterations <: MOI.AbstractModelAttribute end

MOI.is_set_by_optimize(::IterativeRefinementIterations) = true

function MOI.get(opt::Optimizer, ::IterativeRefinementIterations)
    return Int64(opt.ir_iterations)
end

function MOI.get(opt::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    _check_valid(opt, vi)
    MOI.check_result_index_bounds(opt, attr)
    return opt.primal[vi.value]
end

function MOI.get(
    opt::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros},
)
    _check_valid(opt, ci)
    MOI.check_result_index_bounds(opt, attr)
    idx = _constraint_position(opt, ci, :eq)
    return _evaluate_vector_affine(opt.eq_functions[idx], opt.primal)
end

function MOI.get(
    opt::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives},
)
    _check_valid(opt, ci)
    MOI.check_result_index_bounds(opt, attr)
    idx = _constraint_position(opt, ci, :nn)
    return _evaluate_vector_affine(opt.nn_functions[idx], opt.primal)
end

function MOI.get(
    opt::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.SecondOrderCone},
)
    _check_valid(opt, ci)
    MOI.check_result_index_bounds(opt, attr)
    idx = _constraint_position(opt, ci, :soc)
    return _evaluate_vector_affine(opt.soc_functions[idx], opt.primal)
end

function MOI.get(
    opt::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros},
)
    _check_valid(opt, ci)
    MOI.check_result_index_bounds(opt, attr)
    idx = _constraint_position(opt, ci, :eq)
    _, rows = opt.eq_constraints[idx]
    return -opt.dual_eq[rows]
end

function MOI.get(
    opt::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives},
)
    _check_valid(opt, ci)
    MOI.check_result_index_bounds(opt, attr)
    idx = _constraint_position(opt, ci, :nn)
    _, rows = opt.nn_constraints[idx]
    return opt.dual_cone[rows]
end

function MOI.get(
    opt::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.SecondOrderCone},
)
    _check_valid(opt, ci)
    MOI.check_result_index_bounds(opt, attr)
    idx = _constraint_position(opt, ci, :soc)
    _, rows = opt.soc_constraints[idx]
    return opt.dual_cone[rows]
end
