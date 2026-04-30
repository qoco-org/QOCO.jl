module QOCO

export Optimizer

using QOCO_jll
using SparseArrays
using LinearAlgebra

const libqoco = QOCO_jll.qoco

include("c_api.jl")
include("MOI_wrapper/MOI_wrapper.jl")

end # module QOCO
