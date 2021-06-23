# Multivariate Lazy Kernel Density Estimator ====================================================================
# Acknowledgement: Code structure refers to Nonparametric part of ['statsmodels'](https://github.com/statsmodels/statsmodels)
# 
# Copyright of statsmodels: 
# 
#     Copyright (C) 2006, Jonathan E. Taylor
#     All rights reserved.
#     
#     Copyright (c) 2006-2008 Scipy Developers.
#     All rights reserved.
#     
#     Copyright (c) 2009-2018 statsmodels Developers.
#     All rights reserved.


# KDE kernels
# @Ping: More Efficient way is we can pass kde as argument to avoid repeated calculation (like level), 
# But I didn't do that because I wish to make all kernel functions irrelavent to KDE objects, for easier refractor in the future. 
# I'm planing to integrate this file into KernelDensity.jl some day.

# unordered categorical default KDE='aitchisonaitken'
function aitchison_aitken(bandwidth::Real, observations::RealVector, x::Real)
    num_levels = length(unique(observations))
    kernel_value = ones(length(observations)) * bandwidth / (num_levels - 1)
    idx = observations .== x
    kernel_value[idx] .= (idx * (1 - bandwidth))[idx]
    return kernel_value
end

# ordered categorical default KDE='wangryzin'
function wang_ryzin(bandwidth::Real, observations::RealVector, x::Real)
    kernel_value = 0.5 * (1 - bandwidth) * (bandwidth .^ abs.(observations .- x))
    idx = observations .== x
    kernel_value[idx] = (idx * (1-bandwidth))[idx]
    return kernel_value
end

# continuous default KDE='gaussian'
function gaussian(bandwidth::Real, observations::RealVector, x::Real)
    (1 / sqrt(2*π)) * exp.(-(observations.-x).^2 / (bandwidth^2*2))
end

# default kernel function of difference types
type_kernel = Dict(Continuous => gaussian, Categorical => wang_ryzin, UnorderedCategorical => aitchison_aitken)

function create_unordered_map(candidates::Vector)
    unordered_to_index, index_to_unordered = Dict(), Dict()
    for (i, elem) in enumerate(candidates)
        unordered_to_index[elem] = i
        index_to_unordered[i] = elem
    end
    (unordered_to_index, index_to_unordered)
end

# Lazy evaluation KDE
Base.@kwdef mutable struct LazyKDE
    data::RealVector
    bandwidth::Real
    type::DimensionType
    kernel::Function
    is_number::Bool
end

function pdf(kde::LazyKDE, x::Real; keep_all=true)
    densities = kde.kernel(kde.bandwidth, kde.data, x)
    if keep_all
        return densities
    else
        return mean(densities) / kde.bandwidth
    end
end

# Multivariate KDE based on LazyKDE
Base.@kwdef mutable struct MultivariateKDE
    # Type of every dimension, continuous or discrete
    dims::Vector{DimensionType}
    # KDE from different dimensions
    KDEs::Vector{LazyKDE}
    # observations: An k*n matrix, where k is dimension of KDEs and n is number of observations
    observations::RealVectorVector
    mat_observations::Matrix
    # In such case, we assign an index 1:N to every value, where N=|D_i|, then we pass the 1:N to KDE
    unordered_to_index::Dict{LazyKDE, Dict{Any, Real}}
    index_to_unordered::Dict{LazyKDE, Dict{Real, Any}}
end
# Constructor without candidates
function MultivariateKDE(dims::Vector{DimensionType}, observations::Vector{Vector{T}} where T)
    MultivariateKDE(dims, nothing, convert(Vector{Vector}, observations))
end
function MultivariateKDE(dims::Vector{DimensionType}, observations::Vector{Vector})
    MultivariateKDE(dims, nothing, observations)
end
function MultivariateKDE(dims::Vector{DimensionType}, bws::Union{Vector{T} where T<:Real, Nothing}, observations::Vector{Vector{T}} where T)
    MultivariateKDE(dims, bws, convert(Vector{Vector}, observations))
end
function MultivariateKDE(dims::Vector{DimensionType}, bws::Union{Vector{T} where T<:Real, Nothing}, observations::Vector{Vector})
    for (dim_type, observation) in zip(dims, observations)
        if dim_type isa UnorderedCategorical && !(observation[1] isa Real)
            error("If there is Uncatogorical dimension and not a number, should specify its candidate value.")
        end
    end
    MultivariateKDE(dims, bws, observations, nothing)
end
# Constructor with candidates
function MultivariateKDE(dims::Vector{DimensionType}, observations::Vector{Vector}, candidates::Union{Dict{Int, Vector}, Nothing})
    MultivariateKDE(dims, nothing, observations, candidates)
end
function MultivariateKDE(dims::Vector{DimensionType}, bws::Union{Vector{T} where T<:Real, Nothing}, observations::Vector{Vector}, candidates::Union{Dict{Int, Vector}, Nothing})
    mat_observations = hcat(observations...)
    MultivariateKDE(dims, bws, mat_observations, candidates)
end
function MultivariateKDE(dims::Vector{DimensionType}, bws::Union{Vector{T} where T<:Real, Nothing}, observations::Vector{Vector{T}} where T, candidates::Union{Dict{Int, Vector}, Nothing})
    MultivariateKDE(dims, bws, convert(Vector{Vector}, observations), candidates)
end
# Constructor with default bandwidth
function MultivariateKDE(dims::Vector{DimensionType}, bws::Union{Vector{T} where T<:Real, Nothing}, mat_observations::Matrix, candidates::Union{Dict{Int, Vector}, Nothing})
    _KDEs, _observations, _unordered_to_index, _index_to_unordered = Vector{LazyKDE}(), RealVectorVector(), Dict{Int, Dict{Any, Real}}(), 
                                                                            Dict{Int, Dict{Real, Any}}()
    is_numbers = Vector{Bool}()
    for (i, dim_type_i) in zip(1:size(mat_observations)[1], dims)
        _observations_i = mat_observations[i, :]
        if (dim_type_i isa UnorderedCategorical && !(_observations_i[1] isa Real))
            _unordered_to_index_i, _index_to_unordered_i = create_unordered_map(candidates[i])
            _unordered_to_index[i], _index_to_unordered[i] = _unordered_to_index_i, _index_to_unordered_i
            _observations_i = [_unordered_to_index_i[elem] for elem in _observations_i]
            push!(is_numbers, false)
        else
            push!(is_numbers, true)
        end
        push!(_observations, _observations_i)
    end
    if bws isa Nothing
        bws = default_bandwidth(_observations)
    end
    for (_observations_i, bandwidth_i, dim_type_i, is_number) in zip(_observations, bws, dims, is_numbers)
        kde_i = LazyKDE(_observations_i, bandwidth_i, dim_type_i, type_kernel[typeof(dim_type_i)], is_number)
        push!(_KDEs, kde_i)
    end
    unordered_to_index, index_to_unordered = Dict{LazyKDE, Dict{Any, Real}}(), Dict{LazyKDE, Dict{Real, Any}}()
    for i in 1:size(mat_observations)[1]
        if haskey(_index_to_unordered, i)
            unordered_to_index[_KDEs[i]] = _unordered_to_index[i]
            index_to_unordered[_KDEs[i]] = _index_to_unordered[i]
        end
    end
    MultivariateKDE(dims, _KDEs, _observations, mat_observations, unordered_to_index, index_to_unordered)
end

# pdf of MultivariateKDE, using GPKE(Generalized Product Kernel Estimator)
function gpke(multi_kde::MultivariateKDE, x::Vector)
    Kval = Array{Real}(undef, (length(multi_kde.observations[1]), length(multi_kde.observations)))
    for (i, _kde, _x) in zip(1:length(x), multi_kde.KDEs, x)
        # println(_kde.is_number)
        if !_kde.is_number
            _x = multi_kde.unordered_to_index[_kde][_x]
        end
        Kval[:, i] = pdf(_kde, _x)
    end
    iscontinuous = [_dim isa Continuous ? true : false for _dim in multi_kde.dims]
    dens = prod(Kval, dims=2) / prod([kde.bandwidth for kde in multi_kde.KDEs][iscontinuous])
    sum(dens)
end

# Alias of gpke
function pdf(multi_kde::MultivariateKDE, x::Vector)
    gpke(multi_kde, x)
end

# Scott's normal reference rule of thumb bandwidth parameter
function default_bandwidth(observations::RealVectorVector)
    X = std.(observations)
    1.06 * X * length(observations).^(-1 / (4 + length(observations[1])))
end