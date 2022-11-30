"""
A simple Lennard-Jones cluster model.
"""
module LJClusters

using MosimoBase
using LennardJones

export LJCluster, LJCluster2D, LJCluster3D

Base.@kwdef struct LJCluster{T} <: MosiModel{T}
    N::Int
    ϵ::Float64 = 1.0
    σ::Float64 = 1.0
    # If ≤ 0, no cutoff
    r_cutoff::Float64 = 2.5σ
    box::T = zero(T)
end
LJCluster2D(N; kwargs...) = LJCluster{Vector2}(; N, kwargs...)
LJCluster3D(N; kwargs...) = LJCluster{Vector3}(; N, kwargs...)

function _V(rs, ϵ, σ, dist, r_cutoff)
    N = length(rs)
    uij = if r_cutoff > 0
        (ri, rj) -> LennardJones.lj_potential_uij_cutoff(ri, rj; ϵ, σ, dist, r_cutoff)
    else
        (ri, rj) -> LennardJones.lj_potential_uij(ri, rj; ϵ, σ, dist)
    end
    V = 0.0
    @inbounds for i in 1:N-1
        ri = rs[i]
        for j in i+1:N
            V += uij(ri, rs[j])
        end
    end
    V
end

function _∇V(rs, ϵ, σ, dist, r_cutoff)
    N = length(rs)
    fij = if r_cutoff > 0
        (ri, rj) -> LennardJones.lj_potential_fij_cutoff(ri, rj; ϵ, σ, dist, r_cutoff)
    else
        (ri, rj) -> LennardJones.lj_potential_fij(ri, rj; ϵ, σ, dist)
    end
    ∇V = zeros(eltype(rs), N)
    @inbounds for i in 1:N-1
        for j in i+1:N
            f_ij = fij(i, j)
            ∇V[i] -= f_ij
            ∇V[j] += f_ij
        end
    end
    ∇V
end
function _∇V(rs, i, ϵ, σ, dist, r_cutoff)
    N = length(rs)
    fij = if r_cutoff > 0
        (ri, rj) -> LennardJones.lj_potential_fij_cutoff(ri, rj; ϵ, σ, dist, r_cutoff)
    else
        (ri, rj) -> LennardJones.lj_potential_fij(ri, rj; ϵ, σ, dist)
    end
    ∇V = zero(eltype(rs))
    ri = rs[i]
    @inbounds for j in 1:N
        i ≡ j && continue
        ∇V -= fij(ri, rs[j])
    end
    ∇V
end

MosimoBase.potential_energy_function(model::LJCluster{T}, rs::AbstractArray{T}) where {T} = _V(
    rs, model.ϵ, model.σ, distance_function(model), model.r_cutoff
)
MosimoBase.force_function(model::LJCluster{T}, rs::AbstractArray{T}) where {T} = _∇V(
    rs, model.ϵ, model.σ, distance_function(model), model.r_cutoff
)
MosimoBase.force_function(model::LJCluster{T}, rs::AbstractArray{T}, i) where {T} = _∇V(
    rs, i, model.ϵ, model.σ, distance_function(model), model.r_cutoff
)

end # module