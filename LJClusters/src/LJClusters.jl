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
MosimoBase.name(model::LJCluster) = "LJCluster-$(is_3d(model) ? "3D" : "2D")"
MosimoBase.natoms(model::LJCluster) = model.N

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

function _∇V(rs, ϵ, σ, dist, r_cutoff, is_force, ∇V = similar(rs))
    N = length(rs)
    fij = if r_cutoff > 0
        (ri, rj) -> LennardJones.lj_potential_fij_cutoff(ri, rj; ϵ, σ, dist, r_cutoff)
    else
        (ri, rj) -> LennardJones.lj_potential_fij(ri, rj; ϵ, σ, dist)
    end
    fill!(∇V, zero(eltype(rs)))
    @inbounds for i in 1:N-1
        for j in i+1:N
            f_ij = fij(rs[i], rs[j])
            ∇V[i] += is_force ? f_ij : -f_ij
            ∇V[j] += is_force ? -f_ij : f_ij
        end
    end
    ∇V
end
function _∇Vi(rs, i, ϵ, σ, dist, r_cutoff)
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
MosimoBase.force_function(model::LJCluster{T}, rs::AbstractArray{T}; inplace=similar(rs)) where {T} = _∇V(
    rs, model.ϵ, model.σ, distance_function(model), model.r_cutoff, true, inplace
)
MosimoBase.force_function(model::LJCluster{T}, rs::AbstractArray{T}, i) where {T} = -_∇Vi(
    rs, i, model.ϵ, model.σ, distance_function(model), model.r_cutoff
)
MosimoBase.potential_energy_gradients(model::LJCluster{T}, rs::AbstractArray{T}; inplace=similar(rs)) where {T} = _∇V(
    rs, model.ϵ, model.σ, distance_function(model), model.r_cutoff, false, inplace
)
MosimoBase.kinetic_energy(s::ConfigurationSystem{T}, ::LJCluster{T}) where T = 0.0
function MosimoBase.kinetic_energy(s::MosiSystem{T}, ::LJCluster{T}) where T <: MosiVector
    vs = velocities(s)
    sum(norm_sqr.(vs)) / 2
end
MosimoBase.degree_of_freedom(_::MosiSystem, model::LJCluster) = (is_3d(model) ? 3 : 2) * natoms(model)

# Use a simple packing.
function MosimoBase.generate_initial(
    model::LJCluster{T}, ::Type{ConfigurationSystem};
    rng::AbstractRNG=GLOBAL_RNG
) where T
    N = model.N
    box = model.box
    tmp_box = if box ≡ zero(T)
        L = model.σ * MosimoBase.ceil_root(N, length(T))
        ones(T) * L
    else
        box
    end
    rs = collect(simple_pack(tmp_box, N))
    ConfigurationSystem(rs; box)
end

end # module
