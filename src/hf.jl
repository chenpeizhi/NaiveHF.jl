export HF


mutable struct HF
    nelec::Int
    energy::Float64
    mo_energy::Union{Vector{Float64}, Nothing}
    # Assuming aufbau
    mo_coeff::Union{Matrix{Float64}, Nothing}
    delta_energy::Float64
    # delta_dm::Float64
end

HF(nelec::Int) = HF(nelec, NaN, nothing, nothing, NaN)

