export HF


mutable struct HF
    nelec::Int
    # Assuming aufbau
    mo_coeff::Union{Matrix{Float64}, Nothing}
    mo_energy::Union{Vector{Float64}, Nothing}
    energy::Float64
    delta_energy::Float64
    # delta_dm::Float64
end

HF(nelec::Int) = HF(nelec, nothing, nothing, NaN, NaN)

