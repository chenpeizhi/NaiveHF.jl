using LinearAlgebra
using NLsolve
import TensorOperations: @tensor

export scf!


function buildfock!(fock::Matrix{Float64}, ham::Hamiltonian, dm::Matrix{Float64})

    v = ham.v
    @tensor fock[p, q] = v[p, q, r, s] * dm[r, s] - v[p, r, q, s] * dm[r, s]
    fock .+= ham.h

    return fock
end


"""
Neel initial guess for 2D Hubbard, assuming both `nx` and `ny` are either even or 1.
"""
function neel_init(ham::Hubbard, nelec::Int)

    @assert ham.nx % 2 == 0 || ham.nx == 1
    @assert ham.ny % 2 == 0 || ham.ny == 1
    nmo = ham.nx * ham.ny
    nso = 2nmo
    occ = nelec / nso
    
    tmp = zeros(nso)
    @views if nelec <= nmo
        # I interlace the alpha and beta AOs in this simple GHF code.
        tmp[1:4:end] .= occ
        tmp[4:4:end] .= occ
    else
        tmp[1:4:end] .= 1.0
        tmp[4:4:end] .= 1.0
        occ -= 1.0
        tmp[3:4:end] .= occ
        tmp[2:4:end] .= occ
    end
    
    return diagm(tmp)
end


function scf!(hf::HF, ham::Hamiltonian;
              dm0=nothing, iterations::Int=100, dmconv=1e-8,
              damping::Float64=0.0, diis_dim::Int=8, diis_start_cycle::Int=1)

    if dm0 === nothing
        if hf.mo_coeff === nothing
            dm0 = neel_init(ham, hf.nelec)
        else
            mo_occ = @view hf.mo_coeff[:, 1:hf.nelec]
            dm0 = mo_occ * mo_occ'
        end
    end

    nso = size(ham.h, 1)
    fock = Matrix{Float64}(undef, nso, nso)

    function residual!(res::Matrix{Float64}, dm::Matrix{Float64})
        buildfock!(fock, ham, dm)
        mo_energy, mo_coeff = eigen!(Hermitian(fock))
        hf.mo_energy = mo_energy
        hf.mo_coeff = mo_coeff

        mo_occ = @view mo_coeff[:, 1:hf.nelec]
        mul!(res, mo_occ, mo_occ')

        @views energy = 0.5 * (res[:]' * ham.h[:] + sum(mo_energy[1:hf.nelec]))
        hf.delta_energy = energy - hf.energy
        hf.energy = energy
        println("\nE = ", energy, "  dE = ", hf.delta_energy)

        res .-= dm
        return res
    end

    result = nlsolve(residual!, dm0;
                     method=:anderson, iterations=iterations, ftol=dmconv,
                     beta=1.0-damping, m=diis_dim, aa_start=diis_start_cycle,
                     show_trace=true)

    if result.f_converged
        println("\nSCF has converged.")
    else
        println("\nSCF did not converge.")
    end
    
    return result
end

