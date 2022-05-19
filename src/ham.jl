export Hamiltonian, Hubbard


abstract type Hamiltonian end

struct Hubbard <: Hamiltonian
    h::Matrix{Float64}
    # Two-body electron integrals in Mulliken notation without antisymmetrization
    v::Array{Float64, 4}
    nx::Int
    ny::Int
end

function Hubbard(nx::Int, ny::Int, U::Float64; periodic::Bool=true)
    
    nmo = nx * ny
    nso = 2nmo

    hao = zeros(ComplexF64, 2nmo, 2nmo)
    if nmo == 1
        return hao
    end

    for ix in 1:nx
        for iy in 1:ny
            ind = (ix - 1) * ny + iy

            if ix != nx
                ind1 = ix * ny + iy
                hao[ind, ind1] = hao[ind1, ind] = -1.0
            end

            if iy != ny
                ind2 = ind + 1
                hao[ind, ind2] = hao[ind2, ind] = -1.0
            end
        end
    end

    if periodic
        if nx > 1
            for iy in 1:ny
                ind1 = iy
                ind2 = (nx - 1) * ny + iy
                hao[ind1, ind2] = hao[ind2, ind1] = -1.0
            end
        end

        if ny > 1
            for ix in 1:nx
                ind1 = (ix - 1) * ny + 1
                ind2 = ix * ny
                hao[ind1, ind2] = hao[ind2, ind1] = -1.0
            end
        end
    end

    hao[nmo+1:end, nmo+1:end] .= hao[1:nmo, 1:nmo]

    vao = zeros(nso, nso, nso, nso)

    for p in 1:nmo
        pbar = p + nmo
        vao[p, p, pbar, pbar] = U
        vao[pbar, pbar, p, p] = U
    end

    return Hubbard(hao, vao, nx, ny)
end

