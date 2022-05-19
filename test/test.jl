using NaiveHF

nelec = 4
nx = 6
ny = 1
U = 2.0

ham = Hubbard(nx, ny, U; periodic=false)
hf = HF(nelec)
r = scf!(hf, ham)
display(r)

