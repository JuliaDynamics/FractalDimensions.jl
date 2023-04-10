using Statistics, DynamicalSystems, CairoMakie

@inline @inbounds function lorenz_rule(u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du1 = σ*(u[2]-u[1])
    du2 = u[1]*(ρ-u[3]) - u[2]
    du3 = u[1]*u[2] - β*u[3]
    return SVector{3}(du1, du2, du3)
end

ρ = 28.0
ds = CoupledODEs(lorenz_rule, [0, 10, 0.0], [10.0, ρ, 8/3])

tr, tvec = trajectory(ds, 1000; Δt = 0.02, Ttr = 100)
estimator = :mm
qs = [0.97, 0.98, 0.99, 0.995]
N = length(tr)

fig, axs = subplotgrid(2,2; sharex = true, sharey = true, xlabels = "X", ylabels = "Z")

for (q, ax) in zip(qs, axs)

    Δloc, θ = extremevaltheory_dims_persistences(tr, q;
        compute_persistence = false, estimator
    )
    avedim = mean(Δloc)
    ax.title = "q = $(q), mean D = $(round(avedim; digits = 4))"

    scatter!(ax, tr[:,1], tr[:,3];
        color = Δloc, colormap = :viridis, colorrange = (0,3), markersize = 5
    )
end

Colorbar(fig[:, 3], colormap = :viridis, colorrange = (0, 3))
figuretitle!(fig, "Lorenz63 (ρ = $(ρ)) - Local EVT fractal dim. - Estimator: $(estimator)")
display(fig)

Makie.save(desktop("Lorenz63_ρ=$(ρ)_N=$(N)_EVTDIM_$(estimator).png"), fig)