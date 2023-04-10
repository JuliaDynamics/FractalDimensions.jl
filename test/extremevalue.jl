using Statistics, DynamicalSystems, CairoMakie

@inline @inbounds function lorenz_rule(u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du1 = σ*(u[2]-u[1])
    du2 = u[1]*(ρ-u[3]) - u[2]
    du3 = u[1]*u[2] - β*u[3]
    return SVector{3}(du1, du2, du3)
end

ds = CoupledODEs(lorenz_rule, [0, 10, 0.0], [10.0, 28.0, 8/3])

tr, tvec = trajectory(ds, 50000; Δt = 1.0, Ttr = 100)

qs = [0.95, 0.98, 0.99, 0.999]

fig, axs = subplotgrid(2,2; sharex = true, sharey = true, xlabels = "X", ylabels = "Z")

for (q, ax) in zip(qs, axs)

    Δloc, θ = extremevaltheory_dims_persistences(tr, q; compute_persistence = false)
    avedim = mean(Δloc)
    ax.title = "q = $(q), mean D = $(round(avedim; digits = 4))"

    scatter!(ax, tr[:,1], tr[:,3];
        color = Δloc, colormap = :viridis, colorrange = (1,3), markersize = 5
    )
end

Colorbar(fig[:, 3], colormap = :viridis, colorrange = (1, 3))
fig

Makie.save(desktop("Lorenz63_EVT_fractaldim_julia.png"), fig)