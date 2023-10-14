# using Plots

# loc = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\TrackVersionsProgramsGit\\1-GeneracionDeDatos\\"

# x = collect(-5:0.1:5)
# y = sinc.(x)


# pl = plot(x, y, xlabel = "x", ylabel = "sinc(x)" , label="sinc(x)", lw=2, tickfontsize=12, labelfontsize=15, legendfontsize=15, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)

# savefig(pl, loc*"sinc.pdf")

using PlotlyJS, CSV, DataFrames
df = dataset(DataFrame, "iris")
plot(
    df,
    x=:sepal_length, y=:sepal_width, z=:petal_width, color=:species,
    type="scatter3d", mode="markers"
)
