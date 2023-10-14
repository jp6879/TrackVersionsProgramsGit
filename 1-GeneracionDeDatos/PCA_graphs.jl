using Plots
using StatsPlots
using Distributions
using Random
using LaTeXStrings
using Statistics
using MultivariateStats
using DataFrames
using CSV
using MLJ
using ColorTypes
using PlotlyJS

# Constantes útiles
γ = 2.675e8  # Factor girómagetico del esín nuclear del proton (s⁻¹T⁻¹) de https://physics.nist.gov/cgi-bin/cuu/Value?gammap
G = 8.73e-7  # Gradiente externo (T/μm) de Validating NOGSE’s size distribution predictions in yeast cells Paper 1
D0 = 1e3 # Coeficiente de difusión (μm²/s) del Ejercico
# Voy a dejar todo en μm, s y T

#------------------------------------------------------------------------------------------

# Generación de distribution log-normal, con parámetros lcm y σ, en este caso lcm es el tamaño medio del compartimiento y σ es la desviación estándar del tamaño del compartimiento
# Mientras que σ es el ancho de la distribution de probabilidad con respecto a su media.
# Hagamos una función P(lc) de la distribucion log-normal

function P(lc,lcm,σ)
    return ( exp( -(log(lc) - log(lcm))^2 / (2σ^2) ) ) / (lc*σ*sqrt(2π))
end

#------------------------------------------------------------------------------------------

# Función M_l Magnetización de Hahn, para un tiempo t y un tamaño lc
function Ml_Hahn(t, lc)
    τc = lc^2 / (2 * D0)
    term1 = -γ^2 * G^2 * D0 * τc^2
    term2 = t - τc * (3 + exp(-t / τc) - 4 * exp(-t / (2 * τc)))
    return exp(term1 * term2)
end

#------------------------------------------------------------------------------------------

# Función S_hanh(t)
function S_han(lcm, σ, N, l0, lf, t)
    lc = range(l0, lf, length = N) # Generamos los tamaños de compartimientos lc desde l0 hasta lf

    P_lc = P.(lc,lcm, σ) # Consideramos media lcm y ancho σ
    M_lc = Ml_Hahn.(t, lc) # Calculamos M_lc(t) para cada tamaño de compartimiento
    
    S = sum(M_lc .* P_lc)
    return S
end

#------------------------------------------------------------------------------------------

function GenData(N, lcm, σ, l0, lf, time_sim, time_sample_lenght)
    # Generamos los tiempos de difusión y de tamaños de compartimientos
    t = range(0, time_sim, length = time_sample_lenght)
    lc = range(l0, lf, length = N)

    # Generamos las distribuciones
    P_l = P.(lc, lcm, σ)

    # Calculamos la señal
    S0 = S_han(lcm, σ, N, l0, lf, 0)
    S = S_han.(lcm, σ, N, l0, lf, t) ./ S0
    return t, lc, S, P_l
end

σs = 0.01:0.01:1 # Rango de σ a generar
lcms = 0.5:0.01:6 # Rango de lcm a generar


# Lo que dejamos constante es el número de compartimientos, el rango de tamaños l, el tiempo de simulación final y el muestreo de timepos
N = 2000
time_sample_lenght = 100

# Rango de tamaños de compartimientos en μm
l0 = 0.05
lf = 15

# Tiempo final de simulación en s
tf = 1

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
lc = range(l0, lf, length = N)
t = range(0, tf, length = time_sample_lenght)


function ReadCSVData(N, time_sample_lenght, l0, lf, tf)
    t = range(0, tf, length = time_sample_lenght)
    lc = range(l0, lf, length = N)
    length_t = length(t)
    length_lc = length(lc)
    max_lenght = maximum(length.([t, lc]))

    Probabilitys = zeros(length(lcms), length(σs), max_lenght)
    Signals = zeros(length(lcms), length(σs), max_lenght)

    for lcm in lcms
        for σ in σs
            df = CSV.read("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\TrackVersionsProgramsGit\\1-GeneracionDeDatos\\DatosCSV\\$(lcm)_$(σ)l_2k.csv", DataFrame)
            Probabilitys[findall(x -> x == lcm, lcms), findall(x -> x == σ, σs), :] = df.P_l
            Signals[findall(x -> x == lcm, lcms), findall(x -> x == σ, σs), :] = df.S

        end
    end

    return Probabilitys, Signals
end
# Una vez generados todos los archivos CSV extraemos los datos de estos archivos y los guardamos en arreglos de Julia

Probabilitys, Signals = ReadCSVData(N, time_sample_lenght, l0, lf, tf,)

# Ahora tenemos los datos de las señales y las distribuciones de probabilidad para cada combinación de σ y lcm
# Por cuestiones necesarias extremos la cantidad de σs, lcms, compartimientos lc, y tiempo sampleado t.
lenght_σs =  length(σs)
lenght_lcms = length(lcms)
total_lenght = length(σs)*length(lcms)
length_t = length(t)
length_lc = length(lc)
max_lenght = maximum(length.([t, lc]))


function reshape_data(old_matrix, old_shape, new_shape)

    old_matrix = old_matrix

    dim1, dim2, dim3 = old_shape

    new_matrix = zeros(Float64, dim3, dim1*dim2)

    for i in 1:dim1
        for j in 1:dim2
            for k in 1:dim3
                new_matrix[k,(i - 1) * dim2 + j] = old_matrix[i,j,k]
            end
        end
    end

    return new_matrix

end

function CenterData(Non_C_Matrix)
	data_matrix = Non_C_Matrix
	col_means = mean(data_matrix, dims = 1)
	centered_data = data_matrix .- col_means
	return centered_data
end

# Nuevo tamaño de los datos
new_size = (length(σs) * length(lcms), max_lenght)

# Ahora si tenemos los datos de entrada y salida es decir las señales y las distribuciones de probabilidad
dataIN = reshape_data(Signals, size(Signals), new_size)
dataOUT = reshape_data(Probabilitys, size(Probabilitys), new_size)

# En un momento para tener un DataFrame llenamos los datos de la señal con 0s los sacamos de cada columna.
dataIN = dataIN[1:length_t, :]

# Save dataIN
df_dataIN = DataFrame(dataIN, :auto)
df_dataOUT = DataFrame(dataOUT, :auto)

CSV.write("C:\\Users\\Propietario\\Desktop\ib\\5-Maestría\\TrackVersionsProgramsGit\\1-GeneracionDeDatos\\DataINOUT\\dataIN.csv", df_dataIN)
CSV.write("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\TrackVersionsProgramsGit\\1-GeneracionDeDatos\DataINOUT\\dataOUT.csv", df_dataOUT)

# Read DataFrame like matrix
dataIN = CSV.read("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\TrackVersionsProgramsGit\\1-GeneracionDeDatos\\DataINOUT\\dataIN.csv", DataFrame)
dataOUT = CSV.read("C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\TrackVersionsProgramsGit\\1-GeneracionDeDatos\DataINOUT\\dataOUT.csv", DataFrame)

# CSV as matrix
dataIN = Matrix(dataIN)
dataOUT = Matrix(dataOUT)


# Center data
dataIN_C = CenterData(dataIN)
dataOUT_C = CenterData(dataOUT)


# Esto ya hace PCA sobre la matriz dada donde cada observación es una columna de la matriz
pca_model_Signal = fit(PCA, dataIN_C)

# Esta instancia de PCA tiene distintas funciones como las siguientes

projIN = projection(pca_model_Signal) # Proyección de los datos sobre los componentes principales

# Vector con las contribuciones de cada componente (es decir los autovalores)
pcsIN = principalvars(pca_model_Signal)

# Obtenemos la variaza en porcentaje para cada componente principal
explained_varianceIN = pcsIN / sum(pcsIN) * 100

# Grafiquemos esto para ver que tan importante es cada componente principal
Plots.bar(explained_varianceIN, title="Varianza en porcentaje datos entrada",label = false, xlabel="Componente principal", ylabel="Varianza (%)")


reduced_dataIN = MultivariateStats.transform(pca_model_Signal, dataIN_C)


dim1, dim2, dim3 = size(Signals)
# Esto es para identificar en los datos de que señal y distribución vienen
column_lcm = zeros(dim1*dim2)
column_σs = zeros(dim1*dim2)
aux_lcm = collect(lcms)
aux_σs = collect(σs)

for i in 1:dim1
    for j in 1:dim2
        column_lcm[(i - 1)*dim2 + j] = aux_lcm[i]
        column_σs[(i - 1)*dim2 + j] = aux_σs[j]
    end
end

pca_model_Prob = fit(PCA, dataOUT_C) # Reducimos a 2 las componentes principales
	
projOUT = projection(pca_model_Prob) # Proyección de los datos sobre los componentes principales

# Vector con las contribuciones de cada componente
pcsOUT = principalvars(pca_model_Prob)

# Obtenemos la variaza en porcentaje para cada componente principal
explained_varianceOUT = pcsOUT / sum(pcsOUT) * 100

Plots.bar(explained_varianceOUT, title="Varianza en porcentaje datos salida",label = false, xlabel="Componente principal", ylabel="Varianza (%)")

# Podemos proyectar los datos sobre los componentes principales
reduced_dataOUT = MultivariateStats.transform(pca_model_Prob, dataOUT_C)


df_PCASignals = DataFrame(
		pc1 = reduced_dataIN[1, :],
	    pc2 = reduced_dataIN[2, :],
	    σs = column_σs,
	    lcm = column_lcm,
	)

df_PCASignals

find_σ = 0.01
find_lcm = 0.5

# find in a DataFrame the two columns that matchs with the two values
find_row = findall(x -> x == find_σ, df_PCASignals.σs)
find_column = findall(x -> x == find_lcm, df_PCASignals.lcm)

# find the index of the row and column
find_index = intersect(find_row, find_column)[1]

plot_lcms_S = @df df_PCASignals StatsPlots.scatter(
    :pc1,
    :pc2,
    group = :lcm,
    marker = (0.5,5),
    xaxis = (title = "PC1"),
    yaxis = (title = "PC2"),
    xlabel = "PC1",
    ylabel = "PC2",
    labels = false,  # Use the modified labels
    title = "PCA para S(t)",
)

x = reduced_dataIN[1,find_index]
y = reduced_dataIN[2,find_index]

ref_point = Plots.scatter!((x, y), label = L"l_{cm} = "*" $(df_PCASignals.lcm[find_index]) " * L"σ = " * " $(df_PCASignals.σs[find_index])" , color = "red")
signal_plot = Plots.plot(t, dataIN[:,find_index], label = false, color = "red", xlabel = "t (s)", ylabel = "S(t)", title = L"Señal S(t) para $l_{cm}$ = "*" $(df_PCASignals.lcm[find_index]) " * L"σ = " * " $(df_PCASignals.σs[find_index])")

plll = Plots.plot(plot_lcms_S, signal_plot, layout = (1,2), size = (1000, 600))

# Plots.savefig(plll, "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\TrackVersionsProgramsGit\\1-GeneracionDeDatos\\PCA_Signal_$(df_PCASignals.lcm[101])_$(df_PCASignals.σs[101]).png")

# plot_lcms_S = PlotlyJS.plot(
#     df_PCASignals,
#     x=:pc1, y=:pc2, mode="markers",
#     color=:lcm,
#     marker=attr(size=9, line=attr(width=1, color=:σs)),
#     text=text_values_S,
#     labels=Dict(
#         :pc1 => "PC1",
#         :pc2 => "PC2"
#     ),
#     Layout(title="PCA para S(t)",)
# )

open("./example.html", "w") do io
    PlotlyBase.to_html(io, plot_lcms_S.plot)
end

df_PCAPdist = DataFrame(
    pc1 = reduced_dataOUT[1, :],
    pc2 = reduced_dataOUT[2, :],
    σs = column_σs,
    lcm = column_lcm,
)

text_values_P = string.("σ=", df_PCAPdist.σs)

plot_lcms_P = PlotlyJS.plot(
    df_PCAPdist,
    x=:pc1, y=:pc2, mode="markers",
    color=:lcm,
    marker=attr(size=9, line=attr(width=1, color=:σs)),
    text=text_values_P,
    labels=Dict(
        :pc1 => "PC1",
        :pc2 => "PC2"
    ),
    Layout(title="PCA para P(lc)",)
)

open("./example.html", "w") do io
    PlotlyBase.to_html(io, plot_lcms_P.plot)
end