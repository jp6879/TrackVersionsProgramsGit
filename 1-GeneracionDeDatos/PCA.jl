include("GenData.jl")

using Flux
using Flux: train!
using Statistics
using MultivariateStats
using DataFrames
using CSV
using MLJ
#------------------------------------------------------------------------------------------

# Vamos a hacer una variación de las variables σ y lc y generar una grilla de datos para todos los casos estos datos los vamos a guardar en un archivo .csv
# Para así no hacer todo el proceso de generación de datos nuevamente

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

#------------------------------------------------------------------------------------------

function GenCSVData(N, time_sample_lenght, l0, lf, tf, lcms, σs)

    function fill_missing(value, column, max_lenght)
        if length(column) < max_lenght
            return vcat(column,fill(value, max_lenght - length(column)))
        else
            return column
        end
    end


    for lcm in lcms
        for σ in σs
            t, l, S, P_l = GenData(N, lcm, σ, l0, lf, tf, time_sample_lenght)
            max_lenght = maximum(length.([l,t]))
            
            t = fill_missing(0, t, max_lenght)
            l = fill_missing(0, l, max_lenght)
            S = fill_missing(0, S, max_lenght)
            P_l = fill_missing(0, P_l, max_lenght)

            df = DataFrame(t = t, l = l, S = S, P_l = P_l)
            CSV.write("5-Maestría/TrackVersionsProgramsGit/1-GeneracionDeDatos/DatosCSV/$(lcm)_$(σ)l_2k.csv", df)
        end
    end
end

#------------------------------------------------------------------------------------------

# GenCSVData(N, time_sample_lenght, l0, lf, tf, lcms, σs)

#------------------------------------------------------------------------------------------

#Veamos los datos que generamos

function ReadCSVData(N, time_sample_lenght, l0, lf, tf)
    t = range(0, tf, length = time_sample_lenght)
    lc = range(l0, lf, length = N)
    length_t = length(t)
    length_lc = length(lc)
    max_lenght = maximum(length.([t, lc]))

    Probabilitys = zeros(length(lcms), length(σs), max_lenght)
    Signals = zeros(length(lcms), length(σs), max_lenght)

    for lcm in lcms
        # pl = plot(xlabel = L"$t$ (s)", ylabel = L"S(t)", title = L"Señal de Hahn, $l_{cm} = $"*" $(lcm)")
        # pl2 = plot(xlabel = L"$l_c$ (μm)", ylabel = L"$P(l_c)$", title = L"Distribución de tamaños $l_{cm} = $"*" $(lcm)")
        # xlims!(pl, (0,tf))
        # xlims!(pl2, (0,lf))
        for σ in σs
            df = CSV.read("5-Maestría/TrackVersionsProgramsGit/1-GeneracionDeDatos/DatosCSV/$(lcm)_$(σ)l_2k.csv", DataFrame)
            # plot!(pl,df.t[1:length_t], df.S[1:length_t], label = "lcm = $(lcm), σ = $(σ)", legend = false)
            # plot!(pl2,df.l, df.P_l, label = "lcm = $(lcm), σ = $(σ)", legend = false)

            Probabilitys[findall(x -> x == lcm, lcms), findall(x -> x == σ, σs), :] = df.P_l
            Signals[findall(x -> x == lcm, lcms), findall(x -> x == σ, σs), :] = df.S

        end
        # display(pl)
        # display(pl2)
    end

    return Probabilitys, Signals

end
#------------------------------------------------------------------------------------------

Probabilitys, Signals = ReadCSVData(N, time_sample_lenght, l0, lf, tf,)

lenght_σs =  length(σs)
lenght_lcms = length(lcms)
total_lenght = length(σs)*length(lcms)

size(Signals)
size(Probabilitys)

#------------------------------------------------------------------------------------------

length_t = length(t)
length_lc = length(lc)
max_lenght = maximum(length.([t, lc]))

# Hacemos un anális de componentes principales para ver si podemos reducir la dimensionalidad de los datos

# dataIN = reshape(Signals, length(σs) * length(lcms), max_lenght)
# dataOUT = reshape(Probabilitys, length(σs) * length(lcms), max_lenght)

function reshape_data(old_matrix, old_shape, new_shape)
    # Assuming you have an old_matrix of size (551x100x2000)
    old_matrix = old_matrix  # Replace this with your actual data

    new_matrix = zeros(Float64, 55100, 2000)

    dim1, dim2, dim3 = old_shape

    for i in 1:dim1
        for j in 1:dim2
            for k in 1:dim3
                new_matrix[(i - 1) * dim2 + j,k] = old_matrix[i,j,k]
            end
        end
    end

    return new_matrix

end

new_size = (length(σs) * length(lcms), max_lenght)
dataIN = reshape_data(Signals,size(Signals), new_size)
dataOUT = reshape_data(Probabilitys,size(Probabilitys), new_size)

# Cut the rows of dataIN to the max lenght of t
dataIN = dataIN[:, 1:length_t]
dataIN = transpose(dataIN)

plot(t, Signals[1,1,1:length_t], label = "lcm = $(lcms[1]), σ = $(σs[1])", xlabel = L"$t$ (s)", ylabel = L"S(t)", title = "Señal de Hahn")
plot!(t, Signals[1,20,1:length_t], label = "lcm = $(lcms[1]), σ = $(σs[20])")
plot!(t, Signals[1,100,1:length_t], label = "lcm = $(lcms[1]), σ = $(σs[100])")
plot!(t, Signals[20,1,1:length_t], label = "lcm = $(lcms[20]), σ = $(σs[1])")
plot!(t, Signals[20,100,1:length_t], label = "lcm = $(lcms[20]), σ = $(σs[100])")

# scatter!(t,dataIN[0*100 + 1,:], label = "lcm = $(lcms[1]), σ = $(σs[1])", markersize = 2)
# scatter!(t,dataIN[0*100 + 20,:], label = "lcm = $(lcms[1]), σ = $(σs[20])", markersize = 2)
# scatter!(t,dataIN[0*100 + 100,:], label = "lcm = $(lcms[1]), σ = $(σs[100])", markersize = 2)
# scatter!(t,dataIN[(20 - 1)*100 + 1,:], label = "lcm = $(lcms[20]), σ = $(σs[1])", markersize = 2)
# scatter!(t,dataIN[(20 - 1)*100 + 100,:], label = "lcm = $(lcms[20]), σ = $(σs[100])", markersize = 2)

scatter!(t,dataIN[:,0*100 + 1], label = "lcm = $(lcms[1]), σ = $(σs[1])", markersize = 2)
scatter!(t,dataIN[:,0*100 + 20], label = "lcm = $(lcms[1]), σ = $(σs[20])", markersize = 2)
scatter!(t,dataIN[:,0*100 + 100], label = "lcm = $(lcms[1]), σ = $(σs[100])", markersize = 2)
scatter!(t,dataIN[:,(20 - 1)*100 + 1], label = "lcm = $(lcms[20]), σ = $(σs[1])", markersize = 2)
scatter!(t,dataIN[:,(20 - 1)*100 + 100], label = "lcm = $(lcms[20]), σ = $(σs[100])", markersize = 2)


pca_model_Signal = fit(PCA, dataIN) # Reducimos a 2 las componentes principales
	
reduced_dataIN = MultivariateStats.transform(pca_model_Signal, dataIN)

reduced_dataIN[1,]


# Vector con las contribuciones de cada componente
pcsIN = principalvars(pca_model_Signal)

# Obtenemos la variaza en porcentaje para cada componente principal
explained_varianceIN = pcsIN / sum(pcsIN)

bar(explained_varianceIN, title="Varianza en porcentaje datos entrada",label = false, xlabel="Componente principal", ylabel="Varianza")

pl = plot(label = L"Datos $S(t)$ transformados", legend=:best, xlabel="PC 1", ylabel="PC 2")

dim1, dim2, dim3 = size(Signals)

for i in 1:dim1
    reduced_1 = zeros(100)
    reduced_2 = zeros(100)
    for j in 1:dim2
        reduced_1[j] = reduced_dataIN[1, (i - 1)*dim2 + j]
        reduced_2[j] = reduced_dataIN[2, (i - 1)*dim2 + j]
    end
    scatter!(pl,reduced_1, reduced_2, label="lcm = $(lcms[i])", markersize = 2)
end

pl

#scatter(reduced_dataIN[1, :], reduced_dataIN[2, :], label=L"Datos $S(t)$ transformados", legend=:best, xlabel="PC 1", ylabel="PC 2")

pca_model_Prob = fit(PCA, dataOUT) # Reducimos a 2 las componentes principales
	
reduced_dataOUT = MultivariateStats.transform(pca_model_Prob, dataOUT)

# Vector con las contribuciones de cada componente
pcsOUT = principalvars(pca_model_Prob)

# Obtenemos la variaza en porcentaje para cada componente principal
explained_varianceOUT = pcsOUT / sum(pcsOUT)

bar(explained_varianceOUT, title="Varianza en porcentaje datos salida",label = false, xlabel="Componente principal", ylabel="Varianza")

scatter(reduced_dataOUT[1, :], reduced_dataOUT[2, :], label=L"Datos $P(l)$ transformados", legend=:best, xlabel="PC 1", ylabel="PC 2")