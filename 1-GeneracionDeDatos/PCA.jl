include("GenData.jl")

using MultivariateStats
using DataFrames

#------------------------------------------------------------------------------------------
# Función que cambia la forma de los datos de entrada
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

#------------------------------------------------------------------------------------------
# Función que centra los datos
function CenterData(Non_C_Matrix)
	data_matrix = Non_C_Matrix
	col_means = mean(data_matrix, dims = 1)
	centered_data = data_matrix .- col_means
	return centered_data
end
#------------------------------------------------------------------------------------------

function PCA_data()
    
end



# Vamos a hacer una variación de las variables σ y lc y generar una grilla de datos para todos los casos estos datos los vamos a guardar en un archivo .csv
# Para así no hacer todo el proceso de generación de datos nuevamente

# σs = 0.01:0.01:1 # Rango de σ a generar
# lcms = 0.5:0.01:6 # Rango de lcm a generar

# # Lo que dejamos constante es el número de compartimientos, el rango de tamaños l, el tiempo de simulación final y el muestreo de timepos
# N = 2000
# time_sample_lenght = 100

# # Rango de tamaños de compartimientos en μm
# l0 = 0.05
# lf = 15

# # Tiempo final de simulación en s
# tf = 1

# # Ahora generamos los datos para eso necesitamos hacer el sampling de los lc y los t
# lc = range(l0, lf, length = N)
# t = range(0, tf, length = time_sample_lenght)

# # Por cuestiones de PCA necesarias nos guardamos estas variables: la cantidad de σs, lcms, compartimientos lc, y tiempo sampleado t y el máximo entre estos ultimos
# lenght_σs =  length(σs)
# lenght_lcms = length(lcms)
# total_lenght = length(σs)*length(lcms)
# length_t = length(t)
# length_lc = length(lc)
# max_lenght = maximum(length.([t, lc]))

# Generacion de datos en CSV, los inputs son
# N: cantidad de compartimientos lc
# time_sample_lenght: cantidad de puntos de tiempo
# l0: tamaño mínimo de compartimiento
# lf: tamaño máximo de compartimiento
# tf: tiempo máximo de difusión
# lcms: lista de tamaños medios de compartimientos
# σs: lista de desviaciones estándar de compartimientos

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

# Esta función me genera los CSV no llamarla si ya se generaron los datos
# GenCSVData(N, time_sample_lenght, l0, lf, tf, lcms, σs)

#------------------------------------------------------------------------------------------

#Veamos los datos que generamos

function ReadCSVData(N, time_sample_lenght, l0, lf, tf, lcms, σs)
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
# Una vez generados todos los archivos CSV extraemos los datos de estos archivos y los guardamos en arreglos de Julia

# Probabilitys, Signals = ReadCSVData(N, time_sample_lenght, l0, lf, tf, lcms, σs)

#------------------------------------------------------------------------------------------
# Hacemos un anális de componentes principales para ver si podemos reducir la dimensionalidad de los datos

# Primero comenzamos con los datos de entrada, es decir, los datos de la señal S(t) en este caso tenemos
# señales (puntos de dimensión 100 que se corresponde con el tiempo de muestreo) para cada combinación de σ y lcm, es decir 55100 puntos viviendo en R^100
# veremos que puede hacer PCA para reducir la dimensionalidad de estos datos

# Para hacer este PCA necesitamos una matriz con los datos de entrada, es decir, una matriz de 100x55100
# Para esto vamos a hacer una función que nos permita hacer esto

# Los datos que recibimos son de la forma (551x100x2000) y queremos que sean de la forma (2000x55100) para identificarlos

# Función que cambia la forma de los datos de entrada
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

# Función que centra los datos de entrada

function CenterData(Non_C_Matrix)
	data_matrix = Non_C_Matrix
	col_means = mean(data_matrix, dims = 1)
	centered_data = data_matrix .- col_means
	return centered_data
end

# # Nuevo tamaño de los datos
# new_size = (length(σs) * length(lcms), max_lenght)

# # Ahora si tenemos los datos de entrada y salida es decir las señales y las distribuciones de probabilidad
# dataIN = reshape_data(Signals, size(Signals), new_size)
# dataOUT = reshape_data(Probabilitys, size(Probabilitys), new_size)

# # En un momento para tener un DataFrame llenamos los datos de la señal con 0s los sacamos de cada columna
# dataIN = dataIN[1:length_t, :]

# dataIN_C = CenterData(dataIN)
# dataOUT_C = CenterData(dataOUT)

#------------------------------------------------------------------------------------------
La correspondencia de los datos es la siguiente (lcm[i], σ[j]) -> ((i-1)*length(σs) + j) para i = 1:length(lcms) y j = 1:length(σs)
En los siguientes graficos se puede ver esto mismo pero lo dejamos comentado solo como chequeo por si se quiere probar

plot(t, Signals[1,1,1:length_t], label = "lcm = $(lcms[1]), σ = $(σs[1])", xlabel = L"$t$ (s)", ylabel = L"S(t)", title = "Señal de Hahn")
plot!(t, Signals[1,20,1:length_t], label = "lcm = $(lcms[1]), σ = $(σs[20])")
plot!(t, Signals[1,100,1:length_t], label = "lcm = $(lcms[1]), σ = $(σs[100])")
plot!(t, Signals[20,1,1:length_t], label = "lcm = $(lcms[20]), σ = $(σs[1])")
plot!(t, Signals[20,100,1:length_t], label = "lcm = $(lcms[20]), σ = $(σs[100])")

scatter!(t,dataIN_C[:,0*100 + 1], label = "lcm = $(lcms[1]), σ = $(σs[1])", markersize = 2)
scatter!(t,dataIN_C[:,0*100 + 20], label = "lcm = $(lcms[1]), σ = $(σs[20])", markersize = 2)
scatter!(t,dataIN_C[:,0*100 + 100], label = "lcm = $(lcms[1]), σ = $(σs[100])", markersize = 2)
scatter!(t,dataIN_C[:,(20 - 1)*100 + 1], label = "lcm = $(lcms[20]), σ = $(σs[1])", markersize = 2)
scatter!(t,dataIN_C[:,(20 - 1)*100 + 100], label = "lcm = $(lcms[20]), σ = $(σs[100])", markersize = 2)

#------------------------------------------------------------------------------------------

# # Análisis de componentes principales para las Señales de Hahn

# # Esto ya hacec PCA sobre la matriz dada donde cada observación es una columna de la matriz
# pca_model_Signal = fit(PCA, dataIN_C)

# # Esta instancia de PCA tiene distintas funciones como las siguientes

# projIN = projection(pca_model_Signal) # Proyección de los datos sobre los componentes principales

# # Esto me da la matriz de Proyección, donde cada columna es un componente principal. Esta vendría a ser la matriz de los autovectores de la matriz de covarianza que nos quedamos para reducir la dimensionalidad

# # Vector con las contribuciones de cada componente (es decir los autovalores)
# pcsIN = principalvars(pca_model_Signal)

# # Obtenemos la variaza en porcentaje para cada componente principal
# explained_varianceIN = pcsIN / sum(pcsIN) * 100

# # Grafiquemos esto para ver que tan importante es cada componente principal
# bar(explained_varianceIN, title="Varianza en porcentaje datos entrada",label = false, xlabel="Componente principal", ylabel="Varianza")

# # Podemos proyectar los datos sobre los componentes principales, así tendríamos en 3 dimensiones los datos de entrada
# reduced_dataIN = MultivariateStats.transform(pca_model_Signal, dataIN_C)


# dim1, dim2, dim3 = size(Signals)
# pl = plot(label = L"Datos $S(t)$ transformados", legend=:best, xlabel="PC 1", ylabel="PC 2")
# pl2 = plot(label = L"Datos $S(t)$ transformados", legend=:best, xlabel="PC 1", ylabel="PC 2")
# for i in 1:(dim1-540)
#     reduced_1 = zeros(100)
#     reduced_2 = zeros(100)
#     for j in 1:dim2
#         if (i == 1 && j%5 == 0)
#             scatter!(pl2,(reduced_dataIN[1, (i - 1)*dim2 + j], reduced_dataIN[2, (i - 1)*dim2 + j]), label="lcm = $(lcms[i])" * "σ = $(σs[j])", layout=(1, 1), legend=:best)
#         end
#         reduced_1[j] = reduced_dataIN[1, (i - 1)*dim2 + j]
#         reduced_2[j] = reduced_dataIN[2, (i - 1)*dim2 + j]
#     end
#     scatter!(pl,reduced_1, reduced_2, label="lcm = $(lcms[i])", layout=(1, 1), legend=:best)
# end

# for i=1:2; plot!(pl,[0,projIN[i,1]], [0,projIN[i,2]], arrow=true, label=["1","2"][i], legend=false); end
# for i=1:2; plot!(pl2,[0,projIN[i,1]], [0,projIN[i,2]], arrow=true, label=["1","2"][i], legend=false); end

# pl
# pl2
# plx = scatter(reduced_dataIN[1, :], reduced_dataIN[2, :], label=L"Datos $S(t)$ transformados", legend=:best, xlabel="PC 1", ylabel="PC 2")
# for i=1:2; plot!(plx,[0,projIN[i,1]], [0,projIN[i,2]], arrow=true, label=["V1","V2"][i], legend=:best); end
# plx
#------------------------------------------------------------------------------------------

# Analisis de componentes principales para las distribuciones de probabilidad

# pca_model_Prob = fit(PCA, dataOUT) # Reducimos a 2 las componentes principales

# projOUT = projection(pca_model_Prob) # Proyección de los datos sobre los componentes principales

# # Vector con las contribuciones de cada componente
# pcsOUT = principalvars(pca_model_Prob)

# # Obtenemos la variaza en porcentaje para cada componente principal
# explained_varianceOUT = pcsOUT / sum(pcsOUT) * 100

# bar(explained_varianceOUT, title="Varianza en porcentaje datos salida",label = false, xlabel="Componente principal", ylabel="Varianza")


# Podemos proyectar los datos sobre los componentes principales
# reduced_dataOUT = MultivariateStats.transform(pca_model_Prob, dataOUT)

# pl3 = plot(label = L"Datos $P(l)$ transformados", legend=:best, xlabel="PC 1", ylabel="PC 2")
# pl4 = plot(label = L"Datos $P(l)$ transformados", legend=:best, xlabel="PC 1", ylabel="PC 2")
# for i in 1:(dim1)
#     reduced_1 = zeros(100)
#     reduced_2 = zeros(100)
#     for j in 1:dim2
#         if (i == 1 && j%5 == 0)
#             scatter!(pl3,(reduced_dataOUT[1, (i - 1)*dim2 + j], reduced_dataOUT[2, (i - 1)*dim2 + j]), label="lcm = $(lcms[i])" * "σ = $(σs[j])", layout=(1, 1), legend=:bottomleft)
#         end
#         reduced_1[j] = reduced_dataOUT[1, (i - 1)*dim2 + j]
#         reduced_2[j] = reduced_dataOUT[2, (i - 1)*dim2 + j]
#     end
#     scatter!(pl4,reduced_1, reduced_2, label="lcm = $(lcms[i])", layout=(1, 1), legend=:best)
# end

# for i=1:2; plot!(pl,[0,projOUT[i,1]], [0,projOUT[i,2]], arrow=true, label=["1","2"][i], legend=:bottomleft); end

# pl3
# pl4

# scatter(reduced_dataOUT[1, :], reduced_dataOUT[2, :], label=L"Datos $P(l)$ transformados", legend=:best, xlabel="PC 1", ylabel="PC 2")