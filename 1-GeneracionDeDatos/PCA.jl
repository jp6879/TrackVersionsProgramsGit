include("GenData.jl")

using Flux
using Flux: train!
using Statistics
using MultivariateStats
using DataFrames
using CSV

#------------------------------------------------------------------------------------------
# Vamos a hacer una variación de las variables σ y lc y generar una grilla de datos para todos los casos estos datos los vamos a guardar en un archivo .csv
# Para así no hacer todo el proceso de generación de datos nuevamente

σs = 0.01:0.02:1 # Rango de σ a generar
lcms = 0.5:0.2:6 # Rango de lcm a generar


function GenCSVData()

    # Lo que dejamos constante es el número de compartimientos, el rango de tamaños l, el tiempo de simulación final y el muestreo de timepos
    N = 2000
    time_sample_lenght = 500
    
    # Rango de tamaños de compartimientos en μm
    l0 = 0.05
    lf = 10

    # Tiempo final de simulación en s
    tf = 1

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
            
            t = fill_missing(NaN, t, max_lenght)
            l = fill_missing(NaN, l, max_lenght)
            S = fill_missing(NaN, S, max_lenght)
            P_l = fill_missing(NaN, P_l, max_lenght)

            df = DataFrame(t = t, l = l, S = S, P_l = P_l)
            CSV.write("5-Maestría/TrackVersionsProgramsGit/1-GeneracionDeDatos/DatosCSV/$(lcm)_$(σ)z.csv", df)
        end
    end
end

#------------------------------------------------------------------------------------------

# GenCSVData()

#Veamos los datos que generamos

function ReadCSVData()
    for lcm in lcms
        pl = plot(xlabel = "t (s)", ylabel = "S (t)", title = "Señal de Hahn")
        pl2 = plot(xlabel = "l (μm)", ylabel = "P(l)", title = "Distribución de tamaños de compartimientos")
        xlims!(pl, (0,0.05))
        xlims!(pl2, (0,7))
        for σ in σs
            df = CSV.read("5-Maestría/TrackVersionsProgramsGit/1-GeneracionDeDatos/DatosCSV/$(lcm)_$(σ)z.csv", DataFrame)
            plot!(pl,df.t, df.S, label = "lcm = $(lcm), σ = $(σ)", legend = false)
            plot!(pl2,df.l, df.P_l, label = "lcm = $(lcm), σ = $(σ)", legend = false)
        end
        display(pl)
        display(pl2)
    end
end

ReadCSVData()