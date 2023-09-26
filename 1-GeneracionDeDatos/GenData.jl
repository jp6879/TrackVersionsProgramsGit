using Plots
using Distributions
using Random
using LaTeXStrings
# Poner una semilla random para reproducibilidad (opcional)
Random.seed!(0)

# Constantes útiles
γ = 2.675e8  # Factor girómagetico del esín nuclear del proton (s⁻¹T⁻¹) de https://physics.nist.gov/cgi-bin/cuu/Value?gammap
G = 8.73e-3  # Gradiente externo (T/cm) de Validating NOGSE’s size distribution predictions in yeast cells Paper 1
D0 = 1e-5 # Coeficiente de difusión (cm²/s) del Ejercico

#------------------------------------------------------------------------------------------------------------
# Generación de distribution log-normal, con parámetros lc y σ, en este caso lc es el tamaño medio del compartimiento y σ es la desviación estándar del tamaño del compartimiento
# Mientras que σ es el ancho de la distribution de probabilidad con respecto a su media.
# Una de las dudas es si σ va así o como σ_log puesto que en Julia la distribution log-normal
# es p(x) = ( exp( -(ln(x) - μ)² / (2σ²)) ) / (xσ√(2π))
# N es el número de compartimientos que se quieren generar
function P(lc, σ, N)
    μ = log(lc)
    σ_log = log(σ)
    data = rand(LogNormal(μ, σ_log), N)
    return data
end
#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------
# Función M_l Magnetización de Hahn, para un tiempo t y un tamaño de compartimiento lc
function Ml_Hahn(t, lc)
    τc = lc / (2 * D0)
    term1 = -γ^2 * G^2 * D0 * τc^2
    term2 = t - τc * (3 + exp(-t / τc) - 4 * exp(-t / (2 * τc)))
    return exp(term1 * term2)
end
#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------
# Testear que la Magnetizacion esta bien simulada comparando con
# Fig. 2(a) del paper [2] que muestra el comportamiento de M vs
# (γ2G2D0)1/3t en el rango [1, 100]. La figura muestra varias cur-
# vas, cada una esta asociada a un tamaño distinto de cavidad lc
# determinado por el tiempo de correlacion τc dado en el caption.
function Test_MHahn()
    time = range(0, 100, length=50000)
    factor = (γ^2 * G^2 * D0)^(1/3)
    list = [0.1, 0.15, 0.25, 0.4, 1.]
    lc__values = [2 * D0 * τc / factor for τc in list]

    pl = plot(xlabel=L"(γ^2 G^2 D_0)^{1/3} t", ylabel= "Magnetización " * L"M(t)/M(0)", legend=:best)
    xlims!(0, 100)
    for lc in lc__values
        M = [Ml_Hahn(t, lc) for t in time] ./ Ml_Hahn(0, lc)
        lc = Float32(lc)
        plot!(pl,time.*factor, M, label=L"l_c" * "= $lc cm")
    end
    display(pl)
    return nothing
end
#------------------------------------------------------------------------------------------------------------
# Función main

function main()
    
    return nothing
end

#------------------------------------------------------------------------------------------------------------
# Testear que la Magnetizacion esta bien simulada comparando con figura 2a de Paper 2
Test_MHahn()

main()