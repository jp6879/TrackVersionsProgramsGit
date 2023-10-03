# Generación de datos de RMN Hahn
using Plots
using Distributions
using Random
using LaTeXStrings

# Poner una semilla random para reproducibilidad (opcional)
# Random.seed!(0)

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

#------------------------------------------------------------------------------------------