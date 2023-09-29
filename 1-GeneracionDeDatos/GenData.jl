using Plots
using Distributions
using Random
using LaTeXStrings
using KernelDensity
using QuadGK
# Poner una semilla random para reproducibilidad (opcional)
Random.seed!(0)
gr()

# Constantes útiles
γ = 2.675e8  # Factor girómagetico del esín nuclear del proton (s⁻¹T⁻¹) de https://physics.nist.gov/cgi-bin/cuu/Value?gammap
G = 8.73e-7  # Gradiente externo (T/μm) de Validating NOGSE’s size distribution predictions in yeast cells Paper 1
D0 = 1e3 # Coeficiente de difusión (μm²/s) del Ejercico
# Voy a dejar todo en μm, s y T

#------------------------------------------------------------------------------------------------------------
# Generación de distribution log-normal, con parámetros lc y σ, en este caso lc es el tamaño medio del compartimiento y σ es la desviación estándar del tamaño del compartimiento
# Mientras que σ es el ancho de la distribution de probabilidad con respecto a su media.
# Una de las dudas es si σ va así o como σ_log puesto que en Julia la distribution log-normal
# Mismo con μ
# es p(x) = ( exp( -(ln(x) - μ)² / (2σ²)) ) / (xσ√(2π))
# N es el número de compartimientos que se quieren generar
function P(lc, σ, N)
    μ = lc
    data = rand(LogNormal(μ, σ), N)
    return data
end
#------------------------------------------------------------------------------------------------------------

# Hagamos una función P a mano
function P_handmade(l,lc,σ)
    #lc = log(lc) 
    #σ = log(σ)
    # Vienen de una variable aleatoria 
    # de la cual su exponencial es de una distribucion normal y ahí si son la media y varianza usuales
    # Si se pone el log en estas da una P negativa
    return ( exp( -(log(l) - lc)^2 / (2σ^2)) ) / (l*σ*sqrt(2π))
end

#------------------------------------------------------------------------------------------------------------

# Pruebo si P está bien simulada en ambos casos (lognormal y handmade)
function Test_P()
    μs = [0.7, 0.7, 1.0, 1.0, 2e-4, 2e-4] # Los tomamos solo como si fuesen parámetros, sin unidades
    σs = [0.2, 0.4, 0.2, 0.4, 1, 0.25] # Mismo con esto
    Ns = [200000, 200000, 200000, 200000, 200000, 200000] # Cantidad de compartimientos que se tendrían en cada imágen
    pl = plot(xlabel="l", ylabel="P(l)", legend=:best)
    xlims!(0, 6)
    pl2 = plot(xlabel="l", ylabel="Cuentas / P(l)", legend=:best)
    xlims!(0, 6)
    for i in 1:length(μs)
        p = P(μs[i], σs[i], Ns[i])
        kde_result = kde(p)
        plot!(pl,kde_result.x, kde_result.density, label = L"$\mu$" * "= $(μs[i])" * L" $\sigma$" * "= $(σs[i])")
        histogram!(pl2, p, label=L"$\mu$" * "= $(μs[i])" * L"$\sigma$" * "= $(σs[i])", normed=true, bins = 55)
    end
    display(pl)
    display(pl2)
end
#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------
# Veamos esta función a mano
function TestP_Handmade()
    μs = [0.7, 0.7, 1.0, 1.0, 2e-4, 2e-4] # Los tomamos solo como si fuesen parámetros, sin unidades
    σs = [0.2, 0.4, 0.2, 0.4, 1, 0.25] # Mismo con esto
    l = range(0, 6, length=1000)
    pl = plot(xlabel="l", ylabel="P(l)", legend=:best)
    for i in 1:length(μs)
        p = P_handmade.(l, μs[i], σs[i])
        plot!(pl, l, p, label=L"$\mu$" * "= $(μs[i])" * L" $\sigma$" * "= $(σs[i])")
    end
    display(pl)
    # Hay una que se pasa de 1, la integral dará 1?
    p_sus(x) = ( exp( -(log(x) - 2e-4)^2 / (2*0.25^2)) ) / (x*0.25*sqrt(2π))
    # Integramos p_sus desde 0 a inf
    result, error = quadgk(p_sus, 0, Inf)
    println("La integral de la función p_sus da: ", result)
end
#------------------------------------------------------------------------------------------------------------

# Función M_l Magnetización de Hahn, para un tiempo t y un tamaño medio de compartimiento lc
function Ml_Hahn(t, l)
    τc = l^2 / (2 * D0)
    term1 = -γ^2 * G^2 * D0 * τc^2
    term2 = t - τc * (3 + exp(-t / τc) - 4 * exp(-t / (2 * τc)))
    return exp(term1 * term2)
end
#------------------------------------------------------------------------------------------------------------

# Testear que la Magnetizacion esta bien simulada comparando con
# Fig. 2(a) del paper [2] que muestra el comportamiento de M vs
# (γ2G2D0)1/3t en el rango [1, 100]. La figura muestra varias cur-
# vas, cada una esta asociada a un tamaño distinto de cavidad lc
# determinado por el tiempo de correlacion τc dado en el caption.
function Test_MHahn()
    time = range(0, 1, length=50000)
    factor = (γ^2 * G^2 * D0)^(1/3)
    list = [0.1, 0.15, 0.25, 0.4, 1.]
    lc__values = [sqrt(2 * D0 * value / factor) for value in list]
    pl = plot(xlabel=L"(γ^2 G^2 D_0)^{1/3} t", ylabel= "Magnetización " * L"M(t)/M(0)", legend=:best)
    M = zeros(length(time))
    xlims!(0, 100)
    for lc in lc__values
        # for i in 1:length(time)
        #     M[i] = Ml_Hahn(collect(time)[i], lc)/ Ml_Hahn(0, lc)
        # end
        M = [Ml_Hahn(t, lc) for t in time] ./ Ml_Hahn(0, lc)
        lc = Float32(lc)
        plot!(pl, time.*factor , M, label=L"l_c" * "= $lc μm")
    end
    display(pl)
    return nothing
end
#------------------------------------------------------------------------------------------------------------
# Utilice la Eq. (1) para calcular SHahn(t) para cada valor de t en su
# rango.
# - Para Pl(t) de Eq. (2) considerar un  ́unico valor lc = 3,7μm
# y generar varias distribuciones posibles variando σ.
# - Para Ml,Hahn considerar un  ́unico valor de τc correspondiente
# a lc = 3,7μm y D0 = 10−5cm2/s (ojo con el cambio de unidades)

function S_han(lc, N, time_sim)
    time = range(0, time_sim, length=1000)
    l = range(0.05, 50, length = N)
    σs = [0.01, 0.10, 0.25, 0.50, 1.0]
    pl = plot(xlabel="t (s)", ylabel="Señal (U.A)", legend=:best)
    xlims!(0, 0.05)
    S = zeros(length(time))
    M = zeros(length(time))

    for σ in σs
        for i in 1:length(time)
            for j in 1:length(l)
                p = P(l[j], σ, N)
                kde_result = kde(p)
                p = kde_result.density
                M[i] = Ml_Hahn(collect(time)[i], l[j])
                S[i] += p[j] * M[i]
            end
        end
        plot!(pl,time, S, label=L"σ" * "= $σ")
    end
    #-------------------------------------------------------------------------------------------
    # Veamos ahora con la función simulada a mano, vamos a tomar 2000 l's desde 0.05 hasta 50
    # equidistantes
    
    pl2 = plot(xlabel="t (s)", ylabel="Señal (U.A)", legend=:best)
    xlims!(0, 0.05)
    S2 = zeros(length(time))
    M2 = zeros(length(time))
    for σ in σs
        for i in 1:length(time)
            for j in 1:length(l)
                p2 = P_handmade.(l[j], lc, σ)
                M2[i] = Ml_Hahn(collect(time)[i], l[j])
                S2[i] += p2[j] * M2[i]
            end
        end
        plot!(pl2,time, S2, label=L"σ" * "= $σ")
    end
    display(pl)
    display(pl2)
    return S
end

#------------------------------------------------------------------------------------------------------------

# Función main

function main()
    Test_P()
    TestP_Handmade()
    Test_MHahn()
    S_han(3.7, 2000, 1)
    return nothing
end

#------------------------------------------------------------------------------------------------------------
main()
# p = P(3.7, 0.01, 2000)
# kde_result = kde(p)
# kde_result.x
# p_l = collect(kde_result.x)
# kde_result.density
# histogram(p_l,bins=55)