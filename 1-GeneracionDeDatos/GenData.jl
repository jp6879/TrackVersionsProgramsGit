using Plots
using Distributions
using Random
using LaTeXStrings
using KernelDensity
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
# Mismo con μ
# es p(x) = ( exp( -(ln(x) - μ)² / (2σ²)) ) / (xσ√(2π))
# N es el número de compartimientos que se quieren generar
function P(lc, σ, N)
    #μ = log(lc)
    #σ_log = log(σ)
    data = rand(LogNormal(lc, σ), N)
    return data
end
#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------
# Función M_l Magnetización de Hahn, para un tiempo t y un tamaño medio de compartimiento lc
function Ml_Hahn(t, lc)
    τc = lc^2 / (2 * D0)
    term1 = -γ^2 * G^2 * D0 * τc^2
    term2 = t - τc * (3 + exp(-t / τc) - 4 * exp(-t / (2 * τc)))
    return exp(term1 * term2)
end
#------------------------------------------------------------------------------------------------------------
# Pruebo si P está bien simulada
function Test_P()
    μs = [0.7, 0.7, 1.0, 1.0, 2e-4, 2e-4]
    σs = [0.2, 0.4, 0.2, 0.4, 1, 0.25]
    Ns = [2000, 2000, 2000, 2000, 2000, 2000]
    pl = plot(xlabel="l", ylabel="P(l)", legend=:best)
    xlims!(0, 6)
    pl2 = plot(xlabel="n", ylabel="P(l)", legend=:best)
    for i in 1:length(μs)
        p = P(μs[i], σs[i], Ns[i])
        plot!(pl2, p, label=L"$\mu$" * "= $(μs[i])" * L" $\sigma$" * "= $(σs[i])")
        kde_result = kde(p)
        #plot!(pl,kde_result.x, kde_result.density, label = L"$\mu$" * "= $(μs[i])" * L" $\sigma$" * "= $(σs[i])")
        histogram!(pl, p, label=L"$\mu$" * "= $(μs[i])" * L"$\sigma$" * "= $(σs[i])", normed=true)
    end
    display(pl)
    display(pl2)
end
#------------------------------------------------------------------------------------------------------------
# Hagamos una función P a mano
function P_handmade(l,lc,σ)
    #lc = log(lc)
    #σ = log(σ)
    return ( exp( -(log(l) - lc)^2 / (2σ^2)) ) / (l*σ*sqrt(2π))
end
#------------------------------------------------------------------------------------------------------------
# Veamos si ahora si funciona
function TestP_Handmade()
    l = range(0, 7, length=1000)
    lc = 2
    σ = 1
    P_hand = P_handmade.(l, lc, σ)
    pl = plot(l ,P_hand ,xlabel="l", ylabel="P(l)", legend=:best)
    ylims!(0, 0.010)
    display(pl)
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
    lc__values = [sqrt(2 * D0 * value / factor) for value in list]

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
# Utilice la Eq. (1) para calcular SHahn(t) para cada valor de t en su
# rango.
# - Para Pl(t) de Eq. (2) considerar un  ́unico valor lc = 3,7μm
# y generar varias distribuciones posibles variando σ.
# - Para Ml,Hahn considerar un  ́unico valor de τc correspondiente
# a lc = 3,7μm y D0 = 10−5cm2/s (ojo con el cambio de unidades)

function S_han(lc, N, time_sim)
    time = range(0, time_sim, length=1000)
    σs = [0.01, 0.10, 0.25, 0.50, 1.0]
    pl = plot(xlabel="t", ylabel="Señal", legend=:best)
    xlims!(0, 0.15)
    S = zeros(length(time))
    M = zeros(length(time))

    for σ in σs
        for i in 1:length(time)
            p = P(lc, σ, N)
            M[i] = Ml_Hahn(time[i], lc)
            for l in 1:N
                S[i] += p[l] * M[i]
            end
        end
        plot!(pl,time, S, label=L"σ" * "= $σ")
    end

    # for σ in σs
    #     p = P(lc, σ, N)
    #     M = [Ml_Hahn(t, lc) for t in time]
    #     for l in 1:N
    #         S += p[l] .* M
    #     end
    #     plot!(pl,time, S, label=L"σ" * "= $σ")
    # end

    display(pl)
    return S
end

#------------------------------------------------------------------------------------------------------------

# Función main

function main()
    Test_P()
    TestP_Handmade()
    Test_MHahn()
    S_han(3.7e-4, 2000, 100)
    return nothing
end

#------------------------------------------------------------------------------------------------------------
main()