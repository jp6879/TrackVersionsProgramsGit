using Plots
using Distributions
using Random
using LaTeXStrings
using KernelDensity
using QuadGK
using Flux
using Flux: train!
using Statistics
using MultivariateStats

# Poner una semilla random para reproducibilidad (opcional)
Random.seed!(0)

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

# Testeamos la función P
function TestP()
    lcms = [0.7, 0.7, 1.0, 1.0, 3.7, 2] # Variamos el valor medio lcm
    σs = [0.2, 0.4, 0.2, 0.4, 0.1, 0.25] # Variamos el ancho σ
    lc = range(0, 10, length=1000)
    pl = plot(xlabel= L"lc (\mu m)", ylabel= L"P(lc)", legend=:best, title = "Distribución LogNormal")
    for i in 1:length(lcms)
        p = P.(lc, lcms[i], σs[i])
        plot!(pl, lc, p, label=L"$lcm$" * "= $(lcms[i]) "* L"μm\:" * L"$\sigma$" * "= $(σs[i])")
    end
    display(pl)
end;

TestP()

#------------------------------------------------------------------------------------------

# Función M_l Magnetización de Hahn, para un tiempo t y un tamaño lc
function Ml_Hahn(t, lc)
    τc = lc^2 / (2 * D0)
    term1 = -γ^2 * G^2 * D0 * τc^2
    term2 = t - τc * (3 + exp(-t / τc) - 4 * exp(-t / (2 * τc)))
    return exp(term1 * term2)
end

#------------------------------------------------------------------------------------------

# Testeamos la función Ml_Hahn
function Test_MHahn()
    time = range(0, 1, length=50000)
    factor = (γ^2 * G^2 * D0)^(1/3)
    list = [0.1, 0.15, 0.25, 0.4, 1.]
    lc__values = [sqrt(2 * D0 * value / factor) for value in list] # Valores de lc para los cuales vamos a graficar
    pl = plot(xlabel=L"(γ^2 G^2 D_0)^{1/3} t", ylabel= "Magnetización " * L"M_{lc}(t)/M_{lc}(0)", legend=:best, title = "Magnetización de Hahn")
    M = zeros(length(time))
    xlims!(0, 100)
    for lc in lc__values
        M = [Ml_Hahn(t, lc) for t in time] ./ Ml_Hahn(0, lc)
        lc = Float32(lc)
        plot!(pl, time.*factor , M, label=L"l_c" * "= $lc μm")
    end
    display(pl)
end;

Test_MHahn()

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

# Testeamos la función S(t)
function Test_S()
	N = 2000 # Numero de compartimientos
	lcm = 3.7 # Tamaño medio del compartimiento μm
	# Rango de compartimientos simulados en μm
	l0 = 0.05
	lf = 50
	time = range(0, 1, length = 10000) # Tiempo a simular en s
	σs = [0.01, 0.10, 0.25, 0.50, 1.0] # Distitnos σ
	pl = plot(xlabel="t (s)", ylabel="S(t) (U.A)", legend=:best, title = "Señal " * L"$S_{Hahn}$")
	xlims!(0, 0.015)
	for σ in σs
        S0 = S_han(3.7, σ, 2000, l0, lf, 0)
	    St = S_han.(3.7, σ, 2000, l0, lf, time) ./ S0
	    plot!(pl, time, St, label = L"$\sigma$" * "= $(σ)")
	end
	display(pl)
end;
Test_S()

#------------------------------------------------------------------------------------------

function GenData(N, lcm, σ, l0, lf, time_sim)
    # Generamos los tiempos de difusión y de tamaños de compartimientos
    t = range(0, time_sim, length = 5000)
    lc = range(l0, lf, length = N)
    # Generamos las distribuciones
    P_l = P.(lc, lcm, σ)
    # Calculamos la señal
    S0 = S_han(lcm, σ, N, l0, lf, 0)
    S = S_han.(lcm, σ, N, l0, lf, t) ./ S0
    return t, P_l, S
end

#------------------------------------------------------------------------------------------

# Ahora tenemos un conjunto de pares de datos de entrada y salida.
# Cada valor de t y su correspondiente SHahn(t) es una entrada, y el
# vector P(lc) es la salida correspondiente. O puede verse en forma
# inversa tambien. La idea sera encontrar un mapeo que relacione
# ambos conjuntos de datos.
# Veamos los datos que tenémos

function Test_GenData()
    N = 2000 # Numero de compartimientos
	lcm = 3.7 # Tamaño medio del compartimiento
	# Rango de compartimientos simulados
	l0 = 0.05
	lf = 10
    t_sim = 1.0 # Tiempo a simular
    σ = 0.5
    t, P_l, S = GenData(N, lcm, σ, l0, lf, t_sim)
    println(size(collect(t)))
    println(size(P_l))
    println(size(S))
end

Test_GenData()

#------------------------------------------------------------------------------------------

# Ahora vamos a generar un conjunto de datos de entrada y salida para distintos σ, en principio estos tambien cambian
# con los numeros de compartimientos N, el tiempo que se simula t_sim, y los tamaños de compartimientos l0 y lf.
# por ahora vamos a dejarlos constantes y variar únicamente σ.
# En lo que entiendo los datos que tenemos son para cada σ una secuencia temporal S(t) que se relaciona
# con una distribución de tamaños P(lc). Querriamos ver los datos en 2 dimensiones, para esto querriamos hacer
# una reducción de dimensionalidad con técnicas de Machine Learning.

function GenINOUT(σs, N, t_sim, lcm, l0, lf)
    IN = []
    OUT = []
    for σ in σs
        t, P_l, S = GenData(N, lcm, σ, l0, lf, t_sim)
        t = collect(t)
        in = [t, S]
        out = P_l
        push!(IN, in)
        push!(OUT, out)
    end
    return IN, OUT
end

#------------------------------------------------------------------------------------------

# Testeamos la función GenINOUT

function Test_GenINOUT_PCA()
    σs = 0.05:0.1:1 # Distitnos σ entre 0.01 y 1
    σs = collect(σs)
    N = 2000 # Dejamos fijo el número de compartimientos por ahora
    t_sim = 1.0 # Tiempo a simular lo dejamos fijo
    lcm = 3.7 # Tamaño medio del compartimiento lo dejamos fijo
    l0 = 0.05 # Tamaño mínimo del compartimiento lo dejamos fijo
    lf = 10 # Tamaño máximo del compartimiento lo dejamos fijo
    lc = range(l0, lf, length = N)
    IN, OUT = GenINOUT(σs, N, t_sim, lcm, l0, lf)
    println(size(IN))
    println(size(OUT))

    # Entonces consideramos las entradas como un arreglo de una serie temporal S(t)
    # y un arreglo de tiempos t, y las salidas como un arreglo de una distribución de tamaños P(lc)
    # Apliquemos reducicón de la dimensionalidad a esto
    pl = plot(xlabel=L"t"*" (s)", ylabel=L"S(t)" *" (U.A)", legend=:best, title = "Señal " * L"S_{Hahn}")
    xlims!(0, 0.1)
    for i in 1:length(IN)
        plot!(pl,IN[i][1], IN[i][2], label = L"$\sigma$" * "= $(σs[i])")
    end

    display(pl)

    pl2 = plot(xlabel = L"lc"*" (μm)", ylabel = L"P(lc)\:" * "(μm" * L"^{-1}" *")", legend=:best, title = "Distribución " * L"P(lc)" * " asociadas a "* L"S(t)")
    for i in 1:length(OUT)
        plot!(pl2, lc, OUT[i], label = L"$\sigma$" * "= $(σs[i])")
    end

	display(pl2)

    # Vamos con el PCA para reducir la dimensionalidad
end

# Test_GenINOUT()

σs = 0.05:0.05:1 # Distitnos σ entre 0.01 y 1
σs = collect(σs)
N = 2000 # Dejamos fijo el número de compartimientos por ahora
t_sim = 1.0 # Tiempo a simular lo dejamos fijo
lcmean = 3.7 # Tamaño medio del compartimiento lo dejamos fijo
l0 = 0.05 # Tamaño mínimo del compartimiento lo dejamos fijo
lf = 10 # Tamaño máximo del compartimiento lo dejamos fijo
lc = range(l0, lf, length = N) # Generamos los tamaños de compartimientos lc desde l0 hasta lf
lc = collect(lc) # Lo pasamos a formato vector

IN, OUT = GenINOUT(σs, N, t_sim, lcmean, l0, lf) # Generamos los datos para los distintos σ

# Graficamos los datos generados de S(t)
pl = plot(xlabel=L"t"*" (s)", ylabel=L"S(t)" *" (U.A)", legend=:best, title = "Señal " * L"S_{Hahn}")
xlims!(0, 0.015)
for i in 1:length(IN)
    plot!(pl,IN[i][1], IN[i][2], label = L"$\sigma$" * "= $(σs[i])")
end

display(pl)

# Graficamos los datos generados de P(lc)

pl2 = plot(xlabel = L"lc"*" (μm)", ylabel = L"P(lc)\:" * "(μm" * L"^{-1}" *")", legend=:best, title = "Distribución " * L"P(lc)" * " asociadas a "* L"S(t)")
for i in 1:length(OUT)
    plot!(pl2, lc, OUT[i], label = L"$\sigma$" * "= $(σs[i])")
end

display(pl2)


# Para hacer reducción de dimensionalidad con PCA consideramos primero los datos de entrada, tenemos distintas series temporales para cada σ
# Cada una de estas sería una medición entonces hay un conjunto de 10000 puntos de t en un espacio de 20 dimensiones σ.
# En este caso al aplicar PCA lo que vamos a encontrar son vectores propios correspondientes a patrones instantaneos a traves de las
# series temporales. En cada momento en el tiempo, representamos la aplituda traves de las series temporales como una combinación lineal de estos patrones

# Para el caso de salida es lo mismo solo que estamos considerando N puntos de l en un espacio de 20 dimensiones σ. En este caso al aplicar PCA 
# lo que vamos a encontrar son vectores propios correspondientes a patrones de distribuciones de tamaños a traves de las series temporales.

# Consideremos entonces el arreglo de datos de entrada IN, y el arreglo de datos de salida OUT
dataIN = zeros(length(IN[1][2]), length(σs))
dataOUT = zeros(length(OUT[1]), length(σs))

for i in 1:length(σs)
    dataIN[:,i] = IN[i][2] # En [i][2] están las series S(t)
    dataOUT[:,i] = OUT[i]
end

# Hagamos el PCA para los datos de entrada
pca_modelIN = fit(PCA, dataIN; maxoutdim = 2)  # Reducir a un máximo de 2 componentes principales

# Transformamos los datos de entrada al espacio de menor dimensión
reduced_dataIN = transform(pca_modelIN, dataIN)

# Obtenemos los componentes principales
pcsIN = principalvars(pca_modelIN)

# Obtenemos la variaza en porcentaje para cada componente principal
explained_varianceIN = principalvars(pca_modelIN) / sum(principalvars(pca_modelIN))

# Veamos esta varianza en un gráfico de barras
bar(explained_varianceIN, label="Varianza en porcentaje IN", legend=:topright, xlabel="Componente principal", ylabel="Varianza en porcentaje")

# Teniendo en cuenta esto vemos que la mayor varianza se encuentra en la 1era componente principal podemos prescindir de las demas para hacer reducción de dimensionalidad

reductedIN = reduced_dataIN[1,:] # Tomamos solo la 1era componente principal

# Hagamos lo mismo para los datos de salida

pca_modelOUT = fit(PCA, dataOUT; maxoutdim = 2)  # Reducir a 2 componentes principales

# Transformamos los datos de salida al espacio de menor dimensión
reduced_dataOUT = transform(pca_modelOUT, dataOUT)

# Obtenemos los componentes principales
pcsOUT = principalvars(pca_modelOUT)

# Obtenemos la variaza en porcentaje para cada componente principal
explained_varianceOUT = principalvars(pca_modelOUT) / sum(principalvars(pca_modelOUT))

# Veamos esta varianza en un gráfico de barras
bar(explained_varianceOUT, label="Varianza en porcentaje OUT", legend=:topright, xlabel="Componente principal", ylabel="Varianza en porcentaje")

# Teniendo en cuenta esto vemos que la mayor varianza se encuentra en la 1era componente principal podemos prescindir de las demas para hacer reducción de dimensionalidad

reductedOUT = reduced_dataOUT[1,:] # Tomamos solo la 1era componente principal

# Graficamos los datos de entrada y salida reducidos
scatter(reductedIN, reductedOUT, label="Reduced Data", legend=:topleft, xlabel="Principal Component IN", ylabel="Principal Component OUT")
#------------------------------------------------------------------------------------------

