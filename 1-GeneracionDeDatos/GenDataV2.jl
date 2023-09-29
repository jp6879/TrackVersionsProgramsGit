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

# Generación de distribution log-normal, con parámetros lc y σ, en este caso lc es el tamaño medio del compartimiento y σ es la desviación estándar del tamaño del compartimiento
# Mientras que σ es el ancho de la distribution de probabilidad con respecto a su media.
# Hagamos una función P(l) de la distribucion log-normal

function P(l,lc,σ)
    return ( exp( -(log(l) - log(lc))^2 / (2σ^2) ) ) / (l*σ*sqrt(2π))
end

#------------------------------------------------------------------------------------------

# Testeamos la función P
function TestP()
    lcs = [0.7, 0.7, 1.0, 1.0, 3.7e-6, 2e-4]
    σs = [0.2, 0.4, 0.2, 0.4, 1, 0.25]
    l = range(0, 6, length=1000)
    pl = plot(xlabel= L"l (\mu m)", ylabel= L"P(l)", legend=:best, title = "Distribución LogNormal")
    for i in 1:length(lcs)
        p = P.(l, lcs[i], σs[i])
        plot!(pl, l, p, label=L"$lc$" * "= $(lcs[i]) "* L"μm\:" * L"$\sigma$" * "= $(σs[i])")
    end
    display(pl)
end;

#TestP()

#------------------------------------------------------------------------------------------

# Función M_l Magnetización de Hahn, para un tiempo t y un tamaño medio de compartimiento lc
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
    lc__values = [sqrt(2 * D0 * value / factor) for value in list]
    pl = plot(xlabel=L"(γ^2 G^2 D_0)^{1/3} t", ylabel= "Magnetización " * L"M(t)/M(0)", legend=:best, title = "Magnetización de Hahn")
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
end;

#Test_MHahn()

#------------------------------------------------------------------------------------------

# Función S_hanh(t)
function S_han(lc, σ, N, l0, lf, t)
    l = range(l0, lf, length = N)
    P_l = P.(l,lc, σ) # Consideramos media lc
    M_l = Ml_Hahn.(t, lc) # Calculamos solo para un τc, si se quiere cambiar esto reemplazar lc por l
    S = sum(M_l .* P_l)
    return S
end

#------------------------------------------------------------------------------------------

# Testeamos la función S(t)
function Test_S()
	N = 2000 # Numero de compartimientos
	lc = 3.7 # Tamaño medio del compartimiento
	# Rango de compartimientos simulados
	l0 = 0.05
	lf = 50
	time = range(0, 1, length = 10000) # Tiempo a simular
	σs = [0.01, 0.10, 0.25, 0.50, 1.0] # Distitnos σ
	pl = plot(xlabel="t (s)", ylabel="S(t) (U.A)", legend=:best, title = "Señal " * L"$S_{Hahn}$")
	xlims!(0, 0.015)
	for σ in σs
	    St = S_han.(3.7, σ, 2000, l0, lf, time)
	    plot!(pl, time, St, label = L"$\sigma$" * "= $(σ)")
	end
	display(pl)
end;

#Test_S()

#------------------------------------------------------------------------------------------

function GenData(N, lc, σ, l0, lf, time_sim)
    # Generamos los tiempos de difusión y de tamaños de compartimientos
    t = range(0, time_sim, length = 10000)
    l = range(l0, lf, length = N)
    # Generamos las distribuciones
    P_l = P.(l, lc, σ)
    # Calculamos la señal
    S = S_han.(lc, σ, N, l0, lf, t)

    return t, P_l, S
end

#------------------------------------------------------------------------------------------

# Ahora tenemos un conjunto de pares de datos de entrada y salida.
# Cada valor de t y su correspondiente SHahn(t) es una entrada, y el
# vector P(l) es la salida correspondiente. O puede verse en forma
# inversa tambien. La idea sera encontrar un mapeo que relacione
# ambos conjuntos de datos.
# Veamos los datos que tenémos

function Test_GenData()
    N = 2000 # Numero de compartimientos
	lc = 3.7 # Tamaño medio del compartimiento
	# Rango de compartimientos simulados
	l0 = 0.05
	lf = 50
    t_sim = 1.0 # Tiempo a simular
    σ = 0.5
    t, P_l, S = GenData(N, lc, σ, l0, lf, t_sim)
    println(size(collect(t)))
    println(size(P_l))
    println(size(S))
end

#Test_GenData()

#------------------------------------------------------------------------------------------

# Ahora vamos a generar un conjunto de datos de entrada y salida para distintos σ, en principio estos tambien cambian
# con los numeros de compartimientos N, el tiempo que se simula t_sim, y los tamaños de compartimientos l0 y lf.
# por ahora vamos a dejarlos constantes y variar únicamente σ.
# En lo que entiendo los datos que tenemos son para cada σ una secuencia temporal S(t) que se relaciona
# con una distribución de tamaños P(l). Querriamos ver los datos en 2 dimensiones, para esto querriamos hacer
# una reducción de dimensionalidad con técnicas de Machine Learning.

function GenINOUT(σs, N, t_sim, lc, l0, lf)
    IN = []
    OUT = []
    for σ in σs
        t, P_l, S = GenData(N, lc, σ, l0, lf, t_sim)
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

function Test_GenINOUT()
    σs = 0.05:0.05:1 # Distitnos σ entre 0.01 y 1
    σs = collect(σs)
    N = 2000 # Dejamos fijo el número de compartimientos por ahora
    t_sim = 1.0 # Tiempo a simular lo dejamos fijo
    lc = 3.7 # Tamaño medio del compartimiento lo dejamos fijo
    l0 = 0.05 # Tamaño mínimo del compartimiento lo dejamos fijo
    lf = 7.5 # Tamaño máximo del compartimiento lo dejamos fijo
    l = range(l0, lf, length = N)
    IN, OUT = GenINOUT(σs, N, t_sim, lc, l0, lf)
    println(size(IN))
    println(size(OUT))

    # Entonces consideramos las entradas como un arreglo de una serie temporal S(t)
    # y un arreglo de tiempos t, y las salidas como un arreglo de una distribución de tamaños P(l)
    # Apliquemos reducicón de la dimensionalidad a esto
    pl = plot(xlabel=L"t"*" (s)", ylabel=L"S(t)" *" (U.A)", legend=:best, title = "Señal " * L"S_{Hahn}")
    xlims!(0, 0.015)
    for i in 1:length(IN)
        plot!(pl,IN[i][1], IN[i][2], label = L"$\sigma$" * "= $(σs[i])")
    end

    display(pl)

    pl2 = plot(xlabel = L"l"*" (μm)", ylabel = L"P(l)\:" * "(μm" * L"^{-1}" *")", legend=:best, title = "Distribución " * L"P(l)" * " asociadas a "* L"S(t)")
    for i in 1:length(OUT)
        plot!(pl2, l, OUT[i], label = L"$\sigma$" * "= $(σs[i])")
    end

	display(pl2)

    # Vamos con el PCA para reducir la dimensionalidad
    # Primero tenemos que hacer un reshape de los datos de entrada y salida para que sean un array
    # El primero tendría dimensión length(σs)xlength(time_sim)^2 que en este caso lo tenémos fijado como 10000
    # El segundo tendría dimensión length(σs)xlength(l) que en este caso lo tenémos fijado como 2000
    
end

# Test_GenINOUT()

σs = 0.05:0.01:1 # Distitnos σ entre 0.01 y 1
σs = collect(σs)
N = 2000 # Dejamos fijo el número de compartimientos por ahora
t_sim = 1.0 # Tiempo a simular lo dejamos fijo
lc = 3.7 # Tamaño medio del compartimiento lo dejamos fijo
l0 = 0.05 # Tamaño mínimo del compartimiento lo dejamos fijo
lf = 7.5 # Tamaño máximo del compartimiento lo dejamos fijo
l = range(l0, lf, length = N)
l = collect(l)
IN, OUT = GenINOUT(σs, N, t_sim, lc, l0, lf)
println(size(IN))
println(size(OUT))

# Entonces consideramos las entradas como un arreglo de una serie temporal S(t)
# y un arreglo de tiempos t, y las salidas como un arreglo de una distribución de tamaños P(l)
# Apliquemos reducicón de la dimensionalidad a esto
pl = plot(xlabel=L"t"*" (s)", ylabel=L"S(t)" *" (U.A)", legend=:best, title = "Señal " * L"S_{Hahn}")
xlims!(0, 0.015)
for i in 1:length(IN)
    plot!(pl,IN[i][1], IN[i][2], label = L"$\sigma$" * "= $(σs[i])")
end

display(pl)

pl2 = plot(xlabel = L"l"*" (μm)", ylabel = L"P(l)\:" * "(μm" * L"^{-1}" *")", legend=:best, title = "Distribución " * L"P(l)" * " asociadas a "* L"S(t)")
for i in 1:length(OUT)
    plot!(pl2, l, OUT[i], label = L"$\sigma$" * "= $(σs[i])")
end

display(pl2)

# Vamos con el PCA para reducir la dimensionalidad, primero reducimos la de los datos de entrada
# Ponemos en cada columna la señal S(t) medida para cada σ
# Hacemos lo mismo con los datos de salida

dataIN = zeros(length(IN[1][2]), length(σs))
for i in 1:length(σs)
    dataIN[:,i] = IN[i][2]
end

dataOUT = zeros(length(OUT[1]), length(σs))
for i in 1:length(σs)
    dataOUT[:,i] = OUT[i]
end

dataIN
dataOUT

# Hagamos el PCA para los datos de entrada
pca_modelIN = fit(PCA, dataIN; maxoutdim = 2)  # Reducir a 2 componentes principales

# Transformamos los datos de entrada al espacio de menor dimensión
reduced_dataIN = transform(pca_modelIN, dataIN)

# Obtenemos los componentes principales
pcsIN = principalvars(pca_modelIN)

# Variance explained by each principal component
explained_varianceIN = principalvars(pca_modelIN) / sum(principalvars(pca_modelIN))

# Scatter plot de los datos de entrada originales
scatter(dataIN[:, 1], dataIN[:, 2], label="Original Data", legend=:topleft, xlabel="Feature 1", ylabel="Feature 2", ratio=1)

# Scatter plot de los datos de entrada reducidos
scatter(reduced_dataIN[:, 1], reduced_dataIN[:, 2], label="Reduced Data", legend=:topleft, xlabel="Principal Component 1", ylabel="Principal Component 2", ratio=1)

# Variance explained by each principal component
bar(explained_varianceIN, label="Explained Variance", legend=:topright, xlabel="Principal Component", ylabel="Explained Variance")

# Hagamos el PCA para los datos de salida
pca_modelOUT = fit(PCA, dataOUT; maxoutdim = 2)  # Reducir a 2 componentes principales

# Transformamos los datos de salida al espacio de menor dimensión
reduced_dataOUT = transform(pca_modelOUT, dataOUT)

# Obtenemos los componentes principales
pcsOUT = principalvars(pca_modelOUT)

# Variance explained by each principal component
explained_varianceOUT = principalvars(pca_modelOUT) / sum(principalvars(pca_modelOUT))

# Scatter plot de los datos de salida originales
scatter(dataOUT[:, 1], dataOUT[:, 2], label="Original Data", legend=:topleft, xlabel="Feature 1", ylabel="Feature 2", ratio=1)

# Scatter plot de los datos de salida reducidos
scatter(reduced_dataOUT[:, 1], reduced_dataOUT[:, 2], label="Reduced Data", legend=:topleft, xlabel="Principal Component 1", ylabel="Principal Component 2", ratio=1)

# Variance explained by each principal component
bar(explained_varianceOUT, label="Explained Variance", legend=:topright, xlabel="Principal Component", ylabel="Explained Variance")

# Graficamos en 3D los datos de entrada y salida reducidos
scatter(reduced_dataIN[1, :], reduced_dataOUT[1,:], reduced_dataOUT[2, :], label="Reduced Data", legend=:topleft, xlabel="Principal Component 1", ylabel="Principal Component 2", zlabel="Principal Component 3", ratio=1)

scatter(reduced_dataIN[1, :], reduced_dataOUT[1,:], label="Reduced Data", legend=:topleft, xlabel="Principal Component 1", ylabel="Principal Component 2")
scatter!(reduced_dataIN[1, :], reduced_dataOUT[2,:], label="Reduced Data", legend=:topleft, xlabel="Principal Component 1", ylabel="Principal Component 3")

#------------------------------------------------------------------------------------------

