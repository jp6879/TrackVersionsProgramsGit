using Flux, Statistics, Plots, Distributions, FFTW
using Flux: train!
using Flux: mse
using Random

function data_maker(x, len_seq)
    N = length(x)
    X = []
    while N > 0
        if (N - len_seq) <= 0
            break
        end
        vec = [[xi] for xi in x[N - len_seq:N]] # Esto lo que crea es un vector de secuencias de tamaño len_seq 
        push!(X, vec) # Almacenamos la secuencia en X
        N -= len_seq + 1 
    end 
    # Una vez finalizado esto devolvemos los valores de X y Y que son vectores de secuencias.
    return Vector{Vector{Float32}}.(X)
end

# La función sintética que vamos a considerar es
f(t) = sin(2π*t) + 0.5 * sin(4π*t) + exp(-0.2*t)*cos(2π*t)

# Generamos los datos para un tiempo de 0 a 10 con pasos de 0.1 con ruido gaussiano
t0 = 0f0
t1 = 10f0
t_step = 0.01f0
time_sequence = hcat(t0:t_step:t1...)

# Veamos la señal generada

ys = []

# Generamos 100 secuencias de datos
y_sequence = f.(time_sequence) + reshape(rand(Normal(0, 0.1), length(time_sequence)), 1, :)
for _ in range(1,2000)
    global y_sequence = f.(time_sequence) + reshape(rand(Normal(0, 0.1), length(time_sequence)), 1, :)
    global ys = vcat(ys, y_sequence)
end

# Pasamos a Float32
ys = Float32.(ys)

pl = plot(time_sequence[1,:], ys[1,:], label = false, xaxis = "Tiempo", yaxis = "f(t)", title = "Secuencias de datos")

for i in range(2,2000)
    plot!(pl, time_sequence[1,:], ys[i,:], label = false)    
end

pl

# Generamos los datos de entrenamiento

fs = 1/(t_step) # Frecuencia de muestreo

Fs = []
ωs = []

for i in range(1,2000)
    y = reshape(ys[i,:],1, :)
    F = rfft(y[1,:])
    ω = rfftfreq(length(time_sequence[1,:]), fs)
    push!(Fs, F)
    push!(ωs, ω)
end

# Grafiquemos estas transformadas de Fourier
pl2 = plot(ωs[1], abs.(Fs[1]), label = false, xaxis = "ω", yaxis = "|F(ω)|", title = "Transformada de Fourier")
for i in range(2,2000)
   plot!(pl2,ωs[i], abs.(Fs[i]), label = false)
end
pl2


# Los datos a simular serían
F_sim = []
freq_sim = []

for i in range(1,2000)
    push!(F_sim, abs.(Fs[i])[1:50])
    push!(freq_sim, ωs[i][1:50])
end

# Ponemos los vectores en Float32 y pasamos a una matrix de 100x30
F_sim = Vector{Float32}.(F_sim)
freq_sim = Vector{Float32}.(freq_sim)

# Veamos los datos a simular
pl3 = plot(freq_sim[1], F_sim[1], xlabel = "Frecuencia", ylabel = "|F(ω)|", title = "Datos a simular", label = false)
for i in range(2,2000)
    scatter!(pl3, freq_sim[i], F_sim[i], label = false)
end
pl3    

# Creamos las secuencias de datos
length_sequence = 25

datas = []

for i in range(1,2000)
    X = data_maker(freq_sim[i], length_sequence)
    Y = data_maker(F_sim[i], length_sequence)
    tuple = zip(X,Y)
    push!(datas, tuple)
end

# X = data_maker(freq_sim[end], length_sequence)
# Y = data_maker(F_sim[end], length_sequence)
# data = zip(X, Y)


# Vamos a entrenar con estos datos
# F_sim = Float32.(abs.(F))[1:30]
# freq_sim = Float32.(ω)[1:30]


# Veamos los datos a simular
# scatter(freq_sim, F_sim, label="|F(ω)|", xlabel="ω", ylabel="|F(ω)|", title="Datos a simular", legend=:topleft)

# length_sequence = 12

# X = data_maker(F_sim, length_sequence)
# Y = data_maker(freq_sim, length_sequence)

# Agrupamos los datos
# data = zip(X, Y)

# Creamos el modelo de red neuronal

modelRNN = Chain(
    GRU(1, 200),
    GRU(200, 150),
    Dense(150, 100),
    LSTM(100, 50),
    LSTM(50, 10),
    Dense(10, 1)
)

Flux.reset!(modelRNN)

# Definimos la función de pérdida

function loss(x, y)
    loss = sum(Flux.mse(modelRNN(xi),yi) for (xi, yi) in zip(x,y))
    return loss
end

ps = Flux.params(modelRNN)
opt = Adam(1e-6)

iter = 0
epoch_iter = 0
lossRNN = []

cb = function()
    global iter
    global epoch_iter
    iter += 1
    # Record Loss
    if iter % length(datas[1]) == 0
        epoch_iter += 1
        actual_loss = 0
        for (x, y) in datas[1]
            actual_loss += loss(x, y)
        end
        println("Epoch $epoch_iter || Loss = $actual_loss")
        push!(lossRNN, actual_loss)
    end
end

for _ in 1:2
    for data in datas
        Flux.train!(loss, ps, data, opt, cb = cb)
        Flux.reset!(modelRNN)
    end
end

# Frezee the model
# opt_setup = Flux.setup(Adam(1e-5),modelRNN)
# Flux.freeze!(opt_setup)

# Veamos la predicción
V = [[freq_sim[1][i]] for i in range(1, length(freq_sim[1]))]
prediction = reduce(vcat,[modelRNN(x) for x in V])

# Flux.train!(loss, ps, data, opt, cb = cb)
scatter(freq_sim[1], F_sim[1])
scatter!(freq_sim[1], prediction)