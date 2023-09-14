using Flux, Plots, Distributions, SpecialFunctions, FFTW
using Flux: train!
# La función sintética que vamos a considerar es

f(t) = sin(2π*t) + 0.5 * sin(4π*t) + exp(-0.2*t)*cos(2π*t)

# Generamos los datos para un tiempo de 0 a 10 con pasos de 0.1 con ruido gaussiano
t0 = 0f0
t1 = 10f0
t_step = 0.01f0
time_sequence = hcat(t0:t_step:t1...)

ys = []

for _ in range(1,100)
    y_sequence = f.(time_sequence) + reshape(rand(Normal(0, 0.1), length(time_sequence)), 1, :)
    ys = vcat(ys, y_sequence)
end

ys = Float32.(ys)

pl = plot(time_sequence[1,:], ys[1,:])
for i in range(2,100)
    plot!(pl,time_sequence[1,:], ys[i,:])
end
pl

# Graficamos los datos
# plot(time_sequence[1,:], y_sequence[1,:], label="Datos", xlabel="Tiempo", ylabel="f(t)", title="Datos sintéticos")

# Computamos 100 transformadas de Fourier
fs = 1/(t_step) # Frecuencia de muestreo
Fs = []
ωs = []
for i in range(1,100)
    F = fftshift(fft(ys[i,:]))[475:525]
    ω = fftshift(fftfreq(length(time_sequence[1,:]), fs))[475:525]
    Fs = vcat(Fs, F)
    ωs = vcat(ωs, ω)
end

Fs
ωs

plot(ωs, abs.(Fs), label="Transformada de Fourier", xlabel="Frecuencia", ylabel="|F(f)|", title="Transformada de Fourier")

# F = fftshift(fft(y_sequence[1,:]))[450:550]
# ω = fftshift(fftfreq(length(time_sequence[1,:]), fs))[450:550]

# Graficamos la transformada de Fourier
scatter(ω, abs.(F), label="Transformada de Fourier", xlabel="Frecuencia", ylabel="|F(f)|", title="Transformada de Fourier")

# Ahora los nuevos datos que querémos simular son
F_sim = reshape(Float32.(abs.(Fs)), 1, :)
freq_sim = reshape(Float32.(ωs), 1, :)

#scatter(freq_sim[1,:], F_sim[1,:], label="Datos de entrenamiento", xlabel="Frecuencia", ylabel="|F(f)|", title="Datos de entrenamiento")

freq_sim = freq_sim./maximum(freq_sim)

#scatter(freq_sim[1,:], F_sim[1,:], label="Datos de entrenamiento", xlabel="Frecuencia", ylabel="|F(f)|", title="Datos de entrenamiento")

# Cargmamos los datos en el loader de Flux

train_loader = Flux.DataLoader((freq_sim, F_sim), batchsize=101, shuffle=true)

# Definimos nuestro modelo de red neuronal

model = Chain(Dense(1, 75, relu), Dense(75, 30, relu), Dense(30, 10, relu), Dense(10, 1))

# Definimos la función de loss, vamos a seguir usando MSE

loss(x,y) = mean(abs2.(model(x) .- y))

# Definimos el método de optimización

opt = ADAM(0.005)

# Hacemos la función de callback para guardar el loss en cada iteración

lossNN = []
iter = 0
epoch_iter = 0
cb = function()
    global iter
    global epoch_iter
    iter += 1
    # Record Loss
    if iter % length(train_loader) == 0
        epoch_iter += 1
        actual_loss = loss(train_loader.data[1], train_loader.data[2])
        if epoch_iter % 5 == 0
            println("Epoch $epoch_iter || Loss = $actual_loss")
        end
        push!(lossNN, actual_loss)
    end
end

epochs = 2000
for _ in 1:epochs
    Flux.train!(loss, Flux.params(model, opt), train_loader, opt, cb = cb)
end

# Graficamos los datos de entrenamiento y la predicción de la red neuronal

scatter(freq_sim[1,:], F_sim[1,:], label="Datos de entrenamiento", xlabel="Frecuencia", ylabel="|F(f)|", title="Datos de entrenamiento")
scatter!(freq_sim[1,:], model(freq_sim)[1,:], label="Predicción de la red neuronal", xlabel="Frecuencia", ylabel="|F(f)|", title="Predicción de la red neuronal")


