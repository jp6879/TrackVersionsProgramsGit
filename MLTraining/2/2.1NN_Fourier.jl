using Flux, Plots, Distributions, SpecialFunctions, FFTW
using Flux: train!

# La función sintética que vamos a considerar es
f(t) = sin(2π*t) + 0.5 * sin(4π*t) + exp(-0.2*t)*cos(2π*t)

# Generamos los datos para un tiempo de 0 a 10 con pasos de 0.1 con ruido gaussiano
t0 = 0f0
t1 = 10f0
t_step = 0.01f0
time_sequence = hcat(t0:t_step:t1...)

# Generar una única secuencia de datos es insuficiente ya que tendríamos pocos datos en la región de interes
# de frecuencias
ys = []

# Generamos 100 secuencias de datos
for _ in range(1,100)
    y_sequence = f.(time_sequence) + reshape(rand(Normal(0, 0.1), length(time_sequence)), 1, :)
    ys = vcat(ys, y_sequence)
end
# Las pasamos a Float32
ys = Float32.(ys)

# Veamos como se ven estas secuencias de datos
pl = plot(time_sequence[1,:], ys[1,:], label = false, xaxis = "Tiempo", yaxis = "f(t)", title = "Secuencias de datos")
for i in range(2,100)
    plot!(pl,time_sequence[1,:], ys[i,:], label = false)
end
pl

# Podemos tambien generar datos de validación para este modelo
ys_valid = []
for _ in range(1,30)
    y_sequence = f.(time_sequence) + reshape(rand(Normal(0, 0.1), length(time_sequence)), 1, :)
    ys_valid = vcat(ys_valid, y_sequence)
end
ys_valid = Float32.(ys_valid)

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

# Ploteamos una única de las transfomadas de Fourier
plot(ωs[1:51], abs.(Fs)[1:51], label="Transformada de Fourier", xlabel="Frecuencia", ylabel="|F(f)|", title="Transformada de Fourier")

# Ahora lo mismo para los datos de validación
Fs_valid = []
ωs_valid = []
for i in range(1,30)
    F = fftshift(fft(ys_valid[i,:]))[475:525]
    ω = fftshift(fftfreq(length(time_sequence[1,:]), fs))[475:525]
    Fs_valid = vcat(Fs_valid, F)
    ωs_valid = vcat(ωs_valid, ω)
end

# Ahora los nuevos datos que querémos simular son
F_sim = reshape(Float32.(abs.(Fs)), 1, :)
freq_sim = reshape(Float32.(ωs), 1, :)
freq_sim = freq_sim./maximum(freq_sim) # Normalizamos las frecuencias de entrenamiento

# Los datos para validar son
F_valid = reshape(Float32.(abs.(Fs_valid)), 1, :)
freq_valid = reshape(Float32.(ωs_valid), 1, :)
max = maximum(freq_valid)
freq_valid = freq_valid./max # Normalizamos las frecuencias de validación

# Cargmamos los datos de entrenamiento en el loader de Flux

train_loader = Flux.DataLoader((freq_sim, F_sim), batchsize=51, shuffle=true)

# Definimos nuestro modelo de red neuronal

model = Chain(Dense(1, 75, relu), Dense(75, 30, relu), Dense(30, 10, relu), Dense(10, 1))

# Definimos la función de loss, vamos a seguir usando MSE

loss(x,y) = mean(abs2.(model(x) .- y))

# Definimos el método de optimización
opt = ADAM(0.001)

# Definamos tambien una función para ver la presición de la red neuronal
function accuracy(x,y)
    N_correctos = 0
    N_total = 51
    for i in range(1,N_total)
        println()
        if abs(x[i] - y[i]) < 0.2 # Donde debería estar el 95% de los datos
            N_correctos += 1
        end
    end
    return N_correctos/N_total * 100 # Devuelve el porcentaje de aciertos
end

# Hacemos la función de callback para guardar el loss en cada iteración tanto para
# entrenamiento como para validación

lossNN = []
lossNN_valid = []

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
        if epoch_iter % 1 == 0
            println("Epoch $epoch_iter || Loss = $actual_loss")
        end
        push!(lossNN, actual_loss)
        push!(lossNN_valid, loss(freq_valid, F_valid))
    end
end

epochs = 2000
for _ in 1:epochs
    Flux.train!(loss, Flux.params(model, opt), train_loader, opt, cb = cb)
end

# Como tenemos 100 secuencias vamos a ver cual de estas tiene mejor presición
accuracys = []
i = 1

for _ in range(1,50)
    predict = model(freq_sim)[1,i:i+50]
    F_actual = F_sim[1,i:i+50]
    push!(accuracys, accuracy(predict, F_actual))
    i += 51
end

# Vemos que la secuencia 1 es la que tiene mejor presición
findmax(accuracys)
idxmax = findmax(accuracys)[2] 
init = 1 + idxmax*51
final = 51 + idxmax*51

predict = model(freq_sim)[1,init:final]

freq_sim = freq_sim.*max

# Extraemos los datos de esta secuencia
plot(freq_sim[1,init:final], F_sim[1,init:final])
plot!(freq_sim[1,init:final],predict)

# Graficamos los datos de entrenamiento y la predicción de la red neuronal
scatter(freq_sim[1,:], F_sim[1,:], label="Datos de entrenamiento", xlabel="Frecuencia", ylabel="|F(f)|", title="Datos de entrenamiento")
scatter!(freq_sim[1,:], model(freq_sim)[1,:], label="Predicción de la red neuronal", xlabel="Frecuencia", ylabel="|F(f)|", title="Predicción de la red neuronal")

# Ahora lo mismo con los datos de validación y la predicción de la red neuronal
scatter(freq_valid[1,:], F_valid[1,:], label="Datos de validación", xlabel="Frecuencia", ylabel="|F(f)|", title="Datos de validación")
scatter!(freq_valid[1,:], model(freq_valid)[1,:], label="Predicción de la red neuronal", xlabel="Frecuencia", ylabel="|F(f)|", title="Predicción de la red neuronal")

# Veamos el loss en funcion de la épocas para los datos de entrenamiento y validación
plot(lossNN, label="Loss de entrenamiento", xlabel="Épocas", ylabel="Loss", title="Loss de entrenamiento")
plot!(lossNN_valid, label="Loss de validación", xlabel="Épocas", ylabel="Loss", title="Loss de validación")
