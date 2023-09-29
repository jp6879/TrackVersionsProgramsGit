using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots, ComponentArrays,OptimizationOptimisers,OptimizationFlux
using LineSearches, FFTW, Distributions
rng = Random.default_rng()
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
freq_span = (0.0f0 , 1.0f0)

# Los datos para validar son
F_valid = reshape(Float32.(abs.(Fs_valid)), 1, :)
freq_valid = reshape(Float32.(ωs_valid), 1, :)
max = maximum(freq_valid)
freq_valid = freq_valid./max # Normalizamos las frecuencias de validación

# Creamos el modelo de NODE. Para eso primero hacemos la NN. Usamos una inicialización de pesos no convencional

dudt = Lux.Chain(Lux.Dense(1 => 30, relu),
                    Lux.Dense(30 => 20, relu),
                    Lux.Dense(20 => 15, relu),
                    Lux.Dense(15 => 20, tanh_fast))

# Extreamos los parámetros y los estados

ps, st = Lux.setup(rng, dudt)

# Creamos la NODE

n_ode = NeuralODE(dudt, freq_span, Tsit5(), saveat = 0f0:0.001f0:1f0, 
reltol = 1e-3, abstol = 1e-3)

function predict_neuralode(p)
    Array(n_ode(u0, p, st)[1])
end

u0 = Float32[1.0]
prediction = Array(n_ode(u0, ps, st)[1])

model = Lux.Chain((u0, x, p = n_ode.p) -> n_ode(u0, x, p),
                predict_neuralode,
                Lux.Dense(20, 10),
                Lux.Dense(10, 1)) # Esta f pasa por una capa densa para que la salida sea un número





# Creamos la función loss

function loss_node(p)
    predi = predict_neuralode(p)
    loss = sum(abs2, predi .- F_sim)
    return loss, predi
end

losses = Float32[]

# Creamos la función callback

function callback(params, l, predi)
    push!(losses,l)
    println("Training || Iteration $(length(losses))|| Loss = $l")
    return false
end

pinit = ComponentArray(ps)

callback(pinit, loss_node(pinit)...)

# Entreamos el modelo

adtype = Optimization.AutoZygote() # Tipo de automatic differentiation
optf = Optimization.OptimizationFunction((x, p) -> loss_node(x), adtype) # Función a optimizar
optprob = Optimization.OptimizationProblem(optf, pinit) # Problema de optimización
result_neuralode = Optimization.solve(optprob,
BFGS(; initial_stepnorm=0.01, linesearch=LineSearches.BackTracking());
callback, maxiters=1000)