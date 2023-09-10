using Flux
using DifferentialEquations
using DiffEqFlux
using Plots
using Flux: train!
using Distributions
using DiffEqFlux, DifferentialEquations
using Statistics, LinearAlgebra, Plots
using Flux.Data: DataLoader, Flux
using Optimization, OptimizationOptimJL
using OptimizationFlux, Random

g(t) = sin(2π*t)*exp(-0.1*t) + cos(4π*t) # Función a simular

t_train = Float32.(hcat(0:0.001:2...))
y_train = Float32.(g.(t_train))
trange = t_train[1,:]
tspan = (t_train[1], t_train[end])

data = Flux.DataLoader((t_train |> gpu, y_train |> gpu), batchsize = 1001, shuffle = true)

diffeqarray_to_array(x) = reshape(gpu(x), size(x)[1:2]) # Esto acomoda la solución de la EDO en un arreglo de 2 dimensiones 21 x length(trange)

n_ode = NeuralODE(Chain(Dense(51 => 60, tanh_fast),
                        Dense(60 => 40, tanh_fast),
                        Dense(40 => 30, tanh_fast),
                        Dense(30 => 15, tanh_fast),
                        Dense(15 => 10, tanh_fast),
                        Dense(10 => 51)) |> gpu,
                  tspan, Tsit5(), save_everystep = false,reltol = 1e-4, abstol = 1e-4, save_start = false) |> gpu

n_ode = AugmentedNDELayer(n_ode, 50)

model = Chain((x, p = n_ode.p) -> n_ode(x, p), # En primer lugar manda el input a la red neuronal y luego los parámetros
                Array,  # Lo que devuelve la NODE es la solución desde t0 a t1 y devuelve f en cada paso de tiempo
                diffeqarray_to_array, # Esto solo deja la matriz 21x201
                Dense(51, 1) |> gpu) # Esta f pasa por una capa densa para que la salida sea un número

ps = n_ode.p |> gpu # Extraemos los parámetros de la NODE

loss_node(x, y) = mean((model(x) .- y) .^ 2) # Creamos la función loss

accuraccy_train = [] |> gpu # Vector para acumular la precisión
loss_train = [] |> gpu # Vector para acumular el loss

# Función que calcula la precisión
# function accuracy(y)
#     num_correct = 0 # Contador del número de predicciones correctas
#     predictions = model(t_train) |> gpu# Predicciones
#     for i in 1:length(predictions)
#         if abs(predictions[1,i] - y[1,i]) < 0.1 # Si la predicción es correcta (diferencia menor a 0.1)
#             num_correct += 1 # Se suma 1 al contador
#         end
#     end
#     return (num_correct/length(predictions)) * 100.0 # Se devuelve en porcentaje la precisión 
# end

opt = ADAM(0.005) # Método de Optimization
iter = 0 # Itera en los entrenamientos de la red

# Función de callback para llevar la cuenta del loss y guardar los datos en los vectores
cb = function()
    global iter
    iter += 1
    if iter % length(data) == 0
        actual_loss = loss_node(data.data[1], data.data[2])
        println("Iteration $iter || Loss = $actual_loss")
        push!(loss_train, actual_loss)
        # push!(accuracy_train, accuracy(y_train))
    end
end

# Entrenamiento hasta tantas épocas

for _ in 1:2000
    Flux.train!(loss_node, Flux.params(ps, model), data, opt, cb = cb)
end

# Veamos los resultados
plot1 = scatter(t_train[1,:],y_train[1,:], label="Train data")
scatter!(t_train[1,:],model(data.data[1])[1,:],label = "predicción")

display(plot1)

display(plot(loss_train, label="train loss",xlabel = "Epochs", ylabel = "Loss", title = "Loss on train vs Epochs"))

println("Maximum accuracy on train: ", maximum(accuracy_train[250:end]), "%")
display(plot(accuracy_train, label="train accuracy",xlabel = "Epochs", ylabel = "Accuracy", title = "Accuracy on train vs Epochs"))
