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
# Veamos como se ve esta función

display(plot(t_train[1,:], y_train[1,:], label = "g(t)", xlabel = "t", ylabel = "g(t)", title = "Función a simular", legend = :bottomright))

# Creamos la funcion diferenciable con una red neuronal, vamos a probar usar Lux ya que dice que se recomienda por cuestiones de velocidad
dudt = Flux.Chain(Flux.Dense(31 => 30, tanh_fast),
            Flux.Dense(30 => 20, tanh_fast),
            Flux.Dense(20 => 15, tanh_fast),
            Flux.Dense(15 => 31, tanh_fast)) # Creamos el modelo que va a ser nuestra función diferenciada

diffeqarray_to_array(x) = reshape(x, size(x)[1:2]) # Esto acomoda la solución de la EDO en un arreglo de 2 dimensiones 21 x length(trange)

n_ode = NeuralODE(dudt, tspan, Tsit5(), save_everystep = false,
reltol = 1e-3, abstol = 1e-3, save_start = false)
n_ode = AugmentedNDELayer(n_ode, 30)
ps = n_ode.p
model = Chain((x, p = n_ode.p) -> n_ode(x, p), # En primer lugar manda el input a la red neuronal y luego los parámetros
                Array,  # Lo que devuelve la NODE es la solución desde t0 a t1 y devuelve f en cada paso de tiempo
                diffeqarray_to_array, # Esto solo deja la matriz 21x201
                Dense(31, 1)) # Esta f pasa por una capa densa para que la salida sea un número

data = Flux.DataLoader((t_train, y_train), batchsize = 1001, shuffle = true)
loss_node(x, y) = mean((model(x) .- y) .^ 2)

accuraccy_train = []
loss_train = []


function accuracy(y)
    num_correct = 0
    predictions = model(t_train)
    for i in 1:length(predictions)
        if abs(predictions[1,i] - y[1,i]) < 0.1
            num_correct += 1
        end
    end
    return (num_correct/length(predictions)) * 100.0
end

opt = ADAM(0.005)
iter = 0
cb = function()
    global iter
    iter += 1
    if iter % length(data) == 0
        actual_loss = loss_node(data.data[1], data.data[2])
        println("Iteration $iter || Loss = $actual_loss")
        push!(loss_train, actual_loss)
        push!(accuracy_train, accuracy(y_train))
    end
end

for _ in 1:150
    Flux.train!(loss_node, Flux.params(ps, model), data, opt, cb = cb)
end

plot1 = scatter(t_train[1,:],y_train[1,:], label="Train data")
scatter!(t_train[1,:],model(t_train)[1,:],label = "predicción")

display(plot1)

display(plot(loss_train, label="train loss",xlabel = "Epochs", ylabel = "Loss", title = "Loss on train vs Epochs"))

println("Maximum accuracy on train: ", maximum(accuracy_train[250:end]), "%")
display(plot(accuracy_train, label="train accuracy",xlabel = "Epochs", ylabel = "Accuracy", title = "Accuracy on train vs Epochs"))
