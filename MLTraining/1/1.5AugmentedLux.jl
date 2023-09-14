using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots, ComponentArrays,OptimizationOptimisers,OptimizationFlux
using LineSearches, JLD2, MLUtils, Zygote
#Lux tiene en cuenta mucho la inmutabilidad por lo tanto mantiene una semilla aleatoria
rng = Random.default_rng()

# Creamos la función a simular
g(x) = sin(2π*x)*exp(-0.1*x) + cos(4π*x)

# Creamos los datos de entrenamiento
t_train = Float32.(hcat(0:0.001:2...))
y_train = Float32.(g.(t_train))
tspan = (0.0f0 , 2.0f0)
# Creamos el modelo de NODE. Para eso primero hacemos la NN. Usamos una inicialización de pesos no convencional

dudt = Lux.Chain(Base.Fix1(broadcast, cos),
                Lux.Dense(31 => 32, cos; init_weight=truncated_normal(; std=1e-4)),
                Lux.Dense(32 => 32, cos; init_weight=truncated_normal(; std=1e-4)),
                Lux.Dense(32 => 31; init_weight=truncated_normal(; std=1e-4)))

diffeqarray_to_array(x) = reshape(x, size(x)[1:2])

# Creamos la NODE

n_ode = NeuralODE(dudt, tspan, Tsit5(), saveat = 0f0:0.001f0:2.0f0, 
reltol = 1e-3, abstol = 1e-3)
n_ode = DiffEqFlux.AugmentedNDELayer(n_ode, 30)

model = Lux.Chain((x, p = n_ode.p) -> n_ode(x, p), # En primer lugar manda el input a la red neuronal y luego los parámetros
                diffeqarray_to_array, # Esto solo deja la matriz 21x201
                Lux.Dense(31, 1)) # Esta f pasa por una capa densa para que la salida sea un número

data = DataLoader((t_train, y_train), batchsize = 1001, shuffle = true)

opt = Optimisers.ADAM(0.005f0)

losses = Float32[]

model()

function loss_function(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
    return sum(abs2, y_pred .- y)
end

# Creamos la función callback
function callback(params, l, predi)
    push!(losses,l)
    println("Training || Iteration $(length(losses))|| Loss = $l")
    return false
end

opt = ADAM(0.005f0)
ps, st = Lux.setup(rng, model)

Lux.apply(model, data.data[1], ps, st)

pinit = ComponentArray(ps)

callback(pinit, loss_function(pinit)...)

# Entreamos el modelo
adtype = Optimization.AutoZygote() # Tipo de automatic differentiation
optf = Optimization.OptimizationFunction((x, p) -> loss_node(x), adtype) # Función a optimizar
optprob = Optimization.OptimizationProblem(optf, pinit) # Problema de optimización
result_neuralode = Optimization.solve(optprob,
BFGS(; initial_stepnorm=0.01, linesearch=LineSearches.BackTracking());
callback, maxiters=1000)