using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots, ComponentArrays,OptimizationOptimisers,OptimizationFlux
using LineSearches
#Lux tiene en cuenta mucho la inmutabilidad por lo tanto mantiene una semilla aleatoria
rng = Random.default_rng()

# Creamos la función a simular
g(x) = sin(2π*x)*exp(-0.1*x) + cos(4π*x)

# Creamos los datos de entrenamiento
t_train = Float32.(hcat(0:0.001:1...))
y_train = Float32.(g.(t_train))
tspan = (0.0f0 , 1.0f0)
# Creamos el modelo de NODE. Para eso primero hacemos la NN. Usamos una inicialización de pesos no convencional

dudt = Lux.Chain(Base.Fix1(broadcast, cos),
                Lux.Dense(1 => 32, cos; init_weight=truncated_normal(; std=1e-4)),
                Lux.Dense(32 => 32, cos; init_weight=truncated_normal(; std=1e-4)),
                Lux.Dense(32 => 1; init_weight=truncated_normal(; std=1e-4)))

# Extreamos los parámetros y los estados

ps, st = Lux.setup(rng, dudt)

# Creamos la NODE

n_ode = NeuralODE(dudt, tspan, Tsit5(), saveat = 0f0:0.001f0:1f0, 
reltol = 1e-3, abstol = 1e-3)
u0 = Float32[1.0]
prediction = Array(n_ode(u0, ps, st)[1])

function predict_neuralode(p)
    Array(n_ode(u0, p, st)[1])
end

# Creamos la función loss
function loss_node(p)
    predi = predict_neuralode(p)
    loss = sum(abs2, predi .- y_train)
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