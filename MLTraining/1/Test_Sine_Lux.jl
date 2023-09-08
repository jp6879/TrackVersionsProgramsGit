using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots, ComponentArrays,OptimizationOptimisers,OptimizationFlux
rng = Random.default_rng()
u0 = Float32.([0.0])
datasize = 500
tspan = (0.0f0, 1f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    ω , A = [1.0, 1.0]
    du[1] =  A*cos(2π*ω*t)
end
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = solve(prob_trueode, Tsit5(), saveat = tsteps)
plot(ode_data)
ode_data = Array(ode_data)



dudt2 = Lux.Chain(Lux.Dense(1 => 30, rrelu),
Lux.Dense(30 => 30, rrelu),
Lux.Dense(30 => 25, tanh_fast),
Lux.Dense(25 => 1, tanh_fast))

prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)
p, st = Lux.setup(rng, dudt2)
                


function predict_neuralode(p)
  Array(prob_neuralode(u0, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
callback = function (p, l, pred; doplot = true)
  println(l)
  # plot current prediction against data
  if doplot
    plt = scatter(tsteps, ode_data[1,:], label = "data")
    scatter!(plt, tsteps, pred[1,:], label = "prediction")
    display(plot(plt))
  end
  return false
end
pinit = ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot=true)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob,
                                       ADAM(0.001),
                                       callback = callback,
                                       maxiters = 300)
                                       optprob2 = remake(optprob,u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
                                        Optim.BFGS(initial_stepnorm=0.01),
                                        callback=callback,
                                        allow_f_increases = false)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot=true)