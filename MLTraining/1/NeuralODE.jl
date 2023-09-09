# From scratch Neural networks and Neural ODEs
### ============================================================================
### Layer Struct
### ============================================================================

# A Layer consists of weights, bias, and an activation function to produce the
# output given the input. Additionally, we also save the result before applying
# the activation function to compute the backward pass.
struct Layer{WS, BS, Z, F}
    W::WS # weights
    b::BS # biases
    z::Z # intermediate state
    σ::F # activation function
end

# The Layer constructor takes in the dimensions of the input, output, and an
# activation function. Weight and biases are set accordingly, and an empty array
# is set up to store the intermediate state.
Layer(in::Int, out::Int, σ::Function) =
    Layer(rand(Float32, out, in) .- 0.5f0,  # weights
        zeros(Float32, out),                # biases
        Array{Float32}[],                   # intermediate state vector
        σ)                                  # activation function

# Layer output is computed using the formula σ(W * X + b)
function (l::Layer)(X)
    W, b, z, σ = l.W, l.b, l.z, l.σ
    temp = W * X .+ b             # .+ broadcasting is also compatible with batches
    empty!(z)
    push!(z, temp)  # store intermediate state for back propagation
    return σ.(temp) # apply the activation function element-wise
end
# Layer is updated with partial derivatives and learning rate.
function update!(l::Layer, dW, db, η)
    l.W .-= η * dW
    l.b .-= η * db
end

# Given the derivative of the Cost wrt the ouput, we calculate the partial
# derivatives with respect to weights, biases, and the input.
function derive(l::Layer, ∂Cost∂a_out, a_in)
    dσ = derive(l.σ) # user-defined derivative function of σ

    # Cost wrt intermediate result
    ∂Cost∂z = ∂Cost∂a_out .* dσ.(l.z[1])
    
    # ∂W computes Cost wrt weights for one pair of input and output
    ∂W(∂Cost∂z, a_in) = ∂Cost∂z * a_in'
    
    # Cost wrt weights for entire batch
    ∂Cost∂W = sum(∂W.(eachcol(∂Cost∂z), eachcol(a_in)))
    
    # Cost wrt to bias
    ∂Cost∂b = sum(eachcol(∂Cost∂z))
    
    # Cost wrt input from last layer
    ∂Cost∂a_in = l.W' * ∂Cost∂z
    return ∂Cost∂W, ∂Cost∂b, ∂Cost∂a_in
end

# Back propagation given the input from the previous layer
# and the cost gradient wrt this layer's output
function back_propagate!(l::Layer, ∂Cost∂a_out, a_in, η)
    ∂Cost∂W, ∂Cost∂b, ∂Cost∂a_in = derive(l, ∂Cost∂a_out, a_in) # gradients
    update!(l, ∂Cost∂W, ∂Cost∂b, η) # update parameters
    return ∂Cost∂a_in # Cost wrt input from last layer
end

### ============================================================================
### Model Struct
### ============================================================================

# A Model consists of multiple Layers. Additionally, we store each Layer's
# outputs in an array for the backward pass.
struct Model{LS, OS}
    layers::LS # Layers
    a::OS # Layer outputs
    # Model Constructor
    Model(layers...) = new{typeof(layers), Vector{Array{Float32}}}(layers, [])
end

# Evaluate Model by evaluating the layers sequentially.
function (m::Model)(X)
    # Store Model input
    empty!(m.a)
    push!(m.a, X)
    # Evaluate each layer and store their outputs
    for layer in m.layers
        push!(m.a, layer(m.a[end]))
    end
    # Return Model output
    return pop!(m.a)
end
    # Back-propagate through the Model by back-propagating through each layer.
function back_propagate!(m::Model, ∂Cost∂aL, η)
    # Back propagate through each layer
    ∂Cost∂a_out = ∂Cost∂aL
    for layer in reverse(m.layers)
        a_in = pop!(m.a) # retrieve layer input
        ∂Cost∂a_out = back_propagate!(layer, ∂Cost∂a_out, a_in, η)
    end
end

# Training requires a Model, a Cost function, the training dataset, and the
# learning rate.
function train!(m::Model, Cost, dataset, η)
    costs = Float32[] # store cost of each batch in dataset
    dCost = derive(Cost) # user-defined derivative function of Cost
    # Train Model on each batch in dataset
    for batch in dataset
        X, Y = batch
        out = m(X)
        # Calculate cost
        cost = Cost(out, Y)
        push!(costs, cost)
        # Back propagation
        ∂Cost∂out = dCost(out, Y)
        back_propagate!(m, ∂Cost∂out, η)
    end
# Return average cost of all batches
    return sum(costs) / length(dataset)
end

using DifferentialEquations, Flux, Zygote

# Extend Zygote to work with Neural ODE
function Zygote._zero(xs::AbstractArray{<:AbstractArray}, T=Any)
    return [Zygote._zero(x) for x in xs]
end

### ============================================================================
### Neural ODE Layer Struct and Constructor
### ============================================================================
# A Neural ODE consists of a function f(z, θ, t) that models z's derivative, and
# the parameters `θ` to be optimized, and the time span of the integral.
# Additionally, we also store the ODE solver's solution, since it is useful for
# the backward pass via the adjoint method.
struct NeuralODE{F, P, T, S}
    f::F # derivative model
    θ::P # vector of parameters
    tspan::T # time span [t0, t1]
    sol::S # vector of ODE solution
end

# We store the ODE solver's solution in a vector instead of directly to make the
# `NeuralODE` struct immutable for better performace. At initialization, this is
# simply an empty vector.
function NeuralODE(f, θ, tspan)
    return NeuralODE(f, θ, tspan, DiffEqBase.AbstractODESolution[])
end

### ============================================================================
### Flux compatibility
### ============================================================================

# Using the macro `Flux.@functor` allows the machine learning library Flux to
# mix our `NeuralODE` layer in any model.
Flux.@functor NeuralODE

# We also specify the parameters `θ` to be optimized with `Flux.trainable`. We
# only update `θ` by default, but we can also optimize the time span.
Flux.trainable(node::NeuralODE) = (node.θ,)

### ============================================================================
### Forward pass
### ============================================================================

# The forward pass computes the integration with the ODE solver. The forward
# pass returns an array of the solution at each timestep.
function (node::NeuralODE)(z_t0; alg=Tsit5(), kwargs...)
    f, θ, t0, t1, sol = node.f, node.θ, node.tspan[1], node.tspan[2], node.sol
    return forward!(z_t0, θ, t0, t1; f=f, sol=sol, alg=alg, kwargs...)
end

# Integrate from `t0` to `t1` to calculate `z` at `t1`, also returns `z` at
# each timestep in a vector.
function forward!(z_t0, θ, t0, t1; f, sol, alg, kwargs...)

    # Define and solve ODE problem
    function dzdt(dz, z, θ, t)
        dz .= f(z, θ, t)
    end
    problem = ODEProblem(dzdt, z_t0, (t0, t1), θ)
    solution = solve(problem, alg; kwargs...)
    
    # Store the solution for the backward pass
    empty!(sol)
    push!(sol, solution)
    
    # Return an array of `z` evaluated at each timestep
    return solution.u
end

### ============================================================================
### Backward pass
### ============================================================================

# Since back-propagating through the ODE solver is complex, we define a custom
# backward pass for the Neural ODE via the adjoint method. Flux relies on the
# Zygote library to calculate gradients, and we can define our custom gradient
# via `Zygote.@adjoint`.

Zygote.@adjoint function forward!(z_t0, θ, t0, t1; f, sol, alg, kwargs...)
    # Forward pass
    zs = forward!(z_t0, θ, t0, t1; f=f, sol=sol, alg=alg, kwargs...)
    # Return the forward pass and how to calculate the gradients of the loss wrt
    # `z_t0` and `θ` from the gradient of the loss wrt `z` at each timestep.
    return zs, ∂L∂zs -> backward(∂L∂zs, θ; f=f, sol=sol[1], alg=alg)
end

# Compute the gradients of the loss wrt to `θ`.
function backward(∂L∂zs, θ; f, sol, alg)
    # Calculate the partial derivatives from each relevant `∂L∂z`
    idxs = .!(iszero.(∂L∂zs)) |> collect
    t0 = sol.t[1]
    t1s = sol.t[idxs]
    ∂s = _backward.(∂L∂zs[idxs], Ref(θ), t0, t1s; f=f, sol=sol, alg=alg)
    # Aggregate all partial derivatives
    ∂L∂t1 = ∂s[end][end]
    ∇ = map(+, [∂[1:3] for ∂ in ∂s]...)
    return (∇..., ∂L∂t1)
end

# Given the gradient of the loss wrt `z` at time `t1`, compute the partial
# derivatives wrt `z_t0` and `θ` via the adjoint method.
function _backward(∂L∂z_t1, θ, t0, t1; f, sol, alg)
    # Derivative of the loss wrt `t1`
    ∂L∂t1 = ∂L∂z_t1[:]' * f(sol[end], θ, t1)[:]

    # We define the initial augmented state, which consists of the gradients of
    # the loss wrt to `z_t1` and `θ` and `t1`. `ArrayPartition` from the
    # DifferentialEquations library allows us to combine arrays with different
    # dimensions for a single call to the ODE solver.
    s_t1 = ArrayPartition(∂L∂z_t1, zero(θ), [-∂L∂t1])
    
    # Define the dynamics of the augmented state
    function dsdt(ds, s, θ, t)
        # Compute the Jacobian matrices of `f` wrt `z`, `θ`, and `t`
        _, back = Zygote.pullback(f, sol(t), θ, t)
        # Adjoint dynamics
        d = back(-s.x[1])
        # Zygote returns `nothing` as a strong zero if the function is not
        # dependent on the variable, so we convert to zero for computation
        get_derivative(∆, x) = (∆ == nothing ? zero(x) : ∆)
        ∆s = get_derivative.(d, ds.x[:])
        # Return the derivatives
        for i in 1:3
            ds.x[i] .= ∆s[i]
        end
    end
    
    # Solve ODE backwards
    problem = ODEProblem(dsdt, s_t1, (t1, t0), θ)
    solution = solve(problem, alg)
    s_t0 = solution[end]

    # Return gradients
    return (s_t0.x[1], s_t0.x[2], -s_t0.x[3][1], ∂L∂t1)
end