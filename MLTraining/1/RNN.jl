using Flux
using Flux: train!
using Statistics
using Plots
using Distributions
using ProgressMeter
using Flux: @epochs

i = [1,2,3,4,5,6,7,8,9,10]

ibm_data = exp.(i)# Generate some data
# Convert your data to Float32
ibm_data = Float32.(ibm_data)
# Reshape your data to the general RNN input format in Flux
# Note that this assumes a sequence length of 5000, i.e. the full sequence, which is not necessarily a good idea
X = [[x] for x in ibm_data[1:end-1]]
y = ibm_data[2:end]
# Create the RNN model
myrnn = Chain(RNN(1, 32), Dense(32, 1))
# Choose an optimizer
opt = ADAM(1e-2)
# Keep track of parameters for update
ps = Flux.params(myrnn)
# Define a loss function
function loss(X, y)
    myrnn(X[1]) # Warm up model
    # Compute loss
    mean(abs2(myrnn(x)[1] - y) for (x, y) in zip(X[2:end], y[2:end]))
end

for epoch in 1:10 # Train the RNN for 10 epochs
    Flux.reset!(myrnn) # Reset RNN
    gs = gradient(ps) do # Compute gradients
        loss(X, y)
    end
    Flux.update!(opt, ps, gs) # Update parameters
    println(gs)
end

# Plot predictions

inputs = [[x] for x in i]

plot(myrnn(inputs), label="Prediction")

xs = [x[1][1] for x in X]
ys = [y[1][1] for y in y]

plot(xs)
plot!(ys)


output_size = 5
input_size = 2
Wxh = randn(Float32, output_size, input_size)
Whh = randn(Float32, output_size, output_size)
b   = randn(Float32, output_size)

rnn = Flux.RNNCell(2, 5)

x = rand(Float32, 2) # dummy data
h = rand(Float32, 5)  # initial hidden state

h, y = rnn(h, x)

x = rand(Float32, 2)
h = rand(Float32, 5)

m = Flux.Recur(rnn, h)

y = m(x)

RNN(2, 5)

m = Chain(RNN(2 => 5), Dense(5 => 1))

x = rand(Float32, 2)

m(x)

x = [rand(Float32, 2) for i = 1:3]

y = [m(xi) for xi in x]

using Flux.Losses: mse

function loss(x, y)
  m(x[1]) # ignores the output but updates the hidden states
  sum(mse(m(xi), yi) for (xi, yi) in zip(x[2:end], y))
end

y = [rand(Float32, 1) for i=1:2]

function loss(x, y)
    Flux.reset!(m)
    sum(mse(m(xi), yi) for (xi, yi) in zip(x, y))
  end

loss(x, y)


m = Chain(RNN(1 => 30), Dense(30 => 1))

function loss(x, y)
    sum(mse(m(xi), yi) for (xi, yi) in zip(x, y))
  end
  
seq_init = [[1],
            [2],
            [3]]

size(seq_init)

seq_1 = [[4],
         [5],
         [6]]

seq_2 = [[7],
         [8],
         [9]]
    
y1 = [2*x for x in seq_1]
y2 = [2*x for x in seq_2]

X = [seq_1, seq_2]
Y = [y1, y2]
data = zip(X,Y)

Flux.reset!(m)

[m(x) for x in seq_init]

ps = Flux.params(m)
opt= Adam(1e-3)

for _ in 1:100
    Flux.train!(loss, ps, data, opt)
end

loss(seq_2, y2)

scatter([x[1] for x in seq_1], [x[1] for x in y1], label="Data")
scatter!([x[1] for x in seq_1], [m(x)[1] for x in seq_1], label="Prediction")
scatter!([x[1] for x in seq_2],[x[1] for x in y2],label= "Data")
scatter!([x[1] for x in seq_2], [m(x)[1] for x in seq_2], label="Prediction")