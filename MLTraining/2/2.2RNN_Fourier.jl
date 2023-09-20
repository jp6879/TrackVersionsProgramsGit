using Flux, Statistics, Plots, Distributions, FFTW
using Flux: train!
using Flux: mse
using Random

function data_maker(x, len_seq)
    N = length(x)
    X = []
    while N > 0
        if (N - len_seq) <= 0
            break
        end
        vec = [[xi] for xi in x[N - len_seq:N]] # Esto lo que crea es un vector de secuencias de tamaño len_seq 
        push!(X, vec) # Almacenamos la secuencia en X
        N -= len_seq + 1 
    end 
    # Una vez finalizado esto devolvemos los valores de X y Y que son vectores de secuencias.
    return Vector{Vector{Float32}}.(X)
end

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

# Computamos 100 transformadas de Fourier
fs = 1/(t_step) # Frecuencia de muestreo
Fs = []
ωs = []
for i in range(1,100)
    F = rfft(ys[i,:])[1:51]
    ω = rfftfreq(length(time_sequence[1,:]), fs)[1:51]
    Fs = vcat(Fs, F)
    ωs = vcat(ωs, ω)
end

# Los datos a simular serían
F_sim = Float32.(abs.(Fs))
freq_sim = Float32.(ωs)

plot(freq_sim[1:51],F_sim[1:51])
# Vamos a separar estos datos en secuencias de tamaño 51 para que cada uno represente una transformada de Fourier
length_sequence = 13

X = data_maker(freq_sim, length_sequence)
Y = data_maker(F_sim, length_sequence)

# Creamos el modelo de red neuronal recurrente
modelRNN = Chain(
RNN(length_sequence => 10, relu, init=Flux.glorot_uniform(gain=0.01)),
Dense(10, 50, relu, init = Flux.glorot_uniform(gain=0.01)),
RNN(50 => 10, relu, init=Flux.glorot_uniform(gain=0.01)),
LSTM(10 => 5, init=Flux.glorot_uniform(gain=0.01)),
Dense(5, 1)
)

pen_l2(x::AbstractArray) = sum(abs2, x)/2

function loss(x, y, m)
    loss = sum(Flux.mse(m(xi),yi) for (xi, yi) in zip(x,y))
    return loss
end;

data = zip(X, Y)

ps = Flux.params(modelRNN)

opt = Adam(1e-4)

lossRNN = []
lossRNN_test = []

function accuracy()
    num_correct = 0

    # Transform into a flat Vector{Float32} the Vector{Vector{Float32}}
    flat_vector_train = reduce(vcat,reduce(vcat,x))
    flat_y_train = reduce(vcat,reduce(vcat,Y))

    for i in 1:length(flat_vector_train)
        if abs(modelRNN([flat_vector_train[i]])[1] - flat_y_train[i]) < 0.1
            num_correct += 1
        end
    end

    return (num_correct/length(flat_vector_train)) * 100.0
end

iter = 0
epoch_iter = 0

cb = function()
    global iter
    global epoch_iter
    iter += 1
    # Record Loss

    if iter % length(data) == 0
        epoch_iter += 1
        actual_loss = 0
        actual_loss_test = 0

        for (x, y) in data
            actual_loss += loss(x, y, modelRNN)
        end

        if epoch_iter % 100 == 0
            println("Epoch $epoch_iter || Loss = $actual_loss")
        end

        #push!(lossRNN, actual_loss)

    end
end

for _ in 1:1000
    Flux.train!(loss, ps, data, opt, cb = cb)
end
freq_sim
[modelRNN(x)[1] for x in freq_sim[1]]
prediction = reduce(vcat,[modelRNN(x) for x in [[y] for y in freq_sim]])

scatter(freq_sim[1:51], F_sim[1:51])
scatter!(freq_sim[1:51], prediction[1:51])