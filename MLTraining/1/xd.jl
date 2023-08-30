using Random
using Flux
using Plots

# Generar datos con ruido
function generate_data(num_samples, noise_std)
    t = range(0, stop=2π, length=num_samples)
    y_true = sin.(t)
    y_noisy = y_true + randn(num_samples) .* noise_std
    return t, y_noisy
end

# Dividir los datos en conjuntos de entrenamiento y prueba
function split_data(t, y, train_ratio)
    num_samples = length(t)
    train_size = Int(floor(train_ratio * num_samples))
    
    indices = shuffle(1:num_samples)
    train_indices = indices[1:train_size]
    test_indices = indices[train_size+1:end]
    
    t_train = t[train_indices]
    y_train = y[train_indices]
    t_test = t[test_indices]
    y_test = y[test_indices]
    
    return t_train, y_train, t_test, y_test
end

# Construir el modelo de red neuronal
function build_model(input_size, hidden_size)
    return Chain(
        Dense(input_size, hidden_size, σ),
        Dense(hidden_size, 1)
    )
end

# Escalar los datos y entrenar el modelo
function train_model(model, t_train, y_train, epochs, learning_rate)
    X_train = t_train
    y_train_scaled = y_train
    
    loss_function(x, y) = Flux.mse(model(x), y)
    opt = ADAM(learning_rate)
    
    for epoch in 1:epochs
        Flux.train!(loss_function, Flux.params(model), [(X_train[i], y_train_scaled[i]) for i in 1:length(X_train)], opt)
    end
end

# Evaluar el modelo en el conjunto de prueba
function evaluate_model(model, t_test, y_test)
    y_pred = model.(t_test)
    return y_pred
end

# Parámetros
num_samples = 200
noise_std = 0.1
train_ratio = 0.8
hidden_size = 10
epochs = 100
learning_rate = 0.01

# Generar datos
t, y_noisy = generate_data(num_samples, noise_std)

# Dividir datos
t_train, y_train, t_test, y_test = split_data(t, y_noisy, train_ratio)

# Construir y entrenar el modelo
model = build_model(1, hidden_size)
train_model(model, t_train, y_train, epochs, learning_rate)

# Evaluar el modelo
y_pred = evaluate_model(model, t_test, y_test)

# Plot
plot(t_test, y_test, label="Datos reales", legend=:topright)
plot!(t_test, y_pred, label="Predicciones", linestyle=:dash)