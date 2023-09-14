using Flux, Plots, Distributions, SpecialFunctions, FFTW
using Flux: train!

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

F = rfft(ys[1,:])
ω = rfftfreq(length(time_sequence[1,:]), fs)

for i in range(1,100)
    F = rfft(ys[i,:])
    ω = rfftfreq(length(time_sequence[1,:]), fs)
    Fs = vcat(Fs, F)
    ωs = vcat(ωs, ω)
end

# Ploteamos una única de las transfomadas de Fourier
plot(ωs[1:501], abs.(Fs)[1:501], label="Transformada de Fourier", xlabel="Frecuencia", ylabel="|F(f)|", title="Transformada de Fourier")

# Quiero ver si puedo hacer la IFFT de esto
N = length(time_sequence[1,:])

F = rfft(ys[1,:])

f2 = irfft(F, N)

f2

plot(time_sequence[1,:], f2)
plot!(time_sequence[1,:], ys[1,:])