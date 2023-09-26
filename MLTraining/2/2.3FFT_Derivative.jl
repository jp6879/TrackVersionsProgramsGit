using FFTW, Plots

N = 20
L = 1
xj = (0:N-1)*L/N
f = sin.(2π*xj)
df = 2π*cos.(2π*xj)

k = fftfreq(N)*N
df_fft =  (2π*im/L*k) .* fft(f)
f_fft = fft(f)

f_ifft = ifft(f_fft)
df_ifft = ifft(df_fft)


scatter(k,abs.(f_fft),label="FFT")
scatter!(k,abs.(df_fft),label="FFT derivative")

plot(xj,real(f_fft),label="ifft")
plot!(xj,real(df_ifft),label="ifft derivative")
