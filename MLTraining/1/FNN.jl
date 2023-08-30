using Flux
using Statistics
using Flux: train!


t = range(0,stop=2*pi,length=100)
t = reshape(t,1,100)
