using Turing, ReverseDiff, Memoization
using DifferentialEquations
using Plots
using Statistics
using StatsPlots
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)
using Distributions
using LinearAlgebra
using Turing: Variational

# here we are simulating the signal of two coupled oscillators
# with amplitudes A1 and A2 of kBT/k and kBT/(k+c) where c is the coupling constant
# D = kBT/gamma
# in terms of the standard parameters of a OU process:
# sigma = sqrt(2*D)
# Theta = D/A

function corr_osc(c)
    # function that returns two time series that are correlated through a coupling coefficient c
    μ = 0.0 # mean is zero
    σ = sqrt(2) # D=1
    Θ1 = 1.0
    Θ2 = 1.0+abs(c)

    W1 = OrnsteinUhlenbeckProcess(Θ1,μ,σ,0.0,1.0)
    W2 = OrnsteinUhlenbeckProcess(Θ2,μ,σ,0.0,1.0)
    prob1 = NoiseProblem(W1,(0.0,100.0))
    prob2 = NoiseProblem(W2,(0.0,100.0))
    sol1 = solve(prob1;dt=0.1)
    sol2 = solve(prob2;dt=0.1)

    # creating the two correlated
    x1 = (sol1.u .+ sol2.u)/2
    if c>0 
        x2 = (sol1.u .- sol2.u)/2
    else
        x2 = (sol2.u .- sol1.u)/2
    end
    return x1,x2
end

# Ornstein-Uhlenbeck process
@model ou(rn,T,delta_t) = begin
    ampl ~ Uniform(0.0,5.0)
    b ~ beta(5.0,1.0)
    
    rn[1] ~ Normal(0,sqrt(ampl))
    
    for i=2:T
        rn[i] ~ Normal(rn[i-1]*b,sqrt(ampl*(1-b^2)))
    end
end

# Ornstein-Uhlenbeck process of two coupled oscillators
@model ou_corr(rn1,rn2,T,delta_t) = begin
    ampl1 ~ Uniform(0.0,5.0)
    ampl2 ~ Uniform(0.0,5.0)
    d ~ Uniform(0.0,5.0)
    b1 = exp(-delta_t*d/ampl1)
    b2 = exp(-delta_t*d/ampl2)

    rn1[1] ~ Normal(0,sqrt(ampl1))
    rn2[1] ~ Normal(0,sqrt(ampl2))   

    for i=2:T
        rn1[i] ~ Normal(rn1[i-1]*b1,sqrt(ampl1*(1-b1^2)))
        rn2[i] ~ Normal(rn2[i-1]*b2,sqrt(ampl2*(1-b2^2)))
    end
end

# Ornstein-Uhlenbeck process of two coupled oscillators with thermal noise
@model ou_corrn(rn1,rn2,T,delta_t,::Type{R}=Vector{Float64}) where {R} = begin
    ampl1 ~ Uniform(0.0,5.0)
    ampl2 ~ Uniform(0.0,5.0)
    d ~ Uniform(0.0,5.0)
    b1 = exp(-delta_t*d/ampl1)
    b2 = exp(-delta_t*d/ampl2)
    noise_ampl ~ Uniform(0.0,1)

    r1 = R(undef, T)
    r2 = R(undef, T)

    r1[1] ~ Normal(0,sqrt(ampl1))
    r2[1] ~ Normal(0,sqrt(ampl2))   

    for i=2:T
        r1[i] ~ Normal(r1[i-1]*b1,sqrt(ampl1*(1-b1^2)))
        r2[i] ~ Normal(r2[i-1]*b2,sqrt(ampl2*(1-b2^2)))
    end
    rn1 ~ MvNormal(r1,sqrt(noise_ampl))
    rn2 ~ MvNormal(r2,sqrt(noise_ampl))
end

# Ornstein-Uhlenbeck process of two coupled oscillators with thermal noise
@model ou_corrmn(rn1,rn2,T,delta_t,::Type{R}=Vector{Float64}) where {R} = begin
    ampl1 ~ Uniform(0.0,5.0)
    ampl2 ~ Uniform(0.0,5.0)
    d ~ Uniform(0.0,5.0)
    b1 = exp(-delta_t*d/ampl1)
    b2 = exp(-delta_t*d/ampl2)
    noise_ampl_t ~ Uniform(0.0,1)

    r1 = R(undef, T)
    r2 = R(undef, T)

    r1[1] ~ Normal(0,sqrt(ampl1))
    r2[1] ~ Normal(0,sqrt(ampl2))   

    for i=2:T
        r1[i] ~ Normal(r1[i-1]*b1,sqrt(ampl1*(1-b1^2)))
        r2[i] ~ Normal(r2[i-1]*b2,sqrt(ampl2*(1-b2^2)))
    end
    rn1 ~ MvNormal(r1,sqrt.(noise_ampl_t .+ noise_ampl_t*abs.(r1)))
    rn2 ~ MvNormal(r2,sqrt.(noise_ampl_t .+ noise_ampl_t*abs.(r2)))
end

# Ornstein-Uhlenbeck process with added Gaussian noise
@model oupn(rn,T,delta_t,::Type{R}=Vector{Float64}) where {R} = begin
    ampl ~ Uniform(0.0,5.0)
    b ~ Beta(5.0,1.0)
    noise_ampl ~ Uniform(0.0,1)
    
    b = exp(-delta_t/tau)
    r = R(undef, T)
    
    r[1] ~ Normal(0,sqrt(ampl))
    
    for i=2:T
        r[i] ~ Normal(r[i-1]*b,sqrt(ampl*(1-b^2)))
    end
    rn ~ MvNormal(r,sqrt(noise_ampl))
end

x1, x2 = corr_osc(5)

p1 = Plots.plot(x1)
p1 = Plots.plot!(x2)

pearson = Statistics.cor(x1,x2)
println(pearson)

# lets see whether we can estimate c from the data
y1 = x1 .+ x2
y2 = x1 .- x2
chn = sample(ou_corr(y1,y2,length(y1),0.1), NUTS(0.65), 2000)

print(describe(chn))
p2 = plot(chn)

ampl1 = Array(chn[:ampl1])
ampl2 = Array(chn[:ampl2])

if mean(ampl1)>mean(ampl2)
    c = (ampl1 .- ampl2)./ampl2
else
    c = (ampl1 .- ampl2)./ampl1
end

println("A1,A2: ",mean(ampl1),",",mean(ampl2))
println("c estimate: ",mean(c),"std: ",std(c))

# add some thermal noise
norm_dist = Normal(0,sqrt(0.2)) # 20% thermal noise
x1n = x1 .+ rand(norm_dist,length(x1))
x2n = x2 .+ rand(norm_dist,length(x2))

pearsonn = Statistics.cor(x1n,x2n)

y1n = x1n .+ x2n
y2n = x1n .- x2n
chnn = sample(ou_corrn(y1n,y2n,length(y1n),0.1), NUTS(0.65), 2000)

ampl1n = Array(chnn[:ampl1])
ampl2n = Array(chnn[:ampl2])

if mean(ampl1n)>mean(ampl2n)
    cn = (ampl1n .- ampl2n)./ampl2n
else
    cn = (ampl1n .- ampl2n)./ampl1n
end

println("A1,A2: ",mean(ampl1n),",",mean(ampl2n))
println("c estimate: ",mean(cn),"std: ",std(cn))

# add some thermal and multiplicative noise
ratio = 1
Tnoise = 0.4
x1mn = x1 .+ [rand(Normal(0,sqrt(Tnoise*(1+ratio*abs(x))))) for x in x1]
x2mn = x2 .+ [rand(Normal(0,sqrt(Tnoise*(1+ratio*abs(x))))) for x in x2]

pearsonn = Statistics.cor(x1mn,x2mn)

y1mn = x1mn .+ x2mn
y2mn = x1mn .- x2mn
modelmn = ou_corrmn(y1mn[1:400],y2mn[1:400],length(y1mn[1:400]),0.1)
chnmn = sample(modelmn, NUTS(0.65), 2000)

ampl1mn = Array(chnmn[:ampl1])
ampl2mn = Array(chnmn[:ampl2])

if mean(ampl1mn)>mean(ampl2mn)
    cmn = (ampl1mn .- ampl2mn)./ampl2mn
else
    cmn = (ampl1mn .- ampl2mn)./ampl1mn
end

r1avg = mean(Array(group(chnmn, :r1));dims=1)[1,:]
plot(r1avg)
plot!(y1[1:400],alpha=0.5)

println("A1,A2: ",mean(ampl1mn),",",mean(ampl2mn))
println("c estimate: ",mean(cmn),"std: ",std(cmn))

p2 = plot(chnmn[[:ampl1,:ampl2,:noise_ampl_t]])

plot(x1, label="x1")
plot!(x1mn, label="x1 + noise")

plot(x2, label="x2")
plot!(x2mn, label="x2 + noise")

# lets try ADVI
advi = ADVI(10, 1000)
setchunksize(8)
q = vi(modelmn, advi)

# sampling
z = rand(q, 1000)
avg = vec(mean(z; dims = 2))
zstd = vec(std(z; dims = 2))
y1avg = avg[5:404]
y2avg = avg[405:end]

ampl1amn = z[1,:]
ampl2amn = z[2,:]

if mean(ampl1amn)>mean(ampl2amn)
    camn = (ampl1amn .- ampl2amn)./ampl2amn
else
    camn = (ampl1amn .- ampl2amn)./ampl1amn
end

println("A1,A2: ",mean(ampl1amn),",",mean(ampl2amn))
println("c estimate: ",mean(camn),"std: ",std(camn))

# compare advi to ground truth
plot(y1avg, label="estimate")
plot!(y1[1:400],label="ground truth")
plot!(y1mn[1:400],alpha=0.5,label="data")

# compare advi and MCMC in predicting original time-series
plot(r1avg, label="MCMC")
plot!(y1avg, label="ADVI")
plot!(y1[1:400],label="ground truth",alpha=0.5)
