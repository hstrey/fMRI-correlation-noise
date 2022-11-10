using NPZ, Plots, Statistics, Turing, Distributions
using ReverseDiff, Memoization
using StatsPlots

Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

subj1 = Array{Float64}(npzread("oxytocinRSdata_npz/Subject001.npy"))
mpfc = subj1[:,1]
llp = subj1[:,2]
rlp = subj1[:,3]
pcc = subj1[:,4]

plot(llp)
plot!(rlp)

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

pearsonllp_rlp = Statistics.cor(llp,rlp)
pearsonpcc_rlp = Statistics.cor(pcc,rlp)
pearsonpcc_llp = Statistics.cor(pcc,llp)
pearsonpcc_mpfc = Statistics.cor(pcc,mpfc)
pearsonllp_mpfc = Statistics.cor(llp,mpfc)
pearsonrlp_mpfc = Statistics.cor(rlp,mpfc)

y1n = rlp .+ mpfc
y2n = rlp .- mpfc
modeln = ou_corrn(y1n,y2n,length(y1n),0.802)
chnn = sample(modeln, NUTS(0.65), 2000)

p2 = plot(chnn[[:ampl1,:ampl2,:noise_ampl]])

ampl1n = Array(chnn[:ampl1])
ampl2n = Array(chnn[:ampl2])

if mean(ampl1n)>mean(ampl2n)
    cn = (ampl1n .- ampl2n)./ampl2n
else
    cn = (ampl1n .- ampl2n)./ampl1n
end

println("A1,A2: ",mean(ampl1n),",",mean(ampl2n))
println("c estimate: ",mean(cn),"std: ",std(cn))
println("rho ",mean(cn)/(mean(cn)+2))
