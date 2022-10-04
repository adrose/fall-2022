using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM

# read in function to create state transitions for dynamic model
include("create_grids.jl")

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: reshaping the data
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# load in the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)


# create bus id variable
df = @transform(df, bus_id = 1:size(df,1))

#---------------------------------------------------
# reshape from wide to long (must do this twice be-
# cause DataFrames.stack() requires doing it one 
# variable at a time)
#---------------------------------------------------
# first reshape the decision variable
dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
rename!(dfy_long, :value => :Y)
dfy_long = @transform(dfy_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfy_long, Not(:variable))

# next reshape the odometer variable
dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)
dfx_long = @transform(dfx_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfx_long, Not(:variable))

# join reshaped df's back together
df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
sort!(df_long,[:bus_id,:time])


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2: estimate a static version of the model
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Isolate vars
X = [ones(size(df_long,1),1) df_long.Branded df_long.Odometer]
Y = df_long.Y
## Code taken from: https://github.com/OU-PhD-Econometrics/fall-2022-private/blob/master/PS2solutions.jl#L38
function logit(alpha, X, y)

    P = exp.(X*alpha)./(1 .+ exp.(X*alpha))

    loglike = -sum( (y.==1).*log.(P) .+ (y.==0).*log.(1 .- P) )

    return loglike
end
alpha_hat_optim = optimize(a -> logit(a, X, Y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(alpha_hat_optim.minimizer)
# Cross check with GLM
# Following code from: https://github.com/OU-PhD-Econometrics/fall-2022-private/blob/master/PS2solutions.jl#L53
alpha_hat_glm = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())
println(alpha_hat_glm)
## Results are agree:
# (Intercept)   1.92596 
# Odometer     -0.148154
# Branded       1.05919 

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3a: read in data for dynamic model
#:::::::::::::::::::::::::::::::::::::::::::::::::::
## some logic guided by: https://github.com/jabbring/dynamic-discrete-choice & https://jabbring.github.io/dynamic-discrete-choice/dynamicDiscreteChoice.m.html#_default_8
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
dfy = @select(df,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20)
dfx = @select(df, :Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20)
dfm = @select(df,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
dfr = @select(df, :RouteUsage)
dfz = @select(df, :Zst)
dfb = @select(df, :Branded)
zval,zbin,xval,xbin,xtran = create_grids()
@views @inbounds function ddc(alpha, dfy, dfx, dfm, dfr, dfz, dfb,zval,zbin,xval,xbin,xtran)
    future_val = zeros(size(xtran,1), 2, 21)
    for t = 20:-1:1
        for b = 0:1
            bval = b + 1
            for z in 1:zbin
                for x in 1:xbin
                    index = x + (z -1)*xbin
                    u1 = alpha[1] + alpha[2] * xval[x] + alpha[3]*b + xtran[index,:]'*future_val[(z-1)*xbin+1:z*xbin,bval,t+1]
                    u0 = alpha[1] + alpha[3]*b + xtran[index,:]'*future_val[(z-1)*xbin+1:z*xbin,b+1,t+1]
                    future_val[index,bval,t] = .9 * log(exp(u1) + exp(u0))
                end
            end
        end
    end
    log_lik = 0
    for i=1:size(df,1)
        for t = 1:size(dfx, 2)
            index1 = 1+(dfz[i,1]-1)*xbin
            index2 = dfx[i,t] + (dfz[i,1]-1)*xbin 
            P = exp.(alpha[1] + alpha[2]*dfm[i,t] + alpha[3]*dfb[i,1] + (xtran[index2,:].-xtran[index1,:])⋅future_val[index1:index1+xbin-1,dfb[i,1]+1,t+1]) ./ (1 .+ exp.(alpha[1] + alpha[2]*dfm[i,t] + alpha[3]*dfb[i,1] + (xtran[index2,:].-xtran[index1,:])⋅future_val[index1:index1+xbin-1,dfb[i,1]+1,t+1]))
            log_lik = log_lik + P
        end
    end
    return -log_lik
end
alpha_int = [1.92596, -0.148154, 1.05919]
theta_hat_loglike = optimize(alpha -> loglike_function(alpha, dfy, dfm, dfx, dfb, dfz, zbin, xbin, xtran), alpha_int, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100_000, show_every = 10))