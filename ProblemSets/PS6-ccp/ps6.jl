using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using DataFramesMeta
using CSV
include("create_grids.jl")

function all_ans()
    ## Load data - taken from: https://github.com/OU-PhD-Econometrics/fall-2022-private/blob/master/PS5solutions.jl#L19
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body,DataFrame)

    # create bus id variable
    df = @transform(df, :bus_id = 1:size(df,1))

    #---------------------------------------------------
    # reshape from wide to long (must do this twice be-
    # cause DataFrames.stack() requires doing it one 
    # variable at a time)
    #---------------------------------------------------
    # first reshape the decision variable
    dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfy_long, Not(:variable))

    # next reshape the odometer variable
    dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfx_long, Not(:variable))

    # join reshaped df's back together
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
    sort!(df_long,[:bus_id,:time])
    ## end code taken from: https://github.com/OU-PhD-Econometrics/fall-2022-private/blob/master/PS5solutions.jl#L19


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    ## Now create the quadratic forms of the varaibles
    df_long.Odometer2 = df_long.Odometer.^2
    df_long.RouteUsage2 = df_long.RouteUsage.^2
    df_long.time2 = df_long.time.^2

    ##create glm object
    mod = glm(@formula(Y ~ Odometer * Branded * time * Odometer2 * RouteUsage2 * time2), df_long, Binomial(), LogitLink())

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Code taken from: https://github.com/OU-PhD-Econometrics/fall-2022-private/blob/master/PS5solutions.jl#L75
    zval,zbin,xval,xbin,xtran = create_grids()
    # End code taken from: https://github.com/OU-PhD-Econometrics/fall-2022-private/blob/master/PS5solutions.jl#L75

    # Load the state values from the observed data
    ## Code taken from: https://github.com/OU-PhD-Econometrics/fall-2022-private/blob/master/PS5solutions.jl#L61
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body,DataFrame)
    Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])
    X = Matrix(df[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
    Z = Vector(df[:,:RouteUsage])
    B = Vector(df[:,:Branded])
    N = size(Y,1)
    T = size(Y,2)
    xstate = Matrix(df[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
    zstate = Vector(df[:,:Zst])
    ## End code taken from: https://github.com/OU-PhD-Econometrics/fall-2022-private/blob/master/PS5solutions.jl#L61


    ## Construct data frame
    df_state = DataFrame(Odometer = kron(ones(zbin),xval),RouteUsage = kron(zval,ones(xbin)),time = zeros(size(xtran,1)), Branded = zeros(size(xtran,1)))
    function fun_read(df_state, df_long,mod, zval, zbin, xval, xbin, xtran, zstate, xstate)
        T = convert(Integer, maximum(df_long.time))
        future_val = zeros(size(xtran, 1), 2, T+1)
        pred_vals_tmp = zeros(size(xtran,1), )
        for t=2:T
            for b=0:1
                println(b)
                df_tmp = df_state
                df_tmp.time = replace(df_tmp.time, 0 =>t)
                df_tmp.time2 = df_tmp.time .^2 
                df_tmp.Odometer2 = df_tmp.Odometer.^2
                df_tmp.RouteUsage2 = df_tmp.RouteUsage.^2
                df_tmp.Branded = replace(df_tmp.Branded, 0 =>b)
                pred_vals = 1 .- predict(mod, df_tmp)
                futureValue = -0.90 .*log.(pred_vals) 
                future_val[:, b+1, t] = futureValue ## For some reason my future value is identical for branded and !Branded????
            end
        end
        FVT1 = zeros(size(xstate,1), T)
        for i=1:size(xstate, 1)
            for t=1:T
                ## Logic taken from: https://github.com/OU-PhD-Econometrics/fall-2022-private/blob/master/PS5solutions.jl#L127
                row0 = (zstate[i]-1)*xbin+1
                row1  = xstate[i,t] + (zstate[i]-1)*xbin
                ## End logic...
                FVT1[i,t] = (xtran[row1,:].-xtran[row0,:])'*future_val[row0:row0+xbin-1,df_long.Branded[i]+1,t+1]
            end
        end
        return(FVT1)
    end
    FVT1 = fun_read(df_state,  df_long, mod, zval, zbin, xval, xbin, xtran, zstate, xstate)
    FVT1 = FVT1'[:]
    df_long = @transform(df_long,fv = FVT1)   
    theta_hat_ccp_glm = glm(@formula(Y ~ Odometer + Branded),df_long, Binomial(), LogitLink(),offset=df_long.fv)
    println(theta_hat_ccp_glm)
end

@time all_ans()