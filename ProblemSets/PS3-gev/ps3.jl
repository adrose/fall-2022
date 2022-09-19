using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

function all_ans()
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 1
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
    df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation

    ## Code taken from: https://github.com/OU-PhD-Econometrics/fall-2022-private/blob/master/PS2solutions.jl#L75
    ## Adapt this function to also accept the z matrix
    function mlogit(alpha, X, y, Z) ## add Z matrix here

        alpha_z = last(alpha)
        alpha_x = alpha[1:size(alpha,1)-1]
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha_x,K,J-1) zeros(K)]
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            vals1 = (X*bigAlpha[:,j])
            vals2 = (Z[:,j] .- Z[:,J])*alpha_z ## standardize to final Z column
            num[:,j] = exp.(vals1 .+ vals2)
            dem .+= num[:,j]
        end
        P = num./repeat(dem,1,J)

        loglike = -sum( bigY.*log.(P) )

        return loglike
    end
    rand_start =  [rand(7*size(X,2) + 1) ; ]
    mlogit(rand_start, X, y, Z)
    alpha_hat_optim = optimize(a -> mlogit(a, X, y, Z), rand_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    println(alpha_hat_optim.minimizer)
    ## Try a wide range of starting points to deal with local versus global minimum issues
    iter_count = 25
    outMin = zeros(size(alpha_hat_optim.minimizer, 1), iter_count) 
    outLik =  zeros(iter_count) 
    #Random.seed!(16)
    #Threads.@threads for i=1:iter_count ## export set JULIA_NUM_THREADS 8 prior to run from bash terminal for multithreading
    #    rand_start = [rand(7*size(X,2) + 1) ; ]
    #    tmp_optim = optimize(a -> mlogit(a, X, y, Z), rand_start, LBFGS(), Optim.Options(g_tol = 1e-6, iterations=100_000, show_trace=true, show_every=50))
    #    outMin[:,i] = tmp_optim.minimizer
    #    outLik[i] = tmp_optim.minimum
    #end
    ### Looks like we have a global optimum
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 2
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # γ can be interpreted similar as a coefficient from a linear regression: 
    # the change in one unit of the independent variable (elnwages)
    # provides this much increase in the dependent variable (utility).  

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 3
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    ## Nesting structure:
    #1 == 1:3
    #2 == 4:7
    #3 == >7
    function mlogit(alpha, X, y, Z)

        alpha_x = alpha[1:end-3]
        lambda = alpha[end-2:end-1]
        alpha_z = alpha[end]
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [repeat(alpha_x[1:K],1,3) repeat(alpha_x[K+1:2K],1,4) zeros(K)]
        num = zeros(N,J)
        dem = zeros(N)
        vals1 = zeros(N, 3)
        vals2 = zeros(N, 4)
        vals1_index = 1
        vals2_index = 1
        for j=1:J
            if j in [1,2,3]
                vals1[:, vals1_index] = exp.((X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*alpha_z)./lambda[1])
                vals1_index = vals1_index + 1
            elseif j in [4,5,6,7]
                vals2[:, vals2_index] = exp.((X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*alpha_z)./lambda[2])
                vals2_index = vals2_index + 1
            end
        end
        vals1_index = 1
        vals2_index = 1
        vals3_index = 1
        for j=1:J
            if j in [1,2,3]
                num[:,j] = vals1[:,vals1_index].*sum(vals1, dims=2).^(lambda[1]-1)
                vals1_index = vals1_index + 1
            elseif j in [4,5,6,7]
                num[:,j] = vals2[:,vals2_index].*sum(vals2, dims=2).^(lambda[2]-1)
                vals2_index = vals2_index + 1
            else
                num[:,j] = repeat([1], N)
                vals3_index = vals3_index + 1
            end
        end
        dem1 = sum(vals1, dims=2).^lambda[1]
        dem2 = sum(vals2, dims=2).^lambda[2]
        dem = 1 .+ dem1 .+ dem2

        P = num./repeat(dem,1,J)

        loglike = -sum( bigY.*log.(P) )

        return loglike
    end

    β_wc=rand(3)
    β_bc=rand(3)
    λ_wc=rand(1)
    λ_bc=rand(1)
    γ=rand(1)
    rand_start = vcat(β_wc, β_bc, λ_wc, λ_bc, γ)
    mlogit(rand_start, X, y, Z)
    alpha_hat_optim = optimize(a -> mlogit(a, X, y, Z), rand_start, LBFGS(), Optim.Options(g_tol = 1e-6, iterations=100_000, show_trace=true, show_every=50))
    println(alpha_hat_optim.minimizer)
    ## Now again explore local versus global
    #iter_count = 3
    #outMin = zeros(size(alpha_hat_optim.minimizer, 1), iter_count) 
    #outLik =  zeros(iter_count) 
    #Random.seed!(16)
    #Threads.@threads for i=1:iter_count ## export set JULIA_NUM_THREADS 8 prior to run
    #    rand_start1 =  rand_start .* rand(9)
    #    tmp_optim = optimize(a -> mlogit(a, X, y, Z), rand_start1, LBFGS(), Optim.Options(g_tol = 1e-6, iterations=100_000, show_trace=true, show_every=50))
    #    outMin[:,i] = tmp_optim.minimizer
    #    outLik[i] = tmp_optim.minimum
    #end
    ## Very weird behavior----idenitical likelihood but very different coefficients across these solutions...
    ## Not sure if I need to restrict the lambda value 0<lambda[1:2]<1 ?????
end