using CSV
using DataFrames
using Distributions
using FreqTables
using GLM
using HTTP
using LinearAlgebra
using Optim
using Random
using Statistics

function all_ans()
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 1
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code


    ## Function taken from: https://github.com/OU-PhD-Econometrics/fall-2022-private/blob/master/PS3solutions.jl#L14
    ## Adapted to use quadrature
    function mlogit_with_Z(theta, X, Z, y)

        alpha = theta[1:end-1]
        gamma = theta[end]
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]

        T = promote_type(eltype(X),eltype(theta))
        num   = zeros(T,N,J)
        dem   = zeros(T,N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
            dem .+= num[:,j]
        end

        P = num./repeat(dem,1,J)

        loglike = -sum( bigY.*log.(P) )

        return loglike
    end
    startvals = [2*rand(7*size(X,2)).-1; .1]
    td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), startvals; autodiff = :forward)
    # run the optimizer
    theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    theta_hat_mle_ad = theta_hat_optim_ad.minimizer
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, theta_hat_mle_ad)
    theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
    println("logit estimates with Z")
    println([theta_hat_mle_ad theta_hat_mle_ad_se]) # these standard errors match Stata

    ## End code taken from: https://github.com/OU-PhD-Econometrics/fall-2022-private/blob/master/PS3solutions.jl#L14


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 2
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # The previous coefficient was: -0.09420499679668991 - suggesting that as wage increased the expected utitliy decreased, 
    # which goes against basic utility maximization theory. The updated coefficient of: 0.1246690576664692 -- suggests the opposite direction and is
    # more in line with utility maximization theory.


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 3
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    include("lgwt.jl")
    d = Normal(0,1)
    # get quadrature nodes and weights for 7 grid points
    nodes, weights = lgwt(7,-4,4)
    # now compute the integral over the density and verify it's 1
    sum(weights.*pdf.(d,nodes))  # --> total area under normal curve
    # evalutes to 1.004 ~ 1
    # now compute the expectation and verify it's 0
    sum(weights.*nodes .*pdf.(d,nodes)) # --> mean of normal curve
    # evalutes to 2*10^-17 ~ 0
    d = Normal(0,2)
    nodes, weights = lgwt(7,-10,10)
    sum(weights.* nodes.^2 .*pdf.(d,nodes))
    # evalutes to 3.265514281891983 -- ctrue varaince is 4; not very close
    nodes, weights = lgwt(10,-10,10)
    sum(weights.* nodes.^2 .*pdf.(d,nodes))
    # evaluates to 4.038977384853661 -- much closer to 4 than 7 nodes

    function No_mc_Only_mc(D, f, c, d, a)
        ## identify all of our sample values
        R = c - f
        rand_samp = rand(Uniform(f,c), D)
        ## find denisty of the normal distirbuion at these points
        if a in [1]
            vals_mean = (rand_samp .^ 2) .* (pdf.(d, rand_samp))
        elseif a in [2]
            vals_mean = (pdf.(d, rand_samp)) .* rand_samp
        else 
            vals_mean = (pdf.(d, rand_samp))
        end
        vals_mean = mean(vals_mean)
        vals_mean = R * vals_mean
        # Prep output
        output = DataFrame(D = D, mu = vals_mean)
    end
    No_mc_Only_mc(1_000_000, -10, 10, Normal(0,2), 1)
    ## Evaluates to 3.99 - very close to 4
    No_mc_Only_mc(1_000_000, -10, 10, Normal(0,2), 2)
    ## Evaluted to 0.000807947 - very close to 0
    No_mc_Only_mc(1_000_000, -10, 10, Normal(0,2), 3)
    ## Evaluated to 1.00134 - very close to 1
    No_mc_Only_mc(1_000, -10, 10, Normal(0,2), 1)
    ## Evaluated to 3.79479 - further from 4 than 1_000_000 rand samps
    No_mc_Only_mc(1_000, -10, 10, Normal(0,2), 2)
    ## Evaluated to -0.0307 - further from 0 than 1_000_000 rand samps
    No_mc_Only_mc(1_000, -10, 10, Normal(0,2), 3)
    ## EValuated to 0.96... - furhter from 1 than 1_000_000 rand samps

    ## Explore standard error? -- Is this a repeated sampling procedure?
    x = Vector{Float64}()
    for i =1:size(mil_se, 1)
        tmp_val = No_mc_Only_mc(1_000_000, -10, 10, Normal(0,2), 3)
        append!(x, tmp_val.mu)
    end
    mean(x) # Mean of the means == 1.000002
    Statistics.std(x) # Standard error == 0.001
    x = Vector{Float64}()
    for i =1:size(mil_se, 1)
        tmp_val = No_mc_Only_mc(1_000, -10, 10, Normal(0,2), 3)
        append!(x, tmp_val.mu)
    end
    mean(x) # mean of the means == .997
    Statistics.std(x) # Standard error == 0.05 ; Much larger! - 50x Larger 

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 4
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    function retP(theta, X, Z, y, node)       
        alpha = theta[1:end-1]
        gamma = theta[end]
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        n_n = length(nodes)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]

        T = promote_type(eltype(X),eltype(theta))
        num   = zeros(T,N,J)
        dem   = zeros(T,N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*node)
            dem .+= num[:,j]
        end

        P = num./repeat(dem,1,J)
        return P
    end

    fac_expo = function(Y)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        return bigY
    end

    startvals = [2*rand(7*size(X,2)).-1; .1]
    ## Run through a single permutation here
    retP(startvals, X, Z, y, .3)
    ##turn this loop into a function
    #out_sum = [0]
    #for l=eachindex(nodes)
    #    println(l)
    #    F = fac_expo(y)
    #    p_mat = retP(startvals, X, Z, y, nodes[l])
    #    p_mat = prod(p_mat.^F, dims=2)
    #    out_sum = out_sum .+ (weights[l]*p_mat*pdf(d, nodes[l]))
    #end
    function all_loop(startvals, x, Z, y)
        F = fac_expo(y)
        sigma = startvals[end]
        pop!(startvals)
        mu = startvals[end]
        pop!(startvals)
        nodes, weights = lgwt(7, -sigma, sigma)
        out_sum = [0]
        d = Normal(mu,sigma)
        for l= eachindex(nodes)
            p_mat = retP(startvals, X, Z, y, nodes[l])
            p_mat = prod(p_mat.^F, dims=2)
            out_sum  = out_sum .+ (weights[l]*p_mat*pdf(d, nodes[l]))
        end
        loglike = -sum(log.(out_sum))
    end
    startvals = [2*rand(7*size(X,2)).-1; .1; 0; 4]
    all_loop(startvals, X, Z, y)
    # theta_hat_optim_ad = optimize(theta -> all_loop(theta, X, Z, y), startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 5
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    function all_loop(theta, X, Z, y, D)
        F = fac_expo(y)
        sigma = theta[end]
        #pop!(theta)
        mu = theta[end-1]
        #pop!(theta)
        theta_exp = theta[1:22]
        nodes, weights = lgwt(7, -sigma, sigma)
        out_sum = [0]
        f = mu - (sigma*5)
        c = mu + (sigma*5)
        vals = rand(Uniform(f,c), D)
        range_val = c - f
        d = Normal(mu, sigma)
        for l=1:length(vals)
            p_mat = retP(theta_exp, X, Z, y, vals[l])
            p_mat = prod(p_mat.^F, dims=2)
            out_sum  = out_sum .+ ((range_val/D)*p_mat*pdf(d, vals[l]))
        end
        loglike = -sum(log.(out_sum))
    end
    startvals = [2*rand(7*size(X,2)).-1; .1; 0; 4]
    all_loop(startvals, X, Z, y, 1_000_000)
    #theta_hat_optim_ad = optimize(startvals -> all_loop(startvals, X, Z, y, 1_000_000), startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
end
all_ans()