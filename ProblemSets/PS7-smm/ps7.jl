using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using DataFramesMeta
using CSV
##using SMM

function all_ans()

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    ## Code taken from: https://github.com/OU-PhD-Econometrics/fall-2022-private/blob/master/PS2solutions.jl#L19
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    function ols(beta, X, y)
        ssr = (y.-X*beta)'*(y.-X*beta)
        return ssr
    end
    beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    ## End code taken from: https://github.com/OU-PhD-Econometrics/fall-2022-private/blob/master/PS2solutions.jl#L19
    function ols_GMM(beta,X,y)
        ssr = (y.-X*beta)'*I*(y.-X*beta)
        return ssr
    end
    beta_hat_ols_GMM = optimize(b -> ols_GMM(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    ## Code taken from: https://github.com/OU-PhD-Econometrics/fall-2022-private/blob/master/PS2solutions.jl#L61
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7

    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation

    function mlogit(alpha, X, y)

        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]

        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem .+= num[:,j]
        end

        P = num./repeat(dem,1,J)

        loglike = -sum( bigY.*log.(P) )

        return loglike
    end
    ## End code taken from: https://github.com/OU-PhD-Econometrics/fall-2022-private/blob/master/PS2solutions.jl#L61

    fac_expo = function(Y)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        return bigY
    end

    function mlogit_GMM(alpha, X, y)

        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        g = zeros(N,J)
        bigY = fac_expo(y)
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem .+= num[:,j]
        end
        P = num./repeat(dem,1,J)
        ## Now loop through and subtract the probability from the true event
        ## for every J
        for j=1:J
            for n=1:N
                g[n,j] = bigY[n,j] - P[n,j]
            end
        end
        ## Flatten G
        g=g[:]
        # Return weighted (identity mat) sum of squares
        loglike = g'*I*g
        return loglike
    end

    ## Orig start vals
    alpha_rand = rand(6*size(X,2))
    alpha_hat_optim = optimize(a -> mlogit(a, X, y), alpha_rand, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    alpha_hat_mle = alpha_hat_optim.minimizer
    # GMM Orig start vals
    alpha_hat_optim_gmm = optimize(a -> mlogit_GMM(a, X, y), alpha_hat_mle, LBFGS(), Optim.Options(g_tol = 1e-4, iterations=100, show_trace=true, show_every=50))
    alpha_hat_gmm = alpha_hat_optim_gmm.minimizer
    # GMM rand start vals
    alpha_hat_optim_gmmr = optimize(a -> mlogit_GMM(a, X, y), alpha_rand, LBFGS(), Optim.Options(g_tol = 1e-4, iterations=100, show_trace=true, show_every=50))
    alpha_hat_gmmr = alpha_hat_optim_gmmr.minimizer

    ###############
    # I was never get this to run efficiently
    # However, when the answers were run with really loose optimization params the coefficients were very different
    # This lightly suggests there are issues with local and global optimum
    ##############


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    function simVals(N, J, alpha, nPred)
        expo = J - 1
        #alpha_rand = rand(nPred*expo)
        bigAlpha = [reshape(alpha,nPred,expo) zeros(nPred)]
        ## Create our simulees
        simX = zeros(N, nPred)
        ## Add values
        for i=1:nPred
            simX[:,i] = rand(N)
        end
        ## Now estimate the true P matrix
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(simX*bigAlpha[:,j])
            dem .+= num[:,j]
        end
        P = num./repeat(dem,1,J)
        ## Assign outcome
        simY = zeros(N)
        prefShock = rand(N)
        for j = 1:J
            ## Identify values greater than error
            identVals = sum(P[:,j:J], dims=2) .> prefShock
            simY = simY + identVals
        end
        return simX,simY
    end

    ## sim vals
    alpha_true = rand(9*3)
    outX, outY = simVals(10000, 10, alpha_true, 3)
    ## Estimate w/ MLE & GMM
    alpha_rand = rand(9*3)
    mlogit(alpha_rand, outX, outY)
    mlogit_GMM(alpha_rand, outX, outY)
    mle_est = optimize(a -> mlogit(a, outX, outY), alpha_rand, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    gmm_est = optimize(a -> mlogit_GMM(a, outX, outY), alpha_rand, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100, show_trace=true, show_every=50))
    ##### GMM DOES NOT CONVERGE!!!


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 4 -- DNE
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    #using SMM, DataFrames
    #MA = SMM.parallelNormal() # Note: this line may take up to 5 minutes to execute
    #dc = SMM.history(MA.chains[1])
    #dc = dc[dc[!,:accepted].==true, :]
    #println(describe(dc))


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 5
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    ## Code taken from: https://github.com/OU-PhD-Econometrics/fall-2022/blob/master/LectureNotes/09-SMM/09slides.Rmd#L316
    function ols_smm(θ, X, y, D)
        K = size(X,2)
        N = size(y,1)
        β = θ[1:end-1]
        σ = θ[end]
        if length(β)==1
            β = β[1]
        end
        # N+1 moments in both model and data
        gmodel = zeros(N+1,D)

        # data moments are just the y vector itself
        # and the variance of the y vector
        gdata  = vcat(y,var(y))
        #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!! ####
        # This is critical!                   #
        Random.seed!(1234)                    #
        # You must always use the same ε draw #
        # for every guess of θ!               #
        #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ###
        # simulated model moments
        for d=1:D
            ε = σ*randn(N)
            ỹ = X*β .+ ε
            gmodel[1:end-1,d] = ỹ
            gmodel[  end  ,d] = var(ỹ)
        end
        # criterion function
        err = vec(gdata .- mean(gmodel; dims=2))
        # weighting matrix is the identity matrix
        # minimize weighted difference between data and moments
        J = err'*I*err
        return J
    end
    ## End code taken from: https://github.com/OU-PhD-Econometrics/fall-2022/blob/master/LectureNotes/09-SMM/09slides.Rmd#L316
    alpha_true = rand(4*3)
    outX, outY = simVals(10000, 4, alpha_true, 4)
    alpha_rand = rand(5)
    ols_smm(rand(5), outX, outY, 1000)
    smm_res = optimize(a -> ols_smm(a, outX, outY, 1000), alpha_rand, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100, show_trace=true, show_every=50))
end
all_ans()