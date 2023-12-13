using Optim, DataFrames, CSV, HTTP, GLM, FreqTables

function all_ans()    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 1
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    #using Optim
    f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
    negf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
    startval = rand(16) 
    result = optimize(negf, startval, LBFGS())

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 2
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    #using DataFrames
    #using CSV
    #using HTTP
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    function ols(beta, X, y)
        ssr = (y.-X*beta)'*(y.-X*beta)
        return ssr
    end
    beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(),Optim.Options(g_tol=1e-6, iterations=100_000,show_trace=true))
    println(beta_hat_ols.minimizer)

    #using GLM
    bols = inv(X'*X)*X'*y
    df.white = df.race.==1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 3
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    function logit(alpha, X, d)
        P = exp.(X*alpha)./(1 .+ exp.(X*alpha))
        loglike = -sum( (d.==1).*log.(P) .+ (d.==0).*log.(1 .- P) ) 
        return loglike
    end
    alpha_hat_optim = optimize(a -> logit(a, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(alpha_hat_optim.minimizer)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 4
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    alpha_hat_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
    println(alpha_hat_glm)


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 5
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    #using FreqTables
    freqtable(df, :occupation) # note small number of obs in some occupations
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    freqtable(df, :occupation) # problem solved
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation

    function mlogit(alpha, X, d)
        K = size(X,2)
        J = length(unique(d))
        N = length(d)
        bigD = zeros(N,J) 
        for j=1:J
            bigD[:,j] = d.==j
        end
        tmpAlpha = [reshape(alpha,K,J-1) zeros(K)]
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*tmpAlpha[:,j])
            dem .+= num[:,j]
        end
        P = num./repeat(dem,1,J) 
        loglike = -sum( bigD.*log.(P) )
        return loglike
    end
    alpha_start = rand(6*size(X,2))
    alpha_hat_optim = optimize(a -> mlogit(a, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    alpha_hat_mle = alpha_hat_optim.minimizer
    println(alpha_hat_mle)
end


all_ans()