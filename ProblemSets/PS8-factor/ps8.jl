using MultivariateStats
using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using Distributions
using DataFrames
using DataFramesMeta
using CSV

all_ans = function()
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    df = DataFrame(CSV.File("./ProblemSets/PS8-factor/nlsy.csv"))
    mod = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    cor_mat = cor(Matrix(select(df, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK])))

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    mod2 = lm(@formula(logwage~black+hispanic+female+schoolt+gradHS+grad4yr+asvabAR+asvabCS+asvabMK+asvabNO+asvabPC+asvabWK), df)
    ## Yes -- including all of these variables with this amount of collinearity will certainly disrupt the infenetial capability of this model
    ## With this amount of multicollinearity, bouncing beta's are surley to manifest -- this will lead to the incorrect sign on coefficents, as well as a reduction
    ## in the "true" significance of the parameter as the standard errors will increase

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 4
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    asvabMat = Matrix(select(df, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]))
    M = fit(PCA, asvabMat; maxoutdim=1)
    df.pcaVals = loadings(M)
    mod3 = lm(@formula(logwage~black+hispanic+female+schoolt+gradHS+grad4yr+pcaVals), df)
    ## Not very significant first component?? -- am I doing this wrong?

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 5
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    F = fit(FactorAnalysis, asvabMat; maxoutdim=1)
    df.faVals = loadings(M)
    mod3 = lm(@formula(logwage~black+hispanic+female+schoolt+gradHS+grad4yr+faVals), df)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 6
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    ## Needs to include:
    # ASVAB scores
    # Demo vars
    # log wage
    # Covaraite vars
    ## Prep all vars here
    asvabMat = Matrix(select(df, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]))
    df.Int = ones(size(df,1))
    demoMat = Matrix(select(df, [:Int, :black, :hispanic,:female]))
    logWage = df.logwage
    covarsMat = Matrix(select(df, [:schoolt,:gradHS,:grad4yr]))
    covarsMat = Matrix(select(df, [:schoolt,:gradHS,:grad4yr, :black, :hispanic,:female]))
    ## Create a function to esimtate the measurment model params
    facModel = function(asvabMat,demoMat,demoParms,sigmaJ,gamma,draws)
        P = size(asvabMat, 2)
        N = size(asvabMat, 1)
        Z = size(demoMat, 2)
        R = size(draws, 1)
        alpha = reshape(demoParms, Z, P)
        out_mat = zeros(N,P)
        for r=1:R
            for p=1:P
                vals = (asvabMat[:,p] .- (demoMat*alpha[:,p] .+ gamma[p]*draws[r])) ./ sigmaJ
                mod_lik = 1 / sigmaJ .* pdf.(Normal(0,1), vals)
                out_mat[:,p] = mod_lik
            end 
        end
        return(out_mat)
    end
    regModel = function(yVar, otherCovs, betaWeights, sigmaW, gammaVal,draws)
        N = size(otherCovs, 1)
        out_mat = zeros(N,R)
        #draws = 8 .* rand(R).-4
        R = size(draws, 1)
        for r=1:R
            vals = yVar .-(otherCovs*betaWeights .+ gammaVal*draws[r])
            mod_lik = (1/sigmaW) .* pdf.(Normal(0,1), vals./sigmaW)
            out_mat[:,r] = mod_lik
        end
        return(out_mat)
    end

    organizeAll = function(y, asvabMat, demoMat, otherCovs, R, allParams)
        ## Organize all of the parameter values
        ## Start with the measurment model params
        demoParams = allParams[1:24]
        betaWeights = allParams[25:31]
        gammaFac = allParams[32:37]
        sigmaJ = allParams[38:44]
        sigmaW = allParams[45]
        ## Now esitmate the like values
        fac_like = facModel(asvabMat,demoMat,demoParams,gammaFac,sigmaJ,draws)
        fac_like = prod(fac_like, dims=2)
        reg_like = regModel(y, otherCovs, betaWeights, sigmaW, rand(1), draws)
        ## Now idenitfy the liklihood
        out_like = zeros(N)
        for r = 1:R
            out_like += draws[r].*prod(Mlike; dims=2).*Ylike[:,r].*pdf(Normal(0,1),draws[r])
        end
        return(-out_like)
    end

    organizeAll(logWage, asvabMat, demoMat, otherCovs, 10,000, rand(45))
    ## Now optimize
    rand_start = rand(45)
    op_vals = optimize(organizeAll(logWage, asvabMat, demoMat, otherCovs, 10,000,rand_start), rand_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100, show_trace=true, show_every=50))
end
all_ans()


    
