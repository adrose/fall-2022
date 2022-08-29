# Run this code if first time executing script
#using Pkg
#Pkg.add("JLD")
#Pkg.add("JLD2")
#Pkg.add("Random")
#Pkg.add("LinearAlgebra")
#Pkg.add("Statistics")
#Pkg.add("CSV")
#Pkg.add("DataFrames")
#Pkg.add("FreqTables")
#Pkg.add("Distributions")

# Load libraries
using JLD2, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions


###############
# Q1 function
###############
function q1()
    Random.seed!(1234);
    A = rand(Uniform(-5,10), 10, 7);
    B = rand(Normal(-2, 15), 10, 7);
    C = cat(A[1:5,1:5],B[1:5,end-1:end]; dims=2);
    D = A;
    indiciesLess = (D.<=0);
    D = D.*indiciesLess;
    lengthA = length(A);
    uniqueD = length(unique(D));
    E = reshape(B, length(B));
    F = cat(A,B, dims=3);
    F = permutedims(F, [3 1 2]);
    G = kron(B, C);
    #G2 = kron(C,F);
## Provides the following error:
#ERROR: MethodError: no method matching kron(::Matrix{Float64}, ::Array{Float64, 3})
#Closest candidates are:
#kron(::Any, ::Any, ::Any, ::Any...) at operators.jl:591
#kron(::AbstractVecOrMat, ::Number) at ~/Documents/julia-1.8.0/share/julia/stdlib/v1.8/LinearAlgebra/src/dense.jl:445
#kron(::VecOrMat, ::Union{SparseArrays.AbstractSparseMatrixCSC, SparseArrays.SparseVector, Union{Adjoint{var"#s884", var"#s883"}, Transpose{var"#s884", var"#s883"}} where {var"#s884", var"#s883"<:SparseArrays.AbstractSparseMatrixCSC}}) at ~/Documents/julia-1.8.0/share/julia/stdlib/v1.8/SparseArrays/src/linalg.jl:1435
## Can't compute the kron of a 3d matrix?
    save("matrixpractice.jld","A",A,"B",B,"C",C,"D",D,"E",E,"F",F,"G",G);
    save("firstmatrix.jld","A",A,"B",B,"C",C,"D",D);
    CSV.write("Cmatrix.csv",DataFrame(C, :auto));
    CSV.write("Dmatrix.dat",DataFrame(D, :auto); delim="\t");
    return A,B,C,D
end

###############
# Q2 function here
###############
function q2(A,B,C)
    AB = zeros(size(A));
    iterLength = length(A)
    for i in 1:iterLength 
        #println(i)
        tmpVal = A[i] * B[i]
        AB[i] = tmpVal 
    end
    ## sans for loop
    AB2 = A.*B;
    ## For loop here
    Cprime = zeros(size(C));
    iterLength = length(C);
    for i in 1:iterLength
     #println(i)
     tmpVal = C[i];
     if tmpVal <=5 && tmpVal >= -5
         Cprime[i] = tmpVal;
     end
    end
    ## sans for loop
    indexVals = (C.>=-5) .& (C.<=5);
    Cprime2 = C[indexVals];
    N = 15169
    K = 6
    T = 5
    X = cat([cat([ones(N,1) rand(N,1).<=(0.75*(6-t)/5) (15+t-1).+(5*(t-1)).*randn(N,1) (π*(6-t)/3).+(1/exp(1)).*randn(N,1) rand(Binomial(20,0.6),N) rand(Binomial(20,0.5),N)];dims=3) for t=1:T]...;dims=3)
    β = vcat([cat([1+0.25*(t-1) log(t) -sqrt(t) exp(t)-exp(t+1) t t/3];dims=1) for t=1:T]...)'
    Y = hcat([cat(X[:,:,t]*β[:,t] + .36*randn(N,1);dims=2) for t=1:T]...) 
   return nothing
end

###############
# Q3 Here
###############
function q3()
    nlsw88 = DataFrame(CSV.File("./ProblemSets/PS1-julia-intro/nlsw88.csv"))
    save("nlsw88.jld","nlsw88",nlsw88)
    perNM = mean(nlsw88.never_married) # About 10% have neveer been married
    perCG = mean(nlsw88.collgrad) # About 24% are college grads
    freqRACE = freqtable(nlsw88, :race)
    perRace = freqRACE ./ size(nlsw88, 1)
    #1     │  0.728851 ## Not sure of coding but here are the breakdown by factor level
    #2     │  0.259573
    #3     │ 0.0115761
    summarystats = describe(unique(nlsw88))
    # 1 │ idcode         2612.65       1         2614.0      5159              0  Int64 ## Here is the table it returns.. 
    # 2 │ age              39.1532    34           39.0        46              0  Int64
    # 3 │ race              1.28272    1            1.0         3              0  Int64
    # 4 │ married           0.64203    0            1.0         1              0  Int64
    # 5 │ never_married     0.104185   0            0.0         1              0  Int64
    # 6 │ grade            13.0989     0           12.0        18              2  Union{Missing, Int64} ## 2 grade values are missing
    # 7 │ collgrad          0.236866   0            0.0         1              0  Int64
    # 8 │ south             0.419412   0            0.0         1              0  Int64
    # 9 │ smsa              0.703918   0            1.0         1              0  Int64
    #10 │ c_city            0.29163    0            0.0         1              0  Int64
    #11 │ industry          8.18952    1            8.0        12             14  Union{Missing, Int64}
    #12 │ occupation        4.64283    1            3.0        13              9  Union{Missing, Int64}
    #13 │ union             0.245474   0            0.0         1            368  Union{Missing, Int64}
    #14 │ wage              7.76695    1.00495      6.27227    40.7466         0  Float64
    #15 │ hours            37.2181     1           40.0        80              4  Union{Missing, Int64}
    #16 │ ttl_exp          12.535      0.115385    13.125      28.8846         0  Float64
    #17 │ tenure            5.97785    0.0          3.83333    25.9167        15  Union{Missing, Float64}
    freqtable(nlsw88, :industry, :occupation) 
    #industry ╲ occupation │       1        2        3        4        5        6        7        8        9       10       11       12       13  missing ## Here is the output table
    #──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    #1                     │       1        1        6        1        0        2        0        0        1        5        0        0        0        0
    #2                     │       1        0        3        0        0        0        0        0        0        0        0        0        0        0
    #3                     │       1        5       16        1        0        2        0        1        0        0        3        0        0        0
    #4                     │      24       37       73       13       29      185        0        1        0        0        5        0        0        0
    #5                     │       4        9       49        2        5       12        0        6        0        0        2        0        0        1
    #6                     │       7       65      113       50       10       23        0       58        0        2        2        0        2        1
    #7                     │       8       56       95       29        0        0        0        4        0        0        0        0        0        0
    #8                     │      13        9       39        4        1        1        0       16        0        0        2        0        1        0
    #9                     │       2        5       14        1        1       12       28       33        0        0        0        0        0        1
    #10                    │       2        2        4        1        0        1        0        6        0        0        1        0        0        0
    #11                    │     208       56      213        0        5        5        0      155        0        0        0        0      182        0
    #12                    │      45       18      100        0        2        1        0        5        0        0        1        2        1        1
    #missing               │       1        1        1        0        0        2        0        1        0        2        0        0        1        5
    wageonly = nlsw88[:,[:industry,:occupation,:wage]] 
    grouper = groupby(wageonly, [:industry,:occupation]) 
    combine(grouper, valuecols(grouper) .=> mean) ## output df is 99x3
    return nothing
end

###############
# Q4 Here
###############
function q4()
    mats = load("firstmatrix.jld")
    A = mats["A"]
    B = mats["B"]
    C = mats["C"]
    D = mats["D"]
    function matrixops(m1,m2)
        # This function will will perform three operations on two matricies of equal dim(A & B)
        # op 1: returns the elemebt-by-element product of the inputs
        # op 2: returns the product of A-trnaspose Binomial
        # op 3: returns the sum of A + B
        if size(m1)!=size(m2)
            error("inputs must have the same size.")         end
        ret1 = m1.*m2
        ret2 = m1'*m2 
        ret3 = sum(m1+m2)
        return ret1,ret2,ret3
    end

    matrixops(A,B) 
    #matrixops(C,D) ## Returns the following error:
    #ERROR: inputs must have the same size.
    #Stacktrace:
    # [1] error(s::String)
    #   @ Base ./error.jl:35
    # [2] matrixops(m1::Matrix{Float64}, m2::Matrix{Float64})
    #   @ Main ~/Documents/fall-2022/ProblemSets/PS1-julia-intro/ps1.jl:152
    # [3] top-level scope
    #   @ ~/Documents/fall-2022/ProblemSets/PS1-julia-intro/ps1.jl:160
    nlsw88 = DataFrame(CSV.File("./ProblemSets/PS1-julia-intro/nlsw88.csv"))
    matrixops(convert(Array,nlsw88.ttl_exp),convert(Array,nlsw88.wage))
    return nothing
end

## Run code here
A,B,C,D = q1()
q2(A,B,C)
q3()
q4()