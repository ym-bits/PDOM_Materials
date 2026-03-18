include("algorithm_utilities.jl")

using TSVD

struct Composite_l0_Affine
    lambda::Float64
    A::AbstractArray
    b::AbstractArray
    Q::AbstractArray
    Q_inv::AbstractArray
    Kappa::AbstractArray
end
(h::Composite_l0_Affine)(x) = h.lambda * norm(x[1:Int(length(x)/2)], 0)+ 0.5*norm(h.A*x[Int(length(x)/2)+1:end]-h.b)
function ProximalCore.prox!(y, h::Composite_l0_Affine, x,gamma)
    gl = gamma*h.lambda
    x1 = x[1:Int(length(x)/2)]
    y1 = ones(size(x1))
    for j in 1:Int(length(x)/2)
        
        if abs.(x[j]) <= sqrt(2*gl)
            y[j] = 0
        else
            y[j] = x[j]
        end
    end
    replace!(y1, NaN=>0)
    A = h.A
    b = h.b
    aa = Diagonal(1.0 ./ (h.Kappa .+ (1 / gamma)))
    y[Int(length(x)/2)+1:end] = (h.Q*aa*h.Q_inv)*(A'*b+(1/gamma)*x[Int(length(x)/2)+1:end])
    return h.lambda * norm(y[1:Int(length(x)/2)], 0)
end

struct Composite_RPCA_L0_IndRank
    Y::AbstractArray
    lambda::Float64
    rank::Int
    T::AbstractArray
    Q::AbstractArray
    Q_inv::AbstractArray
    Kappa::AbstractArray
end
(h::Composite_RPCA_L0_IndRank)(x) =  0.5*norm(h.Y-x[1:Int(size(x)[1]/4),:]-x[Int(size(x)[1]/4)+1:Int(size(x)[1]/2),:])^2+h.lambda * norm(x[Int(size(x)[1]/2)+1:Int(size(x)[1]/2)+Int(size(x)[1]/4),:], 0)

function ProximalCore.prox!(y, h::Composite_RPCA_L0_IndRank, x,gamma)
    m,n = size(x)
    y[1:Int(m/2),:] =  (h.Q*inv(h.Kappa+Matrix(I,size(h.Kappa)))*h.Q_inv)*(h.T'*h.Y+x[1:Int(m/2),:])
    gl = gamma*h.lambda
    for i in 1:Int(size(x)[2])

        for j in Int(m/2)+1:Int(m/2)+Int(m/4)
            if abs.(x[j,i]) <= sqrt(2*gl)
                y[j,i] = 0
            else
                y[j,i] = x[j,i]
            end

        end
    end
    U, S, V = tsvd(x[Int(m/2)+Int(m/4)+1:end,:], h.rank)
    M = S .* V'
    y[Int(m/2)+Int(m/4)+1:end,:] = U*M
    return h.lambda * norm(x[Int(size(x)[1]/2)+1:Int(size(x)[1]/2)+Int(size(x)[1]/4),:], 0)+0.5*norm(h.Y-x[1:Int(size(x)[1]/4),:]-x[Int(size(x)[1]/4)+1:Int(size(x)[1]/2),:])^2
end

struct Composite_l0_IndRank
    lambda::Float64
    r::Int
end
(h::Composite_l0_IndRank)(X) = rank(X[Int(size(X)[1]/2)+1:end,:]) > h.r ? eltype(X)(Inf) : eltype(X)(0)+h.lambda * norm(X[1:Int(size(X)[1]/2),:], 0)
function ProximalCore.prox!(y, h::Composite_l0_IndRank, x,gamma)
    gl = gamma*h.lambda
    for i in 1:Int(size(x)[2])
        
        for j in 1:Int(size(x)[1]/2)
            if abs.(x[j,i]) <= sqrt(2*gl)
                y[j,i] = 0
            else
                y[j,i] = x[j,i]
            end
            
        end
    end
    
    U, S, V = tsvd(x[Int(size(x)[1]/2)+1:end,:], h.r)
    M = S .* V'
    y[Int(size(x)[1]/2)+1:end,:] = U*M
    return h.lambda*norm(y[1:Int(size(y)[1]/2),:], 0)
end