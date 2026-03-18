using DelimitedFiles
using LinearAlgebra
using ProximalOperators
using ProximalAlgorithms
using Plots
using LaTeXStrings
using ColorSchemes
using StatsBase
using Distributions
using Random
using Noise
using LinearMaps
using BenchmarkTools
using ProximalCore

function split_half(data::AbstractArray)
    n = length(data)
    midpoint = ceil(Int, n / 2)
    
    if isa(data, Vector)
        return data[1:midpoint], data[midpoint+1:end]
    elseif isa(data, Matrix)
        rows, _ = size(data)
        row_midpoint = ceil(Int, rows / 2)
        
        top_half = data[1:row_midpoint, :]
        bottom_half = data[row_midpoint+1:end, :]
        
        return top_half, bottom_half
    else
        error("Unsupported data type")
    end
end


function Hessian_Fun(H::AbstractArray)

    lambda_max = maximum(eigen(H).values)

    H_inverse = H

    try
        H_inverse =  inv(H) 
    catch
        H_inverse = inv(H+0.0001*Matrix(I,size(H)))
    end

    return H,H_inverse,lambda_max

end

function generate_Hessian(beta::Float64,x::AbstractArray)
    m,_ = size(x)
    m = Int(m/2)
    II = Matrix{Float64}(I, m, m)  # Identity matrix of size 2x2

    # Calculate the coefficients
    a = beta+1/beta
    b = -beta

    # Construct the matrix blocks
    A = a * II
    B = b * II

    # Combine the blocks into the final matrix
    Q = [A B; B A]

    H = [a b;b a]

    lambda_max = maximum(eigen(H).values)

    return Q,lambda_max
end


function generate_Hessian_inverse(beta::Float64,x::AbstractArray)
    m,n = size(x)
    m = Int(m/2)
    II = Matrix{Float64}(I, m, m)  # Identity matrix of size mxm

    # Calculate the coefficients
    a = (beta * (beta^2 + 1)) / (2 * beta^2 + 1)
    b = beta^3 / (2 * beta^2 + 1)

    # Construct the matrix blocks
    A = a * II
    B = b * II

    # Combine the blocks into the final matrix
    Q_inv = [A B; B A]

    return Q_inv
end

function Smooth_q(x::AbstractArray, beta_relax::Float64)
    # Split the vector x into two halves
    x1, x2 = split_half(x)

    # Compute the smoothness function value
    value = (beta_relax / 2) * norm(x1 - x2)^2 + (1 / (2 * beta_relax)) * (norm(x1)^2 + norm(x2)^2)

    return value
end

function Dogleg_path(gradient_k::AbstractArray, hessian_inv::AbstractArray, alpha_pathstep, tau::Float64,gamma::Float64)

    # Compute p_tau, the step in the steepest descent direction scaled by tau
    p_tau = -tau * gradient_k

    # Compute p_N, the Newton step
    p_N = -hessian_inv * gradient_k

    # Compute the dogleg path
    dogleg_path = p_tau + alpha_pathstep * (p_N - p_tau)

    path_gamma = dogleg_path*gamma

    return dogleg_path,path_gamma
end

function Dogleg_path_gamma(standard_path::AbstractArray,gamma::Float64)
    
    path_gamma = standard_path*gamma

    return path_gamma
end

function Gradient_Tau_alpha(gradient_k::AbstractArray,path::AbstractArray,tau)

    gradient = 1/(norm(path)^2)*(path'*gradient_k)[1]*path

    tau_alpha = -(norm(path)^2)/(path'*gradient_k)[1]
    tau_alpha = Ref{Float64}(tau_alpha)

    sign_logic!(tau_alpha, tau)

    return gradient,tau_alpha[]

end


function Gradient_alpha(gradient_k::AbstractArray,path::AbstractArray)
    gradient = 1/(norm(path)^2)*(path'*gradient_k)[1]*path
    return gradient 
end

function Tau_alpha(gradient_k::AbstractArray,path::AbstractArray,tau)

    tau_alpha = -(norm(path)^2)/(path'*gradient_k)[1]
    tau_alpha = Ref{Float64}(tau_alpha)

    sign_logic!(tau_alpha, tau)
    
    return tau_alpha[]
end

function Model_q_G(q_k,gradient_alpha::AbstractArray,x_k::AbstractArray,x_update::AbstractArray,tau_alpha)

    value = q_k + ((x_update-x_k)'*gradient_alpha)[1]+1/(2*tau_alpha)*norm(x_update-x_k)^2

    return value
end

function termination_logic!(α::Int, β::Int, γ::Ref{Int})
    θ = α + 1  # Increment α
    ψ = β  # Assign β to ψ for clarity in logic
    if θ > ψ
        γ[] = 0  # Update γ using a reference
    end
    return θ
end

function comparison_logic!(α::Float64, β::Float64, γ::Ref{Int}, δ::Ref{Float64}, ε::Float64)
    if α >= β
        γ[] = 0  # Update γ using a reference
    else
        δ[] = δ[] / ε  # Update δ using a reference
    end
end

function sign_logic!(α::Float64, β::Float64)
    if α[] < 0
        α[] = β
    end
end


function complex_optimality_logic!(iter::Int, opt::Float64, opt_track, x_t::AbstractArray, grad_x_t::AbstractArray, grad_alpha::AbstractArray, tau_alpha::Float64, gamma::Float64, epsilon_abs::Float64, epsilon_rel::Float64, opt_cond::Ref{Int})
    # Calculate the optimality bound
    opt_bound = sqrt(length(x_t)) * epsilon_abs + epsilon_rel * maximum([norm(grad_x_t), norm(grad_alpha), 1/gamma * tau_alpha * norm(x_t), 1/gamma * tau_alpha * norm(x_t)])

    # Check the optimality condition
    if iter > 1
        if norm(opt_bound - opt_track[end])^2 < 1e-35 || opt <= opt_bound
            opt_cond[] = 0
        end
    end

    return opt_bound
end


