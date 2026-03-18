include("algorithm_utilities.jl")


Base.@kwdef struct Algorithm_Parameters
    lambda_max::Float64
    Hessian_inverse::AbstractArray
    epsilon_abs::Float64 = 1e-12
    epsilon_rel::Float64 = 1e-12
    maxiter::Int = 1000  
    gamma::Float64  = 0.98
    expansion::Float64 = 1.1
end



function PDOM(x_ini,q,h,alg_para)
    x_intial = x_ini
    gradient_initial = q'(x_intial)

    hessian_k = alg_para.Hessian_inverse
    tau = 1/alg_para.lambda_max
    alpha_pathstep = Ref{Float64}(1.0)
    max_inner = 10

    standard_path = copy(x_intial)
    tau_alpha = Tau_alpha(gradient_initial,standard_path,tau)
    epsilon_abs = alg_para.epsilon_abs
    epsilon_rel = alg_para.epsilon_rel
    beta = alg_para.expansion
    gamma = alg_para.gamma
    
     # --- Data collection ---
    optimality_tracking = []
    x_tracking = []
    push!(x_tracking,x_intial)

    # --- Predefine variables ---
    x_t = randn(size(x_intial))
    
    x = x_intial
    gradient_alpha = randn(size(x_intial))
        
    # === Main loop ===
    optimality_condition = Ref{Int}(1)
    iteration = 1

    while optimality_condition[] == 1 && iteration < alg_para.maxiter
        l = 1
        x_k = x
        gradient_k = q'(x_k)
        q_k = q(x_k)
        Flag = Ref{Int}(1)
        while Flag[] == 1
            
            standard_path,path_gamma = Dogleg_path(gradient_k,hessian_k,alpha_pathstep[],tau,gamma)
            gradient_alpha,tau_alpha = Gradient_Tau_alpha(gradient_k,standard_path,tau)
            x_t,_ = prox(h, x_k+path_gamma,gamma*tau_alpha)
            q_t = q(x_t)
            model_q = Model_q_G(q_k,gradient_alpha,x_k,x_t,tau_alpha)
            comparison_logic!(model_q, q_t, Flag, alpha_pathstep, beta)
            l = termination_logic!(l, max_inner, Flag)

        end

        objective_x_t = q(x_t)+h(x_t)
        x_l,_= prox(h,x_k-tau*gradient_k,tau)
        objective_x_l = q(x_l)+h(x_l)

        if objective_x_t >= objective_x_l
            x_t = x_l
            alpha_pathstep = Ref{Float64}(1.0)
        end

        gradient_x_t = q'(x_t)
        optimality = norm((x_t-x_k))
        optimality_bound = complex_optimality_logic!(iteration, optimality,optimality_tracking, x_t, gradient_x_t, gradient_alpha, tau_alpha, gamma, epsilon_abs, epsilon_rel, optimality_condition)
        
        x = x_t
        

        push!(x_tracking,x)
        
        push!(optimality_tracking,optimality_bound)

        iteration = iteration + 1

    end

    return x_tracking

end