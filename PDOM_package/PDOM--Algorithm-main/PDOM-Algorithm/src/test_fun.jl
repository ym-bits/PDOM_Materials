include("algorithm_utilities.jl")
include("general_utilities.jl")
include("main_algorithm.jl")
include("proximal_operator.jl")
include("test_utilities.jl")

function NRE_Plot(pdom_error,fb_error, mAPG_error,panoc_error,panocplus_error)
    pdom_error = filter(x -> x != 0, pdom_error)
    fb_error = filter(x -> x != 0, fb_error)
    mAPG_error = filter(x -> x != 0, mAPG_error)
    panoc_error = filter(x -> x != 0, panoc_error)
    panocplus_error = filter(x -> x != 0, panocplus_error)

    p1 = plot(1:length(fb_error),fb_error, label="Proximal Gradient")
    plot!(p1, mAPG_error, label="mAPG")
    plot!(p1, panoc_error, label="PANOC")
    plot!(p1, panocplus_error, label="PANOCPLUS")
    plot!(p1, pdom_error, label="PDOM")
    plot!(p1, yscale=:log10, minorgrid=true)
    plot!(p1, legend=:best,legendfontsize=6)
    plot!(p1, xlabel=L"Iteration")
    plot!(p1, ylabel=L"Normalized \quad Recovery \quad Error")

    return p1

end

function Subg_Plot(pdom_error,fb_error, mAPG_error,panoc_error,panocplus_error)
    pdom_error = filter(x -> x != 0, pdom_error)
    fb_error = filter(x -> x != 0, fb_error)
    mAPG_error = filter(x -> x != 0, mAPG_error)
    panoc_error = filter(x -> x != 0, panoc_error)
    panocplus_error = filter(x -> x != 0, panocplus_error)

    p1 = plot(fb_error, label="Proximal Gradient")
    plot!(p1, mAPG_error, label="mAPG")
    plot!(p1, panoc_error, label="PANOC")
    plot!(p1, panocplus_error, label="PANOCPLUS")
    plot!(p1, pdom_error, label="PDOM")
    plot!(p1, yscale=:log10, minorgrid=true)
    plot!(p1, legend=:best,legendfontsize=6)
    plot!(p1, xlabel=L"Iteration")
    plot!(p1, ylabel=L"Subgradient")

    
    return p1

end


function Normal_Nonconvex_LASSO(m,n,k,sig,loop_time = 1)

    for Loop in 1:loop_time
        @show  Loop
        sparse_signal, A, b_noise = generate_compressed_sensing_data(n, k, m,sig)

        x_true = sparse_signal

        _,Hessian_inverse,lambda_max = Hessian_Fun(A'*A)

        smooth(x) = (1/2)*norm(b_noise-A*x)^2

        lambda = 0.02*norm(A'*b_noise, Inf)

        nonsmooth = ProximalOperators.NormL0(lambda)

        x_ini = randn(n)

        fbiter = ProximalAlgorithms.ForwardBackwardIteration(x0=x_ini, f=smooth, g=nonsmooth,gamma = 1/(lambda_max))
        mAPGiter = ProximalAlgorithms.FastForwardBackwardIteration(x0=x_ini, f=smooth, g=nonsmooth,gamma = 1/(lambda_max))
        panociter = ProximalAlgorithms.PANOCIteration(x0=x_ini, f=smooth, g=nonsmooth)
        panocplusiter = ProximalAlgorithms.PANOCplusIteration(x0=x_ini, f=smooth, g=nonsmooth,gamma = 1/(lambda_max))

        alg = Algorithm_Parameters(lambda_max,Hessian_inverse,1e-12,1e-12,1000,1,1.1)

        x_tracking = PDOM(x_ini,smooth,nonsmooth,alg)

        option = 1
        max_iter = 1000
        index_range = 1:length(x_true)

        fb_error,fb_xs,fb_subgradient = Normalized_error(fbiter,x_true,index_range,max_iter,option)
        mAPG_error,mAPG_xs,mAPG_subgradient = Normalized_error(mAPGiter,x_true,index_range,max_iter,option)
        panoc_error,panoc_xs,panoc_subgradient = Normalized_error(panociter,x_true,index_range,max_iter,option)
        panocplus_error,panocplus_xs,panocplus_subgradient = Normalized_error(panocplusiter,x_true,index_range,max_iter,option)
        pdom_error = Normalized_error_PDOM(x_tracking,x_true,index_range,10-15,option)
        pdom_subgradient = subgradient(x_tracking,index_range)
        
        
        p1 = NRE_Plot(pdom_error,fb_error, mAPG_error,panoc_error,panocplus_error)
        
        p2 = Subg_Plot(pdom_subgradient,fb_subgradient, mAPG_subgradient,panoc_subgradient,panocplus_subgradient)
        
        display(plot(p1, p2, layout=(1, 2), size=(1200, 600)))

        @show pdom_error[end]
        @show fb_error[end]
        @show mAPG_error[end]
        @show panoc_error[end]
        @show panocplus_error[end]



    end
    
end

function Normal_Nonconvex_RPCA(m,rank,density,loop_time = 1)

    for Loop in 1:loop_time
        @show  Loop
        X0 = generate_sparse_matrix(m,density)

        X1 = generate_low_rank_matrix(m,m,rank)

        Y = X0+X1

        I_Matrix = Matrix(I,(m,m))
        I_Matrix = hcat(I_Matrix,I_Matrix)
        Hessian = I_Matrix'*I_Matrix

        _,Hessian_inverse,lambda_max = Hessian_Fun(Hessian)

        smooth(x) = (1/2)*norm(Y-I_Matrix*x)^2

        lambda = 1/sqrt(m)

        nonsmooth = Composite_l0_IndRank(lambda,rank)

        x_ini = randn(2*m,m)

        fbiter = ProximalAlgorithms.ForwardBackwardIteration(x0=x_ini, f=smooth, g=nonsmooth,gamma = 1/(lambda_max))
        mAPGiter = ProximalAlgorithms.FastForwardBackwardIteration(x0=x_ini, f=smooth, g=nonsmooth,gamma = 1/(lambda_max))
        panociter = ProximalAlgorithms.PANOCIteration(x0=x_ini, f=smooth, g=nonsmooth)
        panocplusiter = ProximalAlgorithms.PANOCplusIteration(x0=x_ini, f=smooth, g=nonsmooth,gamma = 1/(lambda_max))

        alg = Algorithm_Parameters(lambda_max,Hessian_inverse,1e-12,1e-12,100,1,1.1)

        x_tracking = PDOM(x_ini,smooth,nonsmooth,alg)

        option = 1
        max_iter = 100
        index_range = m+1:2*m

        x_true = X1

        fb_error,fb_xs,fb_subgradient = Normalized_error(fbiter,x_true,index_range,max_iter,option)
        mAPG_error,mAPG_xs,mAPG_subgradient = Normalized_error(mAPGiter,x_true,index_range,max_iter,option)
        panoc_error,panoc_xs,panoc_subgradient = Normalized_error(panociter,x_true,index_range,max_iter,option)
        panocplus_error,panocplus_xs,panocplus_subgradient = Normalized_error(panocplusiter,x_true,index_range,max_iter,option)
        pdom_error = Normalized_error_PDOM(x_tracking,x_true,index_range,10-15,option)
        pdom_subgradient = subgradient(x_tracking,index_range)
        
        
        p1 = NRE_Plot(pdom_error,fb_error, mAPG_error,panoc_error,panocplus_error)
        
        p2 = Subg_Plot(pdom_subgradient,fb_subgradient, mAPG_subgradient,panoc_subgradient,panocplus_subgradient)
        
        display(plot(p1, p2, layout=(1, 2), size=(1200, 600)))

        @show pdom_error[end]
        @show fb_error[end]
        @show mAPG_error[end]
        @show panoc_error[end]
        @show panocplus_error[end]



    end
    
end

function Standard_Nonconvex_LASSO(m,n,k,sig,loop_time = 1)

    for Loop in 1:loop_time
        @show  Loop
        sparse_signal, A, b_noise = generate_compressed_sensing_data(n, k, m,sig)
        b = reshape(b_noise, m, 1)
        x_true = vcat(sparse_signal,sparse_signal)
        x_true = reshape(x_true, n*2, 1)

        beta = 400.0
        Hessian,lambda_max=generate_Hessian(beta,x_true)
        Hessian_inverse = generate_Hessian_inverse(beta,x_true)
        I_Matrix = Matrix(I,(n,n))
        I_Matrix = hcat(I_Matrix,-I_Matrix)
        smooth(x) = (beta/2)*norm(I_Matrix*x)^2+(1/(2*beta))*norm(x)^2
        eigenvalues,Q = eigen(A'*A);
        Q_inv = inv(Q)
        Kappa = eigenvalues
        lambda = 0.01*norm(A'*b, Inf)
        nonsmooth = Composite_l0_Affine(lambda,A,b,Q,Q_inv,Kappa)

        x_ini = 2*randn(2*n,1)

        fbiter = ProximalAlgorithms.ForwardBackwardIteration(x0=x_ini, f=smooth, g=nonsmooth,gamma = 1/(lambda_max))
        mAPGiter = ProximalAlgorithms.FastForwardBackwardIteration(x0=x_ini, f=smooth, g=nonsmooth,gamma = 1/(lambda_max))
        panociter = ProximalAlgorithms.PANOCIteration(x0=x_ini, f=smooth, g=nonsmooth)
        panocplusiter = ProximalAlgorithms.PANOCplusIteration(x0=x_ini, f=smooth, g=nonsmooth,gamma = 1/(lambda_max))

        alg = Algorithm_Parameters(lambda_max,Hessian_inverse,1e-12,1e-12,1000,1,1.1)

        x_tracking = PDOM(x_ini,smooth,nonsmooth,alg)

        option = 1
        max_iter = 1000
        x_true = sparse_signal
        index_range = 1:length(x_true)

        fb_error,fb_xs,fb_subgradient = Normalized_error(fbiter,x_true,index_range,max_iter,option)
        mAPG_error,mAPG_xs,mAPG_subgradient = Normalized_error(mAPGiter,x_true,index_range,max_iter,option)
        panoc_error,panoc_xs,panoc_subgradient = Normalized_error(panociter,x_true,index_range,max_iter,option)
        panocplus_error,panocplus_xs,panocplus_subgradient = Normalized_error(panocplusiter,x_true,index_range,max_iter,option)
        pdom_error = Normalized_error_PDOM(x_tracking,x_true,index_range,10-15,option)
        pdom_subgradient = subgradient(x_tracking,index_range)
        
        
        p1 = NRE_Plot(pdom_error,fb_error, mAPG_error,panoc_error,panocplus_error)
        
        p2 = Subg_Plot(pdom_subgradient,fb_subgradient, mAPG_subgradient,panoc_subgradient,panocplus_subgradient)
        
        display(plot(p1, p2, layout=(1, 2), size=(1200, 600)))

        @show pdom_error[end]
        @show fb_error[end]
        @show mAPG_error[end]
        @show panoc_error[end]
        @show panocplus_error[end]



    end
    
end


function Standard_Nonconvex_RPCA(m,rank,density,loop_time = 1)

    for Loop in 1:loop_time
        @show  Loop
        X0 = generate_sparse_matrix(m,density)

        X1 = generate_low_rank_matrix(m,m,rank)

        Y = X0+X1

        beta = 200.0
        x_ini = 1*rand(4*m,m)

        Hessian,lambda_max=generate_Hessian(beta,x_ini)
        Hessian_inverse = generate_Hessian_inverse(beta,x_ini)

        T = Matrix(I,(m,m))
        T = hcat(T,T)
        Trans = hcat(T,-T)

        smooth(x) = (beta/2)*norm(Trans*x)^2+(1/(2*beta))*norm(x)^2

        A = T'*T
        eigenvalues,Q = eigen(A);
        Q_inv = inv(Q)
        Kappa = Diagonal(eigenvalues)
        lambda = 4

        nonsmooth = Composite_RPCA_L0_IndRank(Y,lambda,rank,T,Q,Q_inv,Kappa)

        fbiter = ProximalAlgorithms.ForwardBackwardIteration(x0=x_ini, f=smooth, g=nonsmooth,gamma = 1/(lambda_max))
        mAPGiter = ProximalAlgorithms.FastForwardBackwardIteration(x0=x_ini, f=smooth, g=nonsmooth,gamma = 0.5/(lambda_max))
        panociter = ProximalAlgorithms.PANOCIteration(x0=x_ini, f=smooth, g=nonsmooth)
        panocplusiter = ProximalAlgorithms.PANOCplusIteration(x0=x_ini, f=smooth, g=nonsmooth,gamma = 1/(lambda_max))

        alg = Algorithm_Parameters(lambda_max,Hessian_inverse,1e-12,1e-12,100,1,1.1)

        x_tracking = PDOM(x_ini,smooth,nonsmooth,alg)

        option = 1
        max_iter = 100
        index_range = 3*m+1:4*m

        x_true = X1

        fb_error,fb_xs,fb_subgradient = Normalized_error(fbiter,x_true,index_range,max_iter,option)
        mAPG_error,mAPG_xs,mAPG_subgradient = Normalized_error(mAPGiter,x_true,index_range,max_iter,option)
        panoc_error,panoc_xs,panoc_subgradient = Normalized_error(panociter,x_true,index_range,max_iter,option)
        panocplus_error,panocplus_xs,panocplus_subgradient = Normalized_error(panocplusiter,x_true,index_range,max_iter,option)
        pdom_error = Normalized_error_PDOM(x_tracking,x_true,index_range,10-15,option)
        pdom_subgradient = subgradient(x_tracking,index_range)
        
        
        p1 = NRE_Plot(pdom_error,fb_error, mAPG_error,panoc_error,panocplus_error)
        
        p2 = Subg_Plot(pdom_subgradient,fb_subgradient, mAPG_subgradient,panoc_subgradient,panocplus_subgradient)
        
        display(plot(p1, p2, layout=(1, 2), size=(1200, 600)))

        @show pdom_error[end]
        @show fb_error[end]
        @show mAPG_error[end]
        @show panoc_error[end]
        @show panocplus_error[end]



    end
    
end