include("algorithm_utilities.jl")

function Normalized_error_PDOM(x_tracking,x_true,index_range,tol,option = 1)
    normalized_error = []
    subgradient = []
    x_now = x_old = x_tracking[1]
    xs = []
    for x in x_tracking
        if option == 1
            error = norm(x[index_range,:]-x_true)/norm(x_true)
            push!(normalized_error,copy(error))
        else
            error = 20*log10(norm(x[index_range,:]-x_true)/norm(x_true))
            push!(normalized_error,copy(error))
        end
            
        push!(xs, copy(x))
        
        if length(xs) > 1 && norm(xs[end] - xs[end - 1]) / (1 + norm(xs[end])) <= tol
            break
        end
    end
    
    return normalized_error
end



function Normalized_error(solver_iter,x_true,index_range,max_iter=1000,option = 1)
    normalized_error = []
    xs = []
    iter = 1
    for state in solver_iter
        if option == 1      
            error = norm(state.x[index_range,:]-x_true)/norm(x_true)
            push!(normalized_error,copy(error))
        else
            error = 20*log10(norm(state.x-x_true)/norm(x_true))
            push!(normalized_error,copy(error))
        end
            
        push!(xs, copy(state.x))
        
        if length(xs) > 1 
                        if norm(xs[end] - xs[end - 1]) / (1 + norm(xs[end])) <= 1e-15 || iter >= 2000
                    break
                end
        end
        iter = iter + 1
        if iter == max_iter
            break
        end
    end

    function subgradient(x,index_range)
        subgradient = []
        x_now = x_old = x[1]
        
        for i in 1:length(x)-1
            x_now = x[i+1]
            partial_f = norm(x_now[index_range,:]-x_old[index_range,:])
            push!(subgradient,copy(partial_f))
            x_old = x_now
            if partial_f < 1e-7
                break
            end
        end
        return subgradient
    end

    subgradient = subgradient(xs,index_range)
    
    return normalized_error,xs,subgradient
end

function subgradient(x,index_range)
    subgradient = []
    x_now = x_old = x[1]
    
    for i in 1:length(x)-1
        x_now = x[i+1]
        partial_f = norm(x_now[index_range,:]-x_old[index_range,:])
        push!(subgradient,copy(partial_f))
        x_old = x_now
        if partial_f < 1e-7
            break
        end
    end
    return subgradient
end