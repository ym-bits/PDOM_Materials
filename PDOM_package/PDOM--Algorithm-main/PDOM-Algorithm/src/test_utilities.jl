include("algorithm_utilities.jl")



function generate_sparse_signal(length::Int, nonzero_elements::Int)
    signal = zeros(length)
    indices = randperm(length)[1:nonzero_elements]
    v = [-1.75, 1.75]
    mu = 0.25
    signal[indices] .=rand(v, nonzero_elements) .+ mu*(randn(nonzero_elements))
    return signal
end


function generate_measurement_matrix(rows::Int, cols::Int)

    matrix = randn(rows, cols)

    return matrix

end

function generate_compressed_sensing_data(signal_length::Int, nonzero_elements::Int, num_measurements::Int, noise_level::Float64=0.0)
    sparse_signal = generate_sparse_signal(signal_length, nonzero_elements)

    measurement_matrix = generate_measurement_matrix(num_measurements, signal_length)

    observed_measurements = measurement_matrix * sparse_signal
    
    noise = noise_level * randn(num_measurements)
    noisy_observed_measurements = observed_measurements .+ noise

    return sparse_signal, measurement_matrix, noisy_observed_measurements

end

function generate_sparse_matrix(M::Int, density::Float64)

    X1 = zeros(Float64, M, M)
    num_nonzero = Int(density * M * M)
    indices = randperm(M * M)[1:num_nonzero]

    for index in indices
        row = (index - 1) ÷ M + 1  
        col = (index - 1) % M + 1  
        
        X1[row, col] = rand([-1.0, 1.0])
    end

    return X1

end

function generate_low_rank_matrix(M::Int, N::Int, toy_rank::Int)

    X0 = zeros(M, N)
    r = []

    for _ in 1:toy_rank
        push!(r, rand(1, N))
    end
    
    for i in 1:M
        ind = rand(1) * toy_rank .+ 1
        ind = Int(floor(ind[1]))
        X0[i, :] = r[ind]
    end
    
    X0 .-= mean(X0[:])
    
    return X0

end