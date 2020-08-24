using Random
using DelimitedFiles

data, header = readdlm("CASP.csv", ',', Float64, header=true)

# get data
y = data[:,1]
x = data[:,2:end]

N,D = size(x)

@assert D == 9

idx = collect(1:N)

Random.seed!(2020)
shuffle!(idx)

# split into train and test set
Ntrain = Int(round(N * 0.7))
train = idx[1:Ntrain]

test = idx[(Ntrain+1):N]

@info "Split data into Ntrain: $Ntrain, Ntest: $(length(test))"

# save to disk
open("x_train.csv", "w") do io
    writedlm(io, x[train,:], ',')
end

open("y_train.csv", "w") do io
    writedlm(io, y[train,:], ',')
end

open("x_test.csv", "w") do io
    writedlm(io, x[test,:], ',')
end

open("y_test.csv", "w") do io
    writedlm(io, y[test,:], ',')
end
