module UMAPCT


using FileIO
using GeometryBasics
using Graphs
using MeshIO
using Distances
using Meshes
using MultivariateStats
using NearestNeighborDescent
using Random
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC
using StaticArrays
using StatsBase
using UMAP


export isoumap, umap


function smoothknndistance(distances::AbstractVector, ρ::Real)
    k = length(distances)

    function f(σ)
        sum(distances) do x
            exp(-(x - ρ) / σ)
        end
    end

    values = range(0, -(last(distances) - ρ) / (log(log2(k)) - log(k)), 100)
    i = searchsortedfirst(map(f, values), log2(k))
    values[i]
end


function metricspaces(data::AbstractMatrix{T}, k::Integer) where T
    indices, distances = knn_matrices(nndescent(data, k, Euclidean()))
    m, n = size(distances)
    matrix = spzeros(T, n, n)

    for i in 1:n
        ρ = distances[1, i]
        σ = smoothknndistance(distances[:, i], ρ)

        for j in 1:m
            matrix[i, indices[j, i]] = (distances[j, i] - ρ) / σ
        end
    end
   
    matrix
end


function singular!(matrix::AbstractSparseMatrixCSC)
    dropzeros!(matrix)
    
      for (i, j, v) in zip(findnz(matrix)...)
        matrix[i,j] = exp(-v)
      end
    
    matrix
end


function merge(tconorm, matrix::AbstractSparseMatrixCSC)
    matrix = tconorm.(matrix, matrix')
    
    for (i, j, v) in zip(findnz(matrix)...)
        matrix[i, j] = min(matrix[i, j], 1.0)
    end
    
    matrix
end


function realize!(matrix::AbstractSparseMatrixCSC)
    dropzeros!(matrix)
    
    for (i, j, v) in zip(findnz(matrix)...)
        matrix[i, j] = -log(v)
    end
    
    matrix
end


function shortestpaths(matrix::AbstractSparseMatrixCSC)
    state = floyd_warshall_shortest_paths(Graph(matrix), matrix)
    state.dists
end


function isoumap(tconorm, data::AbstractMatrix, k::Integer)
    matrix = realize!(merge(tconorm, singular!(metricspaces(data, k))))
    distances = shortestpaths(matrix)
    distances[isinf.(distances)] .= 0
    predict(fit(MDS, distances; distances=true, maxoutdim=2))
end


function umap(tconorm, data::AbstractMatrix, k::Integer)
    matrix = merge(tconorm, singular!(metricspaces(data, k)))
    embedding = UMAP.initialize_embedding(matrix, 2, Val(:spectral))
    reduce(hcat, UMAP.optimize_embedding(matrix, embedding, embedding, 300, 1, 1//10, 1, 1, 5, move_ref=true))
end


end
