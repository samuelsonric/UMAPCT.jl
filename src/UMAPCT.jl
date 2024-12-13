module UMAPCT


using FileIO
using GeometryBasics
using Graphs
using MeshIO
using Meshes
using MultivariateStats
using NearestNeighbors
using Random; Random.seed!(0);
using SparseArrays
using StaticArrays
using StatsBase


export isoumap


function neighbors(data::AbstractVector, k::Integer)
    knn(KDTree(data), data, k)
end


function normalize!(distances)
    for d in distances
        d ./= maximum(d)
    end
    
    distances
end


function merge(indices, distances)
    n = length(distances)
    matrix = spzeros(n, n)
    
    for (i, d) in enumerate(distances)
        for (j, n) in zip(indices[i], d)
          matrix[i, j] = n
        end
    end
    
    matrix
end


function fuzzysimplicialset!(matrix)
    dropzeros!(matrix)
    
      for (i, j, v) in zip(findnz(matrix)...)
        matrix[i,j] = exp(-v)
      end
    
    matrix
end


function tconorm!(matrix)
    matrix .+= matrix'
    
    for (i, j, v) in zip(findnz(matrix)...)
        matrix[i, j] = min(matrix[i, j], 1.0)
    end
    
    matrix
end


function metricspace!(matrix)
    dropzeros!(matrix)
    
    for (i, j, v) in zip(findnz(matrix)...)
        matrix[i, j] = -log(v)
    end
    
    matrix
end


function shortestpaths(matrix)
    state = floyd_warshall_shortest_paths(Graph(matrix), matrix)
    state.dists
end


function isoumap(data::AbstractVector, k::Integer)
    indices, distances = neighbors(data, k)
    normalize!(distances)
    matrix = merge(indices, distances)
    metricspace!(tconorm!(fuzzysimplicialset!(matrix)))
    shortestdistances = shortestpaths(matrix)
    shortestdistances[isinf.(shortestdistances)] .= 0
    mds = fit(MDS, shortestdistances; distances=true, maxoutdim=2)
    predict(mds)
end


end
