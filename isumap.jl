@info "Add packages"
using Pkg; Pkg.add(["CombinatorialSpaces", "NearestNeighbors", "Graphs", "Random", "SparseArrays", "StaticArrays"])

@info "Load packages"
using CombinatorialSpaces
using Graphs
using NearestNeighbors
using Random
using SparseArrays
using StaticArrays

@info "Load in datapoints"
Random.seed!(0);
N = 8
high_dim = 3
# Note: A node considers itself a neighbor of distance 0.
k_neighbors = 3
input_data = [SVector{high_dim, Float64}(rand(high_dim)) for _ in 1:N]

@info "Run KNN"
function run_knn(data)
  # Euclidean distances are assumed by default.
  kd_tree = KDTree(data)
  idxs, dists = knn(kd_tree, data, k_neighbors)
end
idxs, dists = run_knn(input_data)

@info "Normalize distances to neighbors"
function normalize_distances(dists)
  map(dists) do neighborhood_dists
    farthest_neighbor = maximum(neighborhood_dists)
    map(neighborhood_dists) do n
      n / farthest_neighbor
    end
  end
end
M = normalize_distances(dists)

@info "Merge normlized distances into a single matrix"
function merge_normalized_distances(idxs, M)
  R = spzeros(N,N)
  for (i, neighborhood_dists) in enumerate(M)
    for (j, nd) in zip(idxs[i], neighborhood_dists)
      R[i,j] = nd
    end
  end
  R
end
R = merge_normalized_distances(idxs, M)

@info "Convert to a fuzzy simplicial set"
function to_fuzzy_simplicial_set(R)
  T = dropzeros(R)
  for (i,j,v) in zip(findnz(T)...)
    T[i,j] = exp(-v)
  end
  T
end
T = to_fuzzy_simplicial_set(R)

@info "Apply a t-conorm"
function t_conorm(T)
  U = T + T'
  for (i,j,v) in zip(findnz(U)...)
    U[i,j] = min(U[i,j], 1.0)
  end
  U
end
U = t_conorm(T)

@info "Convert to a metric space"
function to_metric_space(U)
  D = dropzeros(U)
  for (i,j,v) in zip(findnz(D)...)
    D[i,j] = -log(v)
  end
  dropzeros!(D)
end
D = to_metric_space(U)

@info "Find all shortest distances"
function shortest_paths(D)
  floyd_warshall_shortest_paths(Graph(D), D).dists
end
shortest_dists = shortest_paths(D)

# At this point, we pass off to any dimension reduction algorithm, so conclude.
@info "Finished Isumap"

