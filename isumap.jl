@info "Add packages"
using Pkg; Pkg.add(["FileIO", "Graphs", "MeshIO", "Meshes", "NearestNeighbors", "Random", "SparseArrays", "StaticArrays"])

@info "Load packages"
using FileIO
using Graphs
using MeshIO
using Meshes
using NearestNeighbors
using Random; Random.seed!(0);
using SparseArrays
using StaticArrays

@enum DataScenario random_scenario torus_scenario
scenario = torus_scenario

@info "Load in datapoints"
# Note: A node considers itself a neighbor of distance 0.
high_dim, input_data, N, k_neighbors = if scenario == random_scenario
  high_dim = 3
  N = 8
  input_data = [SVector{high_dim, Float64}(rand(high_dim)) for _ in 1:N]
  k_neighbors = 3
  high_dim, input_data, N, k_neighbors
elseif scenario == torus_scenario
  m = load("./torus.obj")
  high_dim = 3
  input_data = [SVector{high_dim, Float64}(p) for p in m.vertex_attributes.position]
  N = length(input_data)
  k_neighbors = 3
  high_dim, input_data, N, k_neighbors
else
  @error "Unspecified scenario"
end

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

