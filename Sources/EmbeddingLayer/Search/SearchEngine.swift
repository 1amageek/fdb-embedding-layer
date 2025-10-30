import Foundation
import Logging
#if canImport(Accelerate)
import Accelerate
#endif

/// High-performance vector similarity search engine
///
/// The SearchEngine actor provides thread-safe similarity search capabilities
/// for vector embeddings stored in FoundationDB. It implements brute-force
/// search with multiple similarity metrics.
///
/// ## Key Features
/// - Multiple similarity metrics (cosine, inner product, Euclidean, Manhattan)
/// - Batch search support for multiple queries
/// - Optional result filtering
/// - Actor-isolated for thread safety
/// - SIMD-ready architecture (Accelerate framework hints)
///
/// ## Performance Characteristics
/// - **Time Complexity**: O(n × d) per query, where n = number of vectors, d = dimension
/// - **Space Complexity**: O(k) for top-k results storage
/// - **Scalability**: Suitable for up to ~100K vectors; consider approximate methods beyond that
///
/// ## Usage Example
/// ```swift
/// let engine = SearchEngine(store: embeddingStore, logger: logger)
///
/// // Single query search
/// let results = try await engine.search(
///     queryVector: queryEmbedding,
///     topK: 10,
///     model: "text-embedding-3-small",
///     metric: .cosine
/// )
///
/// // Batch search
/// let batchResults = try await engine.batchSearch(
///     queryVectors: [query1, query2, query3],
///     topK: 10,
///     model: "text-embedding-3-small",
///     metric: .cosine
/// )
/// ```
///
/// ## Similarity Metrics
/// - **Cosine**: Range [0, 1], higher is better. Best for normalized embeddings.
/// - **Inner Product**: Range (-∞, +∞), higher is better. Fast but requires normalized vectors.
/// - **Euclidean**: Range [0, +∞), lower is better. True geometric distance.
/// - **Manhattan**: Range [0, +∞), lower is better. L1 distance, more outlier-robust.
///
/// ## Future Optimizations
/// - SIMD acceleration via Accelerate framework
/// - Approximate nearest neighbor (ANN) algorithms (HNSW, IVF)
/// - GPU acceleration for large-scale search
/// - Parallel batch processing
public actor SearchEngine {

    // MARK: - Properties

    /// Reference to the embedding store for data access
    private let store: EmbeddingStoreProtocol

    /// Logger for debugging and monitoring
    private let logger: Logging.Logger

    // MARK: - Initialization

    /// Initialize search engine with embedding store and logger
    ///
    /// - Parameters:
    ///   - store: The embedding store to query vectors from
    ///   - logger: Optional logger instance. Defaults to label "com.embedding.search"
    public init(
        store: EmbeddingStoreProtocol,
        logger: Logging.Logger? = nil
    ) {
        self.store = store
        self.logger = logger ?? Logging.Logger(label: "com.embedding.search")
    }

    // MARK: - Public Search Methods

    /// Perform similarity search for a single query vector
    ///
    /// Searches all embeddings in the specified model's namespace and returns
    /// the top-k most similar vectors according to the specified metric.
    ///
    /// ## Algorithm
    /// 1. Retrieve all embeddings for the model from store
    /// 2. Validate query vector dimension matches model dimension
    /// 3. Compute similarity score for each stored vector
    /// 4. Apply optional filter function
    /// 5. Sort by score (descending for similarity, ascending for distance)
    /// 6. Return top-k results
    ///
    /// ## Time Complexity
    /// - O(n × d): where n = number of vectors, d = dimension
    /// - Sorting: O(n log n) if not using partial sort
    /// - Filter: O(n) additional if filter provided
    ///
    /// ## Parameters
    /// - queryVector: The query embedding vector to search for
    /// - topK: Number of top results to return (default: 10)
    /// - model: Model name to search within (namespace isolation)
    /// - metric: Similarity metric to use (default: .cosine)
    /// - filter: Optional predicate to filter results before ranking
    ///
    /// ## Returns
    /// Array of SearchResult sorted by score (best matches first)
    ///
    /// ## Throws
    /// - `EmbeddingError.modelNotFound`: If model doesn't exist in store
    /// - `EmbeddingError.dimensionMismatch`: If query vector dimension doesn't match model
    /// - `EmbeddingError.searchError`: For other search-related errors
    ///
    /// ## Example
    /// ```swift
    /// let results = try await search(
    ///     queryVector: [0.1, 0.2, 0.3, ...],
    ///     topK: 5,
    ///     model: "text-embedding-3-small",
    ///     metric: .cosine,
    ///     filter: { $0.sourceType == .entity }
    /// )
    /// ```
    public func search(
        queryVector: [Float],
        topK: Int = 10,
        model: String,
        metric: SimilarityMetric = .cosine,
        filter: ((EmbeddingRecord) -> Bool)? = nil
    ) async throws -> [SearchResult] {
        logger.debug("Starting search: model=\(model), topK=\(topK), metric=\(metric)")

        // Validate inputs
        guard topK > 0 else {
            throw EmbeddingError.searchError("topK must be positive, got \(topK)")
        }

        guard !queryVector.isEmpty else {
            throw EmbeddingError.invalidVector("Query vector cannot be empty")
        }

        // Get model metadata to validate dimension
        let modelMetadata = try await store.getModelMetadata(model: model)
        guard let metadata = modelMetadata else {
            throw EmbeddingError.modelNotFound(model)
        }

        // Validate dimension
        guard queryVector.count == metadata.dimension else {
            throw EmbeddingError.dimensionMismatch(
                expected: metadata.dimension,
                actual: queryVector.count
            )
        }

        // Retrieve all embeddings for this model
        logger.debug("Retrieving all embeddings for model: \(model)")
        let allEmbeddings = try await store.getAllEmbeddings(model: model)

        if allEmbeddings.isEmpty {
            logger.info("No embeddings found for model: \(model)")
            return []
        }

        logger.debug("Found \(allEmbeddings.count) embeddings to search")

        // Compute similarity scores for all embeddings
        var scoredResults: [(record: EmbeddingRecord, score: Float)] = []
        scoredResults.reserveCapacity(allEmbeddings.count)

        for embedding in allEmbeddings {
            // Apply filter if provided
            if let filter = filter, !filter(embedding) {
                continue
            }

            // Compute similarity score
            let score = try computeSimilarity(
                v1: queryVector,
                v2: embedding.vector,
                metric: metric
            )

            scoredResults.append((record: embedding, score: score))
        }

        logger.debug("Computed \(scoredResults.count) similarity scores")

        // Sort and take top-k
        let sortedResults = filterAndSort(results: scoredResults, topK: topK, metric: metric)

        // Convert to SearchResult
        let searchResults = sortedResults.map { result in
            SearchResult(
                id: result.record.id,
                score: result.score,
                vector: result.record.vector,
                sourceType: result.record.sourceType,
                metadata: result.record.metadata
            )
        }

        logger.info("Search completed: returned \(searchResults.count) results")
        return searchResults
    }

    /// Perform batch similarity search for multiple query vectors
    ///
    /// Efficiently searches for multiple queries in a single call. This is more
    /// efficient than calling search() multiple times because it loads embeddings
    /// only once.
    ///
    /// ## Algorithm
    /// 1. Validate all query vectors have same dimension
    /// 2. Retrieve all embeddings for model once
    /// 3. For each query vector:
    ///    - Compute similarities against all stored vectors
    ///    - Sort and take top-k
    /// 4. Return results for all queries
    ///
    /// ## Time Complexity
    /// - O(q × n × d): where q = number of queries, n = number of vectors, d = dimension
    /// - More efficient than q separate search() calls due to single data retrieval
    ///
    /// ## Parameters
    /// - queryVectors: Array of query embedding vectors
    /// - topK: Number of top results per query (default: 10)
    /// - model: Model name to search within
    /// - metric: Similarity metric to use (default: .cosine)
    ///
    /// ## Returns
    /// Array of result arrays, one per query vector (same order as input)
    ///
    /// ## Throws
    /// - `EmbeddingError.modelNotFound`: If model doesn't exist
    /// - `EmbeddingError.dimensionMismatch`: If any query vector dimension doesn't match
    /// - `EmbeddingError.searchError`: For other search-related errors
    ///
    /// ## Example
    /// ```swift
    /// let results = try await batchSearch(
    ///     queryVectors: [query1, query2, query3],
    ///     topK: 10,
    ///     model: "text-embedding-3-small",
    ///     metric: .cosine
    /// )
    /// // results[0] contains top-10 for query1
    /// // results[1] contains top-10 for query2
    /// // results[2] contains top-10 for query3
    /// ```
    public func batchSearch(
        queryVectors: [[Float]],
        topK: Int = 10,
        model: String,
        metric: SimilarityMetric = .cosine
    ) async throws -> [[SearchResult]] {
        logger.debug("Starting batch search: model=\(model), queries=\(queryVectors.count), topK=\(topK)")

        // Validate inputs
        guard !queryVectors.isEmpty else {
            throw EmbeddingError.searchError("Query vectors array cannot be empty")
        }

        guard topK > 0 else {
            throw EmbeddingError.searchError("topK must be positive, got \(topK)")
        }

        // Get model metadata
        let modelMetadata = try await store.getModelMetadata(model: model)
        guard let metadata = modelMetadata else {
            throw EmbeddingError.modelNotFound(model)
        }

        // Validate all query vectors have correct dimension
        for (index, vector) in queryVectors.enumerated() {
            guard vector.count == metadata.dimension else {
                throw EmbeddingError.dimensionMismatch(
                    expected: metadata.dimension,
                    actual: vector.count
                )
            }

            guard !vector.isEmpty else {
                throw EmbeddingError.invalidVector("Query vector at index \(index) is empty")
            }
        }

        // Retrieve all embeddings once (shared across all queries)
        logger.debug("Retrieving all embeddings for model: \(model)")
        let allEmbeddings = try await store.getAllEmbeddings(model: model)

        if allEmbeddings.isEmpty {
            logger.info("No embeddings found for model: \(model)")
            // Return empty results for each query
            return Array(repeating: [], count: queryVectors.count)
        }

        logger.debug("Processing \(queryVectors.count) queries against \(allEmbeddings.count) embeddings")

        // Process each query
        var batchResults: [[SearchResult]] = []
        batchResults.reserveCapacity(queryVectors.count)

        for (queryIndex, queryVector) in queryVectors.enumerated() {
            // Compute similarities for this query
            var scoredResults: [(record: EmbeddingRecord, score: Float)] = []
            scoredResults.reserveCapacity(allEmbeddings.count)

            for embedding in allEmbeddings {
                let score = try computeSimilarity(
                    v1: queryVector,
                    v2: embedding.vector,
                    metric: metric
                )
                scoredResults.append((record: embedding, score: score))
            }

            // Sort and take top-k for this query
            let sortedResults = filterAndSort(results: scoredResults, topK: topK, metric: metric)

            // Convert to SearchResult
            let searchResults = sortedResults.map { result in
                SearchResult(
                    id: result.record.id,
                    score: result.score,
                    vector: result.record.vector,
                    sourceType: result.record.sourceType,
                    metadata: result.record.metadata
                )
            }

            batchResults.append(searchResults)
            logger.debug("Query \(queryIndex + 1)/\(queryVectors.count): found \(searchResults.count) results")
        }

        logger.info("Batch search completed: processed \(queryVectors.count) queries")
        return batchResults
    }

    // MARK: - Public Utility Methods

    /// Compute similarity score between two vectors
    ///
    /// Utility method to compute similarity between arbitrary vectors without
    /// performing a full database search. Useful for comparing vectors in memory
    /// or for custom ranking logic.
    ///
    /// ## Parameters
    /// - vector1: First vector
    /// - vector2: Second vector
    /// - metric: Similarity metric to use
    ///
    /// ## Returns
    /// Similarity score according to the specified metric
    ///
    /// ## Throws
    /// - `EmbeddingError.dimensionMismatch`: If vectors have different dimensions
    /// - `EmbeddingError.invalidVector`: If vectors are empty or contain invalid values
    ///
    /// ## Example
    /// ```swift
    /// let score = try await engine.getSimilarityScore(
    ///     vector1: embedding1,
    ///     vector2: embedding2,
    ///     metric: .cosine
    /// )
    /// print("Cosine similarity: \(score)")
    /// ```
    public func getSimilarityScore(
        vector1: [Float],
        vector2: [Float],
        metric: SimilarityMetric
    ) async throws -> Float {
        return try computeSimilarity(v1: vector1, v2: vector2, metric: metric)
    }

    // MARK: - Private Similarity Computation

    /// Compute similarity score between two vectors based on metric
    ///
    /// ## Parameters
    /// - v1: First vector
    /// - v2: Second vector
    /// - metric: Similarity metric to use
    ///
    /// ## Returns
    /// Similarity score
    ///
    /// ## Throws
    /// - `EmbeddingError.dimensionMismatch`: If dimensions don't match
    /// - `EmbeddingError.invalidVector`: If vectors are invalid
    private func computeSimilarity(
        v1: [Float],
        v2: [Float],
        metric: SimilarityMetric
    ) throws -> Float {
        guard v1.count == v2.count else {
            throw EmbeddingError.dimensionMismatch(expected: v1.count, actual: v2.count)
        }

        guard !v1.isEmpty && !v2.isEmpty else {
            throw EmbeddingError.invalidVector("Vectors cannot be empty")
        }

        switch metric {
        case .cosine:
            return cosineSimilarity(v1: v1, v2: v2)
        case .innerProduct:
            return innerProduct(v1: v1, v2: v2)
        case .euclidean:
            return euclideanDistance(v1: v1, v2: v2)
        case .manhattan:
            return manhattanDistance(v1: v1, v2: v2)
        }
    }

    // MARK: - Similarity Metric Implementations

    /// Compute cosine similarity between two vectors
    ///
    /// Cosine similarity measures the cosine of the angle between two vectors,
    /// providing a metric that is independent of vector magnitude. Range: [-1, 1],
    /// but typically [0, 1] for embeddings.
    ///
    /// ## Formula
    /// ```
    /// cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
    /// ```
    ///
    /// ## Properties
    /// - Range: [-1, 1] (theoretically), [0, 1] (for embeddings)
    /// - Higher is better (more similar)
    /// - Invariant to vector magnitude
    /// - Best for normalized embeddings
    ///
    /// ## Time Complexity
    /// - O(d): where d = dimension
    ///
    /// ## Future Optimization
    /// - Use vDSP_dotpr and vDSP_svesq from Accelerate for SIMD acceleration
    ///
    /// ## Parameters
    /// - v1: First vector
    /// - v2: Second vector
    ///
    /// ## Returns
    /// Cosine similarity in range [-1, 1]
    private func cosineSimilarity(v1: [Float], v2: [Float]) -> Float {
        // Future optimization: Use Accelerate framework
        // #if canImport(Accelerate)
        // var dotProduct: Float = 0.0
        // var norm1: Float = 0.0
        // var norm2: Float = 0.0
        // vDSP_dotpr(v1, 1, v2, 1, &dotProduct, vDSP_Length(v1.count))
        // vDSP_svesq(v1, 1, &norm1, vDSP_Length(v1.count))
        // vDSP_svesq(v2, 1, &norm2, vDSP_Length(v2.count))
        // return dotProduct / (sqrt(norm1) * sqrt(norm2))
        // #endif

        var dotProduct: Float = 0.0
        var norm1: Float = 0.0
        var norm2: Float = 0.0

        for i in 0..<v1.count {
            dotProduct += v1[i] * v2[i]
            norm1 += v1[i] * v1[i]
            norm2 += v2[i] * v2[i]
        }

        let magnitude = sqrt(norm1) * sqrt(norm2)
        guard magnitude > Float.ulpOfOne else {
            return 0.0 // Avoid division by zero
        }

        return dotProduct / magnitude
    }

    /// Compute inner product (dot product) between two vectors
    ///
    /// Inner product is the sum of element-wise products. For normalized vectors,
    /// it's equivalent to cosine similarity. Faster than cosine as it skips
    /// normalization.
    ///
    /// ## Formula
    /// ```
    /// inner_product(A, B) = Σ(A[i] × B[i])
    /// ```
    ///
    /// ## Properties
    /// - Range: (-∞, +∞)
    /// - Higher is better (more similar)
    /// - Fast (no square root operations)
    /// - Requires normalized vectors for meaningful comparison
    ///
    /// ## Time Complexity
    /// - O(d): where d = dimension
    ///
    /// ## Future Optimization
    /// - Use vDSP_dotpr from Accelerate for SIMD acceleration
    ///
    /// ## Parameters
    /// - v1: First vector
    /// - v2: Second vector
    ///
    /// ## Returns
    /// Inner product (dot product)
    private func innerProduct(v1: [Float], v2: [Float]) -> Float {
        // Future optimization: Use Accelerate framework
        // #if canImport(Accelerate)
        // var result: Float = 0.0
        // vDSP_dotpr(v1, 1, v2, 1, &result, vDSP_Length(v1.count))
        // return result
        // #endif

        var sum: Float = 0.0
        for i in 0..<v1.count {
            sum += v1[i] * v2[i]
        }
        return sum
    }

    /// Compute Euclidean distance (L2 distance) between two vectors
    ///
    /// Euclidean distance is the straight-line distance between two points in
    /// n-dimensional space. This is the most common distance metric.
    ///
    /// ## Formula
    /// ```
    /// euclidean_distance(A, B) = √(Σ(A[i] - B[i])²)
    /// ```
    ///
    /// ## Properties
    /// - Range: [0, +∞)
    /// - Lower is better (more similar)
    /// - Measures true geometric distance
    /// - Sensitive to vector magnitude
    ///
    /// ## Time Complexity
    /// - O(d): where d = dimension
    ///
    /// ## Future Optimization
    /// - Use vDSP_distancesq from Accelerate to avoid sqrt for ranking
    /// - Only compute sqrt for final results if needed
    ///
    /// ## Parameters
    /// - v1: First vector
    /// - v2: Second vector
    ///
    /// ## Returns
    /// Euclidean distance (non-negative)
    private func euclideanDistance(v1: [Float], v2: [Float]) -> Float {
        // Future optimization: Use Accelerate framework
        // For ranking, can use squared distance to avoid sqrt
        // #if canImport(Accelerate)
        // var squaredSum: Float = 0.0
        // var diff = [Float](repeating: 0, count: v1.count)
        // vDSP_vsub(v2, 1, v1, 1, &diff, 1, vDSP_Length(v1.count))
        // vDSP_svesq(diff, 1, &squaredSum, vDSP_Length(diff.count))
        // return sqrt(squaredSum)
        // #endif

        var sumSquaredDiff: Float = 0.0
        for i in 0..<v1.count {
            let diff = v1[i] - v2[i]
            sumSquaredDiff += diff * diff
        }
        return sqrt(sumSquaredDiff)
    }

    /// Compute Manhattan distance (L1 distance) between two vectors
    ///
    /// Manhattan distance is the sum of absolute differences between vector
    /// components. Also known as taxicab or city block distance.
    ///
    /// ## Formula
    /// ```
    /// manhattan_distance(A, B) = Σ|A[i] - B[i]|
    /// ```
    ///
    /// ## Properties
    /// - Range: [0, +∞)
    /// - Lower is better (more similar)
    /// - More robust to outliers than Euclidean
    /// - Faster to compute (no square/sqrt operations)
    ///
    /// ## Time Complexity
    /// - O(d): where d = dimension
    ///
    /// ## Future Optimization
    /// - Use vDSP_vsub and vDSP_vabs from Accelerate
    ///
    /// ## Parameters
    /// - v1: First vector
    /// - v2: Second vector
    ///
    /// ## Returns
    /// Manhattan distance (non-negative)
    private func manhattanDistance(v1: [Float], v2: [Float]) -> Float {
        // Future optimization: Use Accelerate framework
        // #if canImport(Accelerate)
        // var diff = [Float](repeating: 0, count: v1.count)
        // vDSP_vsub(v2, 1, v1, 1, &diff, 1, vDSP_Length(v1.count))
        // var result: Float = 0.0
        // vDSP_vabs(diff, 1, &diff, 1, vDSP_Length(diff.count))
        // vDSP_sve(diff, 1, &result, vDSP_Length(diff.count))
        // return result
        // #endif

        var sum: Float = 0.0
        for i in 0..<v1.count {
            sum += abs(v1[i] - v2[i])
        }
        return sum
    }

    // MARK: - Helper Methods

    /// Filter and sort results to get top-k
    ///
    /// Sorts results by score according to metric semantics and returns top-k.
    /// Uses different sort orders depending on metric type.
    ///
    /// ## Sorting Strategy
    /// - Similarity metrics (cosine, inner product): descending (higher is better)
    /// - Distance metrics (Euclidean, Manhattan): ascending (lower is better)
    ///
    /// ## Time Complexity
    /// - O(n log n): Full sort (could be optimized to O(n log k) with partial sort)
    ///
    /// ## Future Optimization
    /// - Use partial sort (top-k heap) for better performance: O(n log k)
    /// - For very large datasets, consider approximate methods
    ///
    /// ## Parameters
    /// - results: Array of (record, score) tuples
    /// - topK: Number of top results to return
    /// - metric: Similarity metric used (determines sort order)
    ///
    /// ## Returns
    /// Top-k results sorted by score
    private func filterAndSort(
        results: [(record: EmbeddingRecord, score: Float)],
        topK: Int,
        metric: SimilarityMetric
    ) -> [(record: EmbeddingRecord, score: Float)] {
        // Determine sort order based on metric
        let sorted: [(record: EmbeddingRecord, score: Float)]

        switch metric {
        case .cosine, .innerProduct:
            // Higher is better (similarity metrics)
            sorted = results.sorted { $0.score > $1.score }
        case .euclidean, .manhattan:
            // Lower is better (distance metrics)
            sorted = results.sorted { $0.score < $1.score }
        }

        // Take top-k results
        let count = min(topK, sorted.count)
        return Array(sorted.prefix(count))
    }
}

// MARK: - EmbeddingStore Protocol

/// Protocol defining the interface for embedding storage access
///
/// This protocol abstracts the storage layer, allowing different implementations
/// (in-memory, FoundationDB, etc.) and facilitating testing with mock stores.
public protocol EmbeddingStoreProtocol: Actor {

    /// Retrieve model metadata by name
    ///
    /// ## Parameters
    /// - model: Model name
    ///
    /// ## Returns
    /// Model metadata if found, nil otherwise
    ///
    /// ## Throws
    /// Storage-related errors
    func getModelMetadata(model: String) async throws -> EmbeddingModelMetadata?

    /// Retrieve all embeddings for a specific model
    ///
    /// ## Parameters
    /// - model: Model name
    ///
    /// ## Returns
    /// Array of all embedding records for the model
    ///
    /// ## Throws
    /// Storage-related errors
    func getAllEmbeddings(model: String) async throws -> [EmbeddingRecord]
}
