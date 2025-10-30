import Foundation
@preconcurrency import FoundationDB
import Logging
import Synchronization

/// Class responsible for managing embedding model metadata in FoundationDB.
///
/// The ModelManager provides thread-safe operations for:
/// - Registering new embedding models
/// - Retrieving model metadata with caching
/// - Updating and deleting models
/// - Validating model configurations
/// - Tracking model statistics
///
/// ## Key Structure
///
/// Models are stored using Tuple-encoded keys:
/// ```
/// (rootPrefix, "model", modelName) -> EmbeddingModelMetadata (JSON)
/// ```
///
/// ## Caching Strategy
///
/// ModelManager maintains an LRU cache (limit: 100) of model metadata to minimize
/// database reads for frequently accessed models. The cache is automatically
/// invalidated on updates and deletions.
///
/// ## Usage Example
///
/// ```swift
/// let modelManager = ModelManager(
///     database: database,
///     rootPrefix: "embedding:",
///     logger: logger
/// )
///
/// // Register a new model
/// let model = EmbeddingModelMetadata(
///     name: "text-embedding-3-small",
///     dimension: 1536,
///     provider: "openai",
///     description: "OpenAI's text embedding model",
///     version: "3.0"
/// )
/// try await modelManager.registerModel(model)
///
/// // Retrieve model with caching
/// if let cached = try await modelManager.getModel(name: "text-embedding-3-small") {
///     print("Model dimension: \(cached.dimension)")
/// }
///
/// // Get statistics
/// let stats = try await modelManager.getModelStats(name: "text-embedding-3-small")
/// print("Embedding count: \(stats.embeddingCount)")
/// ```
///
/// ## Thread Safety
///
/// Thread safety is provided by:
/// - Swift Synchronization.Mutex for cache access (~1μs overhead vs ~10μs for actor)
/// - FoundationDB transaction model for database operations
public final class ModelManager: @unchecked Sendable {

    // MARK: - Properties

    /// FoundationDB database connection
    ///
    /// Marked as `nonisolated(unsafe)` for compatibility with FoundationDB's threading model.
    /// The actual thread safety is ensured by FoundationDB's internal synchronization.
    nonisolated(unsafe) private let database: any DatabaseProtocol

    /// Root prefix for all model keys in FoundationDB
    ///
    /// Used to namespace model data within the database, allowing multiple
    /// embedding layers to coexist in the same FoundationDB instance.
    private let rootPrefix: String

    /// Logger instance for debugging and monitoring
    private let logger: Logger

    /// Cache state protected by Mutex
    private struct CacheState {
        var modelCache: [String: EmbeddingModelMetadata] = [:]
        var cacheAccessOrder: [String] = []
    }

    /// Maximum number of models to cache
    private let maxCacheSize = 100

    /// Mutex-protected cache state
    private let cache: Mutex<CacheState>

    // MARK: - Initialization

    /// Initialize a new ModelManager instance
    ///
    /// - Parameters:
    ///   - database: FoundationDB database connection
    ///   - rootPrefix: Root prefix for model keys (default: "embedding:")
    ///   - logger: Logger instance for debugging
    ///
    /// - Note: The rootPrefix should be consistent across all components of the
    ///         embedding layer to ensure proper data isolation.
    public init(
        database: any DatabaseProtocol,
        rootPrefix: String = "embedding:",
        logger: Logger = Logger(label: "com.embedding.modelmanager")
    ) {
        self.database = database
        self.rootPrefix = rootPrefix
        self.logger = logger
        self.cache = Mutex(CacheState())

        logger.debug("ModelManager initialized", metadata: [
            "rootPrefix": "\(rootPrefix)"
        ])
    }

    // MARK: - Model Registration

    /// Register a new embedding model in the database
    ///
    /// This operation validates the model metadata and checks for duplicate
    /// model names before registration. The model metadata is stored as JSON
    /// in FoundationDB.
    ///
    /// - Parameter model: Model metadata to register
    ///
    /// - Throws:
    ///   - `EmbeddingError.invalidModel`: If validation fails
    ///   - `EmbeddingError.storageError`: If a model with the same name already exists
    ///   - `EmbeddingError.encodingError`: If JSON encoding fails
    ///
    /// - Note: This operation automatically updates the model cache on success
    ///
    /// ## Example
    ///
    /// ```swift
    /// let model = EmbeddingModelMetadata(
    ///     name: "text-embedding-3-small",
    ///     dimension: 1536,
    ///     provider: "openai"
    /// )
    /// try await modelManager.registerModel(model)
    /// ```
    public func registerModel(_ model: EmbeddingModelMetadata) async throws {
        logger.info("Registering model", metadata: [
            "model": "\(model.name)",
            "dimension": "\(model.dimension)",
            "provider": "\(model.provider)"
        ])

        // Validate model metadata
        try validateModel(model)

        // Check for duplicate
        let exists = try await modelExists(name: model.name)
        if exists {
            logger.error("Model already exists", metadata: ["model": "\(model.name)"])
            throw EmbeddingError.storageError("Model '\(model.name)' already exists")
        }

        // Encode model as JSON
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601

        guard let modelData = try? encoder.encode(model) else {
            throw EmbeddingError.encodingError("Failed to encode model metadata")
        }

        // Store in database
        let key = TupleHelpers.encodeModelKey(rootPrefix: rootPrefix, modelName: model.name)

        try await database.withTransaction { transaction in
            transaction.setValue(Array(modelData), for: key)
        }

        // Update cache
        updateModelCache(model)

        logger.info("Model registered successfully", metadata: ["model": "\(model.name)"])
    }

    // MARK: - Model Retrieval

    /// Get model metadata by name with caching
    ///
    /// This operation first checks the in-memory cache before querying the database.
    /// Cache hits significantly reduce database load for frequently accessed models.
    ///
    /// - Parameter name: Model name to retrieve
    ///
    /// - Returns: Model metadata if found, nil otherwise
    ///
    /// - Throws:
    ///   - `EmbeddingError.encodingError`: If JSON decoding fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// if let model = try await modelManager.getModel(name: "text-embedding-3-small") {
    ///     print("Model dimension: \(model.dimension)")
    /// } else {
    ///     print("Model not found")
    /// }
    /// ```
    public func getModel(name: String) async throws -> EmbeddingModelMetadata? {
        // Check cache first (with Mutex)
        let cached = cache.withLock { state -> EmbeddingModelMetadata? in
            guard let model = state.modelCache[name] else {
                return nil
            }

            // Update access order
            if let index = state.cacheAccessOrder.firstIndex(of: name) {
                state.cacheAccessOrder.remove(at: index)
            }
            state.cacheAccessOrder.append(name)

            return model
        }

        if let model = cached {
            logger.debug("Model cache hit", metadata: ["model": "\(name)"])
            return model
        }

        logger.debug("Model cache miss, querying database", metadata: ["model": "\(name)"])

        // Query database
        let key = TupleHelpers.encodeModelKey(rootPrefix: rootPrefix, modelName: name)

        let modelData = try await database.withTransaction { transaction in
            return try await transaction.getValue(for: key, snapshot: true)
        }

        guard let data = modelData else {
            logger.debug("Model not found", metadata: ["model": "\(name)"])
            return nil
        }

        // Decode JSON
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        guard let model = try? decoder.decode(EmbeddingModelMetadata.self, from: Data(data)) else {
            throw EmbeddingError.encodingError("Failed to decode model metadata for '\(name)'")
        }

        // Update cache
        updateModelCache(model)

        return model
    }

    /// Get all registered models
    ///
    /// This operation performs a range query to retrieve all models stored in the
    /// database. For large numbers of models, consider using pagination or filtering.
    ///
    /// - Returns: Array of all registered model metadata
    ///
    /// - Throws:
    ///   - `EmbeddingError.encodingError`: If JSON decoding fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// let allModels = try await modelManager.getAllModels()
    /// for model in allModels {
    ///     print("\(model.name): \(model.dimension) dimensions")
    /// }
    /// ```
    public func getAllModels() async throws -> [EmbeddingModelMetadata] {
        logger.debug("Retrieving all models")

        let prefix = TupleHelpers.encodeModelRangePrefix(rootPrefix: rootPrefix)
        let (beginKey, endKey) = TupleHelpers.encodeRangeKeys(prefix: prefix)

        var models: [EmbeddingModelMetadata] = []
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        try await database.withTransaction { transaction in
            let sequence = transaction.getRange(
                beginSelector: .firstGreaterOrEqual(beginKey),
                endSelector: .firstGreaterThan(endKey),
                snapshot: true
            )

            for try await (_, value) in sequence {
                if let model = try? decoder.decode(EmbeddingModelMetadata.self, from: Data(value)) {
                    models.append(model)
                    // Opportunistically update cache
                    updateModelCache(model)
                } else {
                    logger.warning("Failed to decode model metadata")
                }
            }
        }

        logger.debug("Retrieved models", metadata: ["count": "\(models.count)"])
        return models
    }

    // MARK: - Model Management

    /// Update existing model metadata
    ///
    /// This operation allows updating model metadata while preserving the model name.
    /// The cache is automatically invalidated and updated on success.
    ///
    /// - Parameter model: Updated model metadata
    ///
    /// - Throws:
    ///   - `EmbeddingError.invalidModel`: If validation fails
    ///   - `EmbeddingError.modelNotFound`: If the model doesn't exist
    ///   - `EmbeddingError.encodingError`: If JSON encoding fails
    ///
    /// - Warning: Changing the dimension of a model with existing embeddings
    ///            may lead to inconsistencies. Consider creating a new model instead.
    ///
    /// ## Example
    ///
    /// ```swift
    /// var model = try await modelManager.getModel(name: "text-embedding-3-small")!
    /// model.description = "Updated description"
    /// try await modelManager.updateModel(model)
    /// ```
    public func updateModel(_ model: EmbeddingModelMetadata) async throws {
        logger.info("Updating model", metadata: ["model": "\(model.name)"])

        // Validate model metadata
        try validateModel(model)

        // Check if model exists
        let exists = try await modelExists(name: model.name)
        if !exists {
            logger.error("Model not found for update", metadata: ["model": "\(model.name)"])
            throw EmbeddingError.modelNotFound(model.name)
        }

        // Encode model as JSON
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601

        guard let modelData = try? encoder.encode(model) else {
            throw EmbeddingError.encodingError("Failed to encode model metadata")
        }

        // Update in database
        let key = TupleHelpers.encodeModelKey(rootPrefix: rootPrefix, modelName: model.name)

        try await database.withTransaction { transaction in
            transaction.setValue(Array(modelData), for: key)
        }

        // Update cache
        updateModelCache(model)

        logger.info("Model updated successfully", metadata: ["model": "\(model.name)"])
    }

    /// Check if a model exists in the database
    ///
    /// This operation performs a snapshot read to check for model existence
    /// without creating read conflicts.
    ///
    /// - Parameter name: Model name to check
    ///
    /// - Returns: true if the model exists, false otherwise
    ///
    /// ## Example
    ///
    /// ```swift
    /// if try await modelManager.modelExists(name: "text-embedding-3-small") {
    ///     print("Model exists")
    /// }
    /// ```
    public func modelExists(name: String) async throws -> Bool {
        let key = TupleHelpers.encodeModelKey(rootPrefix: rootPrefix, modelName: name)

        let value = try await database.withTransaction { transaction in
            return try await transaction.getValue(for: key, snapshot: true)
        }

        return value != nil
    }

    /// Delete a model from the database
    ///
    /// This operation removes the model metadata from the database and invalidates
    /// the cache entry. As a safety measure, it checks if any embeddings exist for
    /// the model before deletion.
    ///
    /// - Parameter name: Model name to delete
    ///
    /// - Throws:
    ///   - `EmbeddingError.modelNotFound`: If the model doesn't exist
    ///   - `EmbeddingError.storageError`: If embeddings exist for the model
    ///
    /// - Warning: This operation does not delete associated embeddings.
    ///            Ensure all embeddings are deleted before removing the model.
    ///
    /// ## Example
    ///
    /// ```swift
    /// // Check for embeddings first
    /// let stats = try await modelManager.getModelStats(name: "old-model")
    /// if stats.embeddingCount == 0 {
    ///     try await modelManager.deleteModel(name: "old-model")
    /// }
    /// ```
    public func deleteModel(name: String) async throws {
        logger.info("Deleting model", metadata: ["model": "\(name)"])

        // Check if model exists
        let exists = try await modelExists(name: name)
        if !exists {
            logger.error("Model not found for deletion", metadata: ["model": "\(name)"])
            throw EmbeddingError.modelNotFound(name)
        }

        // Check if embeddings exist
        let stats = try await getModelStats(name: name)
        if stats.embeddingCount > 0 {
            logger.error("Cannot delete model with existing embeddings", metadata: [
                "model": "\(name)",
                "embeddingCount": "\(stats.embeddingCount)"
            ])
            throw EmbeddingError.storageError(
                "Cannot delete model '\(name)' with \(stats.embeddingCount) existing embeddings"
            )
        }

        // Delete from database
        let modelKey = TupleHelpers.encodeModelKey(rootPrefix: rootPrefix, modelName: name)
        let statsKey = TupleHelpers.encodeStatsKey(
            rootPrefix: rootPrefix,
            model: name,
            statType: "count"
        )

        try await database.withTransaction { transaction in
            // Delete model definition
            transaction.clear(key: modelKey)
            // Delete statistics counter
            transaction.clear(key: statsKey)
        }

        // Remove from cache (with Mutex)
        cache.withLock { state in
            state.modelCache.removeValue(forKey: name)
            if let index = state.cacheAccessOrder.firstIndex(of: name) {
                state.cacheAccessOrder.remove(at: index)
            }
        }

        logger.info("Model deleted successfully", metadata: ["model": "\(name)"])
    }

    // MARK: - Statistics

    /// Model statistics structure
    public struct ModelStats {
        /// Number of embeddings stored for this model
        public let embeddingCount: Int
    }

    /// Get statistics for a model
    ///
    /// This operation counts the number of embeddings stored for the given model
    /// by performing a range query over the embedding keyspace.
    ///
    /// - Parameter name: Model name
    ///
    /// - Returns: Statistics for the model
    ///
    /// - Throws:
    ///   - `EmbeddingError.modelNotFound`: If the model doesn't exist
    ///
    /// - Note: This operation is O(1) as it reads the atomic counter maintained by EmbeddingStore.
    ///
    /// ## Example
    ///
    /// ```swift
    /// let stats = try await modelManager.getModelStats(name: "text-embedding-3-small")
    /// print("Total embeddings: \(stats.embeddingCount)")
    /// ```
    public func getModelStats(name: String) async throws -> ModelStats {
        logger.debug("Getting model statistics", metadata: ["model": "\(name)"])

        // Verify model exists
        let exists = try await modelExists(name: name)
        if !exists {
            throw EmbeddingError.modelNotFound(name)
        }

        // Read the atomic counter (O(1) operation)
        let countKey = TupleHelpers.encodeStatsKey(
            rootPrefix: rootPrefix,
            model: name,
            statType: "count"
        )

        let count = try await database.withTransaction { transaction in
            guard let bytes = try await transaction.getValue(for: countKey, snapshot: true) else {
                return 0
            }

            // Decode as Int64 (since we use Int64 for atomic operations)
            let signedCount = TupleHelpers.decodeInt64(bytes)
            // Count should never be negative, but handle gracefully
            return Int(max(0, signedCount))
        }

        logger.debug("Model statistics retrieved", metadata: [
            "model": "\(name)",
            "embeddingCount": "\(count)"
        ])

        return ModelStats(embeddingCount: count)
    }

    // MARK: - Validation

    /// Validate model metadata
    ///
    /// Ensures that model metadata meets the following requirements:
    /// - Model name is not empty
    /// - Model name contains only valid characters (alphanumeric, dash, underscore)
    /// - Dimension is greater than 0
    /// - Provider is not empty
    ///
    /// - Parameter model: Model metadata to validate
    ///
    /// - Throws: `EmbeddingError.invalidModel` if validation fails
    private func validateModel(_ model: EmbeddingModelMetadata) throws {
        // Validate name
        guard !model.name.isEmpty else {
            throw EmbeddingError.invalidModel("Model name cannot be empty")
        }

        // Check for valid characters in name (alphanumeric, dash, underscore)
        let validNamePattern = "^[a-zA-Z0-9_-]+$"
        let namePredicate = NSPredicate(format: "SELF MATCHES %@", validNamePattern)
        guard namePredicate.evaluate(with: model.name) else {
            throw EmbeddingError.invalidModel(
                "Model name '\(model.name)' contains invalid characters. " +
                "Only alphanumeric characters, dashes, and underscores are allowed."
            )
        }

        // Validate dimension
        guard model.dimension > 0 else {
            throw EmbeddingError.invalidModel(
                "Model dimension must be greater than 0, got \(model.dimension)"
            )
        }

        // Validate provider
        guard !model.provider.isEmpty else {
            throw EmbeddingError.invalidModel("Model provider cannot be empty")
        }

        logger.debug("Model validation passed", metadata: ["model": "\(model.name)"])
    }

    // MARK: - Cache Management

    /// Update the model cache with LRU eviction
    ///
    /// This private method manages the in-memory cache of model metadata.
    /// When the cache exceeds maxCacheSize, the least recently used entry is evicted.
    ///
    /// - Parameter model: Model metadata to cache
    private func updateModelCache(_ model: EmbeddingModelMetadata) {
        cache.withLock { state in
            state.modelCache[model.name] = model

            // Update access order
            if let index = state.cacheAccessOrder.firstIndex(of: model.name) {
                state.cacheAccessOrder.remove(at: index)
            }
            state.cacheAccessOrder.append(model.name)

            // Evict LRU if cache is full
            if state.modelCache.count > maxCacheSize {
                if let lru = state.cacheAccessOrder.first {
                    state.modelCache.removeValue(forKey: lru)
                    state.cacheAccessOrder.removeFirst()
                    logger.debug("Evicted LRU model from cache", metadata: ["model": "\(lru)"])
                }
            }
        }
    }

    /// Clear the entire model cache
    ///
    /// This operation removes all cached model metadata, forcing subsequent
    /// getModel calls to query the database. Useful for testing or when
    /// external processes may have modified model data.
    ///
    /// ## Example
    ///
    /// ```swift
    /// modelManager.clearCache()
    /// // Next getModel call will query the database
    /// ```
    public func clearCache() {
        cache.withLock { state in
            state.modelCache.removeAll()
            state.cacheAccessOrder.removeAll()
        }
        logger.debug("Model cache cleared")
    }
}
