import Foundation
@preconcurrency import FoundationDB
import Logging

/// Actor-based storage layer for embedding vectors in FoundationDB
///
/// The `EmbeddingStore` manages the persistence and retrieval of embedding vectors,
/// providing CRUD operations with built-in caching and batch support.
///
/// Key Features:
/// - Thread-safe operations via actor isolation
/// - LRU cache for frequently accessed embeddings (limit: 1000 items)
/// - Batch operations for efficient bulk writes/reads
/// - Atomic counters for statistics
/// - Multiple query patterns (by model, source type, etc.)
///
/// Storage Schema:
/// - Embedding data: (rootPrefix, "embedding", model, id) -> encoded_record
/// - Source type index: (rootPrefix, "index", "source", sourceType, model, id) -> empty
/// - Statistics: (rootPrefix, "stats", model, "count") -> UInt64
///
/// Example Usage:
/// ```swift
/// let store = EmbeddingStore(
///     database: database,
///     rootPrefix: "embeddings",
///     logger: logger
/// )
///
/// // Save single embedding
/// let record = EmbeddingRecord(...)
/// try await store.save(record: record)
///
/// // Get with cache
/// let retrieved = try await store.get(id: "entity:123", model: "text-embedding-3-small")
///
/// // Batch operations
/// try await store.saveBatch(records)
/// let results = try await store.getBatch(ids: ids, model: model)
/// ```
public actor EmbeddingStore {

    // MARK: - Properties

    /// FoundationDB database instance (non-isolated for concurrent access)
    nonisolated(unsafe) private let database: any DatabaseProtocol

    /// Root prefix for all keys in this store
    private let rootPrefix: String

    /// Logger for debugging and monitoring
    private let logger: Logger

    // MARK: - Cache

    /// LRU cache for embedding records
    /// Key: "\(model):\(id)"
    /// Value: EmbeddingRecord
    private var vectorCache: [String: CacheEntry] = [:]

    /// Ordered list of cache keys for LRU eviction
    private var cacheAccessOrder: [String] = []

    /// Maximum number of embeddings to keep in cache
    private let cacheLimit: Int = 1000

    /// Cache entry with access tracking
    private struct CacheEntry {
        let record: EmbeddingRecord
        var lastAccessed: Date

        init(record: EmbeddingRecord) {
            self.record = record
            self.lastAccessed = Date()
        }

        mutating func updateAccess() {
            self.lastAccessed = Date()
        }
    }

    // MARK: - Initialization

    /// Initialize the EmbeddingStore
    ///
    /// - Parameters:
    ///   - database: FoundationDB database instance
    ///   - rootPrefix: Root prefix for all keys (e.g., "embeddings")
    ///   - logger: Logger instance for debugging
    public init(
        database: any DatabaseProtocol,
        rootPrefix: String,
        logger: Logger
    ) {
        self.database = database
        self.rootPrefix = rootPrefix
        self.logger = logger
    }

    // MARK: - Single Record Operations

    /// Save a single embedding record
    ///
    /// This operation:
    /// 1. Validates the vector
    /// 2. Encodes the record
    /// 3. Writes to main embedding key
    /// 4. Updates source type index
    /// 5. Increments model counter (atomic)
    /// 6. Updates cache
    ///
    /// - Parameter record: The embedding record to save
    /// - Throws: `EmbeddingError` if validation or encoding fails
    public func save(record: EmbeddingRecord) async throws {
        logger.debug("Saving embedding",
                    metadata: ["id": "\(record.id)",
                              "model": "\(record.model)",
                              "dimension": "\(record.dimension)"])

        // Validate vector
        try VectorCodec.validate(record.vector)

        // Encode record
        let encodedValue = try encodeEmbedding(record: record)

        try await database.withTransaction { transaction in
            // Main embedding key: (rootPrefix, "embedding", model, id)
            let embeddingKey = TupleHelpers.encodeEmbeddingKey(
                rootPrefix: self.rootPrefix,
                model: record.model,
                id: record.id
            )

            // Check if this is a new record (for counter increment)
            let existingValue = try await transaction.getValue(for: embeddingKey, snapshot: true)
            let isNewRecord = existingValue == nil

            // Write embedding data
            transaction.setValue(encodedValue, for: embeddingKey)

            // Update source type index
            let sourceIndexKey = TupleHelpers.encodeSourceTypeIndexKey(
                rootPrefix: self.rootPrefix,
                sourceType: record.sourceType.rawValue,
                model: record.model,
                id: record.id
            )
            transaction.setValue([], for: sourceIndexKey)

            // Increment counter only for new records
            if isNewRecord {
                let countKey = TupleHelpers.encodeStatsKey(
                    rootPrefix: self.rootPrefix,
                    model: record.model,
                    statType: "count"
                )
                let increment = TupleHelpers.encodeInt64(Int64(1))
                transaction.atomicOp(key: countKey, param: increment, mutationType: .add)
            }
        }

        // Update cache
        updateCache(record: record)

        logger.debug("Successfully saved embedding",
                    metadata: ["id": "\(record.id)",
                              "model": "\(record.model)"])
    }

    /// Get a single embedding record by ID and model
    ///
    /// This operation checks the cache first before querying the database.
    ///
    /// - Parameters:
    ///   - id: The embedding ID
    ///   - model: The model name
    /// - Returns: The embedding record if found, nil otherwise
    /// - Throws: `EmbeddingError` if decoding fails
    public func get(id: String, model: String) async throws -> EmbeddingRecord? {
        logger.debug("Getting embedding",
                    metadata: ["id": "\(id)", "model": "\(model)"])

        // Check cache first
        let cacheKey = "\(model):\(id)"
        if var entry = vectorCache[cacheKey] {
            entry.updateAccess()
            vectorCache[cacheKey] = entry

            // Update access order
            if let index = cacheAccessOrder.firstIndex(of: cacheKey) {
                cacheAccessOrder.remove(at: index)
            }
            cacheAccessOrder.append(cacheKey)

            logger.debug("Cache hit for embedding",
                        metadata: ["id": "\(id)", "model": "\(model)"])
            return entry.record
        }

        // Cache miss - query database
        let record: EmbeddingRecord? = try await database.withTransaction { transaction in
            let embeddingKey = TupleHelpers.encodeEmbeddingKey(
                rootPrefix: self.rootPrefix,
                model: model,
                id: id
            )

            guard let bytes = try await transaction.getValue(for: embeddingKey, snapshot: true) else {
                return nil
            }

            return try self.decodeEmbedding(bytes: bytes)
        }

        // Update cache if found
        if let record = record {
            updateCache(record: record)
            logger.debug("Database hit for embedding",
                        metadata: ["id": "\(id)", "model": "\(model)"])
        } else {
            logger.debug("Embedding not found",
                        metadata: ["id": "\(id)", "model": "\(model)"])
        }

        return record
    }

    /// Delete a single embedding record
    ///
    /// This operation:
    /// 1. Removes main embedding key
    /// 2. Removes source type index entry
    /// 3. Decrements model counter (atomic)
    /// 4. Removes from cache
    ///
    /// - Parameters:
    ///   - id: The embedding ID
    ///   - model: The model name
    /// - Throws: `EmbeddingError` if the operation fails
    public func delete(id: String, model: String) async throws {
        logger.debug("Deleting embedding",
                    metadata: ["id": "\(id)", "model": "\(model)"])

        // Get existing record to find sourceType
        guard let record = try await get(id: id, model: model) else {
            logger.warning("Attempted to delete non-existent embedding",
                          metadata: ["id": "\(id)", "model": "\(model)"])
            return
        }

        try await database.withTransaction { transaction in
            // Delete main embedding key
            let embeddingKey = TupleHelpers.encodeEmbeddingKey(
                rootPrefix: self.rootPrefix,
                model: model,
                id: id
            )
            transaction.clear(key: embeddingKey)

            // Delete source type index
            let sourceIndexKey = TupleHelpers.encodeSourceTypeIndexKey(
                rootPrefix: self.rootPrefix,
                sourceType: record.sourceType.rawValue,
                model: model,
                id: id
            )
            transaction.clear(key: sourceIndexKey)

            // Decrement counter
            let countKey = TupleHelpers.encodeStatsKey(
                rootPrefix: self.rootPrefix,
                model: model,
                statType: "count"
            )

            // Atomic subtract (add negative value using Int64)
            let decrement = TupleHelpers.encodeInt64(Int64(-1))
            transaction.atomicOp(key: countKey, param: decrement, mutationType: .add)
        }

        // Remove from cache
        let cacheKey = "\(model):\(id)"
        vectorCache.removeValue(forKey: cacheKey)
        if let index = cacheAccessOrder.firstIndex(of: cacheKey) {
            cacheAccessOrder.remove(at: index)
        }

        logger.debug("Successfully deleted embedding",
                    metadata: ["id": "\(id)", "model": "\(model)"])
    }

    /// Check if an embedding exists
    ///
    /// - Parameters:
    ///   - id: The embedding ID
    ///   - model: The model name
    /// - Returns: True if the embedding exists, false otherwise
    public func exists(id: String, model: String) async throws -> Bool {
        logger.debug("Checking embedding existence",
                    metadata: ["id": "\(id)", "model": "\(model)"])

        // Check cache first
        let cacheKey = "\(model):\(id)"
        if vectorCache[cacheKey] != nil {
            return true
        }

        // Check database
        return try await database.withTransaction { transaction in
            let embeddingKey = TupleHelpers.encodeEmbeddingKey(
                rootPrefix: self.rootPrefix,
                model: model,
                id: id
            )

            let value = try await transaction.getValue(for: embeddingKey, snapshot: true)
            return value != nil
        }
    }

    // MARK: - Batch Operations

    /// Save multiple embedding records in a single transaction
    ///
    /// This is more efficient than calling `save` multiple times as it batches
    /// all operations into a single FoundationDB transaction.
    ///
    /// Note: Large batches may exceed FoundationDB transaction limits (10MB).
    /// Consider chunking large batches into smaller groups.
    ///
    /// - Parameter records: Array of embedding records to save
    /// - Throws: `EmbeddingError` if validation or encoding fails
    public func saveBatch(_ records: [EmbeddingRecord]) async throws {
        guard !records.isEmpty else { return }

        logger.info("Saving batch of embeddings",
                   metadata: ["count": "\(records.count)"])

        // Validate all vectors first
        for record in records {
            try VectorCodec.validate(record.vector)
        }

        try await database.withTransaction { transaction in
            // Group records by model for counter updates
            var newRecordsByModel: [String: Int] = [:]

            for record in records {
                // Encode record
                let encodedValue = try self.encodeEmbedding(record: record)

                // Main embedding key
                let embeddingKey = TupleHelpers.encodeEmbeddingKey(
                    rootPrefix: self.rootPrefix,
                    model: record.model,
                    id: record.id
                )

                // Check if new record
                let existingValue = try await transaction.getValue(for: embeddingKey, snapshot: true)
                if existingValue == nil {
                    newRecordsByModel[record.model, default: 0] += 1
                }

                // Write embedding data
                transaction.setValue(encodedValue, for: embeddingKey)

                // Update source type index
                let sourceIndexKey = TupleHelpers.encodeSourceTypeIndexKey(
                    rootPrefix: self.rootPrefix,
                    sourceType: record.sourceType.rawValue,
                    model: record.model,
                    id: record.id
                )
                transaction.setValue([], for: sourceIndexKey)
            }

            // Update counters for each model
            for (model, count) in newRecordsByModel {
                let countKey = TupleHelpers.encodeStatsKey(
                    rootPrefix: self.rootPrefix,
                    model: model,
                    statType: "count"
                )
                let increment = TupleHelpers.encodeInt64(Int64(count))
                transaction.atomicOp(key: countKey, param: increment, mutationType: .add)
            }
        }

        // Update cache for all records
        for record in records {
            updateCache(record: record)
        }

        logger.info("Successfully saved batch of embeddings",
                   metadata: ["count": "\(records.count)"])
    }

    /// Get multiple embedding records in a batch
    ///
    /// - Parameters:
    ///   - ids: Array of embedding IDs
    ///   - model: The model name (all embeddings must use the same model)
    /// - Returns: Array of embedding records (may be smaller than input if some not found)
    /// - Throws: `EmbeddingError` if decoding fails
    public func getBatch(ids: [String], model: String) async throws -> [EmbeddingRecord] {
        guard !ids.isEmpty else { return [] }

        logger.debug("Getting batch of embeddings",
                    metadata: ["count": "\(ids.count)", "model": "\(model)"])

        var results: [EmbeddingRecord] = []
        var uncachedIds: [String] = []

        // Check cache first
        for id in ids {
            let cacheKey = "\(model):\(id)"
            if var entry = vectorCache[cacheKey] {
                entry.updateAccess()
                vectorCache[cacheKey] = entry

                // Update access order
                if let index = cacheAccessOrder.firstIndex(of: cacheKey) {
                    cacheAccessOrder.remove(at: index)
                }
                cacheAccessOrder.append(cacheKey)

                results.append(entry.record)
            } else {
                uncachedIds.append(id)
            }
        }

        // Query database for uncached records
        if !uncachedIds.isEmpty {
            let dbRecords = try await database.withTransaction { transaction in
                var records: [EmbeddingRecord] = []

                for id in uncachedIds {
                    let embeddingKey = TupleHelpers.encodeEmbeddingKey(
                        rootPrefix: self.rootPrefix,
                        model: model,
                        id: id
                    )

                    if let bytes = try await transaction.getValue(for: embeddingKey, snapshot: true) {
                        let record = try self.decodeEmbedding(bytes: bytes)
                        records.append(record)
                    }
                }

                return records
            }

            // Update cache
            for record in dbRecords {
                updateCache(record: record)
            }

            results.append(contentsOf: dbRecords)
        }

        logger.debug("Successfully retrieved batch",
                    metadata: ["requested": "\(ids.count)",
                              "found": "\(results.count)",
                              "cache_hits": "\(ids.count - uncachedIds.count)"])

        return results
    }

    /// Delete multiple embedding records in a batch
    ///
    /// - Parameters:
    ///   - ids: Array of embedding IDs
    ///   - model: The model name
    /// - Throws: `EmbeddingError` if the operation fails
    public func deleteBatch(ids: [String], model: String) async throws {
        guard !ids.isEmpty else { return }

        logger.info("Deleting batch of embeddings",
                   metadata: ["count": "\(ids.count)", "model": "\(model)"])

        // Get existing records to find sourceTypes
        let existingRecords = try await getBatch(ids: ids, model: model)
        let recordMap = Dictionary(uniqueKeysWithValues: existingRecords.map { ($0.id, $0) })

        try await database.withTransaction { transaction in
            var deleteCount = 0

            for id in ids {
                guard let record = recordMap[id] else {
                    continue
                }

                // Delete main embedding key
                let embeddingKey = TupleHelpers.encodeEmbeddingKey(
                    rootPrefix: self.rootPrefix,
                    model: model,
                    id: id
                )
                transaction.clear(key: embeddingKey)

                // Delete source type index
                let sourceIndexKey = TupleHelpers.encodeSourceTypeIndexKey(
                    rootPrefix: self.rootPrefix,
                    sourceType: record.sourceType.rawValue,
                    model: model,
                    id: id
                )
                transaction.clear(key: sourceIndexKey)

                deleteCount += 1
            }

            // Decrement counter
            if deleteCount > 0 {
                let countKey = TupleHelpers.encodeStatsKey(
                    rootPrefix: self.rootPrefix,
                    model: model,
                    statType: "count"
                )

                // Atomic subtract (add negative value using Int64)
                let decrement = TupleHelpers.encodeInt64(Int64(-deleteCount))
                transaction.atomicOp(key: countKey, param: decrement, mutationType: .add)
            }
        }

        // Remove from cache
        for id in ids {
            let cacheKey = "\(model):\(id)"
            vectorCache.removeValue(forKey: cacheKey)
            if let index = cacheAccessOrder.firstIndex(of: cacheKey) {
                cacheAccessOrder.remove(at: index)
            }
        }

        logger.info("Successfully deleted batch",
                   metadata: ["count": "\(existingRecords.count)", "model": "\(model)"])
    }

    // MARK: - Query Operations

    /// Get all embeddings for a specific model
    ///
    /// This performs a range query over the model's key space.
    /// For large datasets, consider implementing pagination.
    ///
    /// - Parameter model: The model name
    /// - Returns: Array of all embedding records for the model
    /// - Throws: `EmbeddingError` if decoding fails
    public func getAllByModel(model: String) async throws -> [EmbeddingRecord] {
        logger.debug("Getting all embeddings for model",
                    metadata: ["model": "\(model)"])

        let records = try await database.withTransaction { transaction in
            var results: [EmbeddingRecord] = []

            let prefix = TupleHelpers.encodeEmbeddingRangePrefix(
                rootPrefix: self.rootPrefix,
                model: model
            )
            let (beginKey, endKey) = TupleHelpers.encodeRangeKeys(prefix: prefix)

            let sequence = transaction.getRange(
                beginSelector: .firstGreaterOrEqual(beginKey),
                endSelector: .firstGreaterOrEqual(endKey),
                snapshot: true
            )

            for try await (_, value) in sequence {
                let record = try self.decodeEmbedding(bytes: value)
                results.append(record)
            }

            return results
        }

        logger.debug("Retrieved embeddings for model",
                    metadata: ["model": "\(model)", "count": "\(records.count)"])

        return records
    }

    /// Get all embeddings for a specific source type and model
    ///
    /// This uses the source type index for efficient querying.
    ///
    /// - Parameters:
    ///   - sourceType: The source type to filter by
    ///   - model: The model name
    /// - Returns: Array of embedding records matching the criteria
    /// - Throws: `EmbeddingError` if decoding fails
    public func getAllBySourceType(
        sourceType: SourceType,
        model: String
    ) async throws -> [EmbeddingRecord] {
        logger.debug("Getting embeddings by source type",
                    metadata: ["sourceType": "\(sourceType.rawValue)",
                              "model": "\(model)"])

        let records = try await database.withTransaction { transaction in
            var results: [EmbeddingRecord] = []

            let prefix = TupleHelpers.encodeSourceTypeIndexRangePrefix(
                rootPrefix: self.rootPrefix,
                sourceType: sourceType.rawValue,
                model: model
            )
            let (beginKey, endKey) = TupleHelpers.encodeRangeKeys(prefix: prefix)

            let sequence = transaction.getRange(
                beginSelector: .firstGreaterOrEqual(beginKey),
                endSelector: .firstGreaterOrEqual(endKey),
                snapshot: true
            )

            // Index only contains IDs, need to fetch actual records
            for try await (key, _) in sequence {
                // Extract ID from key: (rootPrefix, "index", "source", sourceType, model, id)
                let keyWithoutPrefix = Array(key.dropFirst(Tuple(self.rootPrefix, "index", "source", sourceType.rawValue, model).encode().count))
                let elements = try Tuple.decode(from: keyWithoutPrefix)

                if let id = elements.first as? String {
                    let embeddingKey = TupleHelpers.encodeEmbeddingKey(
                        rootPrefix: self.rootPrefix,
                        model: model,
                        id: id
                    )

                    if let bytes = try await transaction.getValue(for: embeddingKey, snapshot: true) {
                        let record = try self.decodeEmbedding(bytes: bytes)
                        results.append(record)
                    }
                }
            }

            return results
        }

        logger.debug("Retrieved embeddings by source type",
                    metadata: ["sourceType": "\(sourceType.rawValue)",
                              "model": "\(model)",
                              "count": "\(records.count)"])

        return records
    }

    /// Count the number of embeddings for a specific model
    ///
    /// This reads the atomic counter maintained by save/delete operations.
    ///
    /// - Parameter model: The model name
    /// - Returns: The count of embeddings
    public func countByModel(model: String) async throws -> UInt64 {
        logger.debug("Counting embeddings for model",
                    metadata: ["model": "\(model)"])

        let count = try await database.withTransaction { transaction in
            let countKey = TupleHelpers.encodeStatsKey(
                rootPrefix: self.rootPrefix,
                model: model,
                statType: "count"
            )

            guard let bytes = try await transaction.getValue(for: countKey, snapshot: true) else {
                return UInt64(0)
            }

            // Decode as Int64 (since we use Int64 for atomic operations)
            let signedCount = TupleHelpers.decodeInt64(bytes)
            // Count should never be negative, but handle gracefully
            return UInt64(max(0, signedCount))
        }

        logger.debug("Model embedding count",
                    metadata: ["model": "\(model)", "count": "\(count)"])

        return count
    }

    // MARK: - Cache Management

    /// Update cache with a new or updated record
    ///
    /// Implements LRU eviction when cache limit is reached.
    ///
    /// - Parameter record: The record to cache
    private func updateCache(record: EmbeddingRecord) {
        let cacheKey = "\(record.model):\(record.id)"

        // Update or add to cache
        if var entry = vectorCache[cacheKey] {
            entry.updateAccess()
            vectorCache[cacheKey] = entry

            // Update access order
            if let index = cacheAccessOrder.firstIndex(of: cacheKey) {
                cacheAccessOrder.remove(at: index)
            }
            cacheAccessOrder.append(cacheKey)
        } else {
            // New entry
            vectorCache[cacheKey] = CacheEntry(record: record)
            cacheAccessOrder.append(cacheKey)

            // Evict if over limit
            if vectorCache.count > cacheLimit {
                evictOldestFromCache()
            }
        }
    }

    /// Evict the least recently used entry from cache
    private func evictOldestFromCache() {
        guard !cacheAccessOrder.isEmpty else { return }

        let oldestKey = cacheAccessOrder.removeFirst()
        vectorCache.removeValue(forKey: oldestKey)

        logger.trace("Evicted cache entry",
                    metadata: ["key": "\(oldestKey)",
                              "cache_size": "\(vectorCache.count)"])
    }

    /// Clear the entire cache
    ///
    /// Useful for testing or when memory needs to be freed.
    public func clearCache() {
        logger.debug("Clearing embedding cache",
                    metadata: ["cache_size": "\(vectorCache.count)"])

        vectorCache.removeAll()
        cacheAccessOrder.removeAll()

        logger.debug("Cache cleared")
    }

    // MARK: - Helper Methods

    /// Encode an embedding record to bytes
    ///
    /// Format:
    /// - Vector bytes (dimension * 4 bytes)
    /// - JSON-encoded metadata (id, model, dimension, sourceType, timestamps, metadata)
    ///
    /// - Parameter record: The embedding record to encode
    /// - Returns: Encoded bytes
    /// - Throws: `EmbeddingError.encodingError` if encoding fails
    private func encodeEmbedding(record: EmbeddingRecord) throws -> FDB.Bytes {
        // Encode vector
        let vectorBytes = VectorCodec.encode(record.vector)

        // Encode metadata as JSON
        let metadata: [String: Any] = [
            "id": record.id,
            "model": record.model,
            "dimension": record.dimension,
            "sourceType": record.sourceType.rawValue,
            "createdAt": record.createdAt.timeIntervalSince1970,
            "updatedAt": record.updatedAt?.timeIntervalSince1970 ?? NSNull(),
            "metadata": record.metadata ?? NSNull()
        ]

        guard let jsonData = try? JSONSerialization.data(withJSONObject: metadata) else {
            throw EmbeddingError.encodingError("Failed to encode metadata to JSON")
        }

        let metadataBytes = [UInt8](jsonData)

        // Combine: [vector_length(4 bytes)] + [vector_bytes] + [metadata_bytes]
        var result = FDB.Bytes()
        result.reserveCapacity(4 + vectorBytes.count + metadataBytes.count)

        // Write vector length
        let vectorLength = UInt32(vectorBytes.count)
        result.append(contentsOf: withUnsafeBytes(of: vectorLength.littleEndian) { Array($0) })

        // Write vector bytes
        result.append(contentsOf: vectorBytes)

        // Write metadata bytes
        result.append(contentsOf: metadataBytes)

        return result
    }

    /// Decode bytes to an embedding record
    ///
    /// - Parameter bytes: The encoded bytes
    /// - Returns: Decoded embedding record
    /// - Throws: `EmbeddingError.encodingError` if decoding fails
    private func decodeEmbedding(bytes: FDB.Bytes) throws -> EmbeddingRecord {
        guard bytes.count >= 4 else {
            throw EmbeddingError.encodingError("Insufficient bytes for decoding")
        }

        // Read vector length
        let vectorLength = bytes[0..<4].withUnsafeBytes {
            $0.load(as: UInt32.self).littleEndian
        }

        guard bytes.count >= 4 + Int(vectorLength) else {
            throw EmbeddingError.encodingError("Insufficient bytes for vector data")
        }

        // Read vector bytes
        let vectorBytes = Array(bytes[4..<(4 + Int(vectorLength))])

        // Read metadata bytes
        let metadataBytes = Array(bytes[(4 + Int(vectorLength))...])

        // Decode metadata JSON
        guard let jsonObject = try? JSONSerialization.jsonObject(with: Data(metadataBytes)),
              let metadata = jsonObject as? [String: Any] else {
            throw EmbeddingError.encodingError("Failed to decode metadata JSON")
        }

        // Extract fields
        guard let id = metadata["id"] as? String,
              let model = metadata["model"] as? String,
              let dimension = metadata["dimension"] as? Int,
              let sourceTypeRaw = metadata["sourceType"] as? String,
              let sourceType = SourceType(rawValue: sourceTypeRaw),
              let createdAtTimestamp = metadata["createdAt"] as? TimeInterval else {
            throw EmbeddingError.encodingError("Missing required metadata fields")
        }

        // Decode vector
        let vector = try VectorCodec.decode(vectorBytes, dimension: dimension)

        // Optional fields
        let updatedAt: Date?
        if let updatedAtTimestamp = metadata["updatedAt"] as? TimeInterval {
            updatedAt = Date(timeIntervalSince1970: updatedAtTimestamp)
        } else {
            updatedAt = nil
        }

        let recordMetadata = metadata["metadata"] as? [String: String]

        return EmbeddingRecord(
            id: id,
            vector: vector,
            model: model,
            dimension: dimension,
            sourceType: sourceType,
            createdAt: Date(timeIntervalSince1970: createdAtTimestamp),
            updatedAt: updatedAt,
            metadata: recordMetadata
        )
    }
}
