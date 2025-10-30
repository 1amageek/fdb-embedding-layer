# fdb-embedding-layer Data Model

## Overview

This document defines the complete data structures used in `fdb-embedding-layer`. The embedding layer transforms structured knowledge (triples, ontology classes, text) into dense vector representations for semantic search and similarity analysis.

All data models are designed with these principles:
- **Codable**: JSON serialization for FoundationDB storage and API interchange
- **Sendable**: Safe concurrent access across actors (Swift 6 compatibility)
- **Type-safe**: Strong typing to prevent errors at compile time
- **Immutable**: Structs are immutable by default for thread safety
- **Validated**: Input validation to ensure data integrity

---

## Table of Contents

1. [Core Models](#core-models)
   - [EmbeddingRecord](#embeddingrecord)
   - [EmbeddingModelMetadata](#embeddingmodelmetadata)
   - [SearchResult](#searchresult)
   - [SimilarityMetric](#similaritymetric)
   - [SourceType](#sourcetype)
2. [Supporting Models](#supporting-models)
   - [EmbeddingBatch](#embeddingbatch)
   - [VectorStatistics](#vectorstatistics)
   - [ModelVersion](#modelversion)
   - [RegenerationJob](#regenerationjob)
3. [Error Types](#error-types)
4. [Validation Rules](#validation-rules)
5. [Relationships](#relationships)
6. [Serialization](#serialization)
7. [Migration Strategy](#migration-strategy)
8. [Usage Examples](#usage-examples)

---

## Core Models

### EmbeddingRecord

Represents a single vector embedding with metadata linking it to source data.

```swift
public struct EmbeddingRecord: Codable, Sendable {
    /// Unique identifier for the source entity
    /// Format examples:
    /// - "triple:12345" - RDF triple embedding
    /// - "class:Person" - Ontology class embedding
    /// - "text:abc123" - Raw text embedding
    public let id: String

    /// Dense embedding vector (typically Float32)
    /// Dimension depends on the model (e.g., 384, 768, 1024, 1536)
    public let vector: [Float]

    /// Embedding model identifier (e.g., "mlx-embed-1024-v1", "text-embedding-3-large")
    public let model: String

    /// Vector dimension (must match vector.count)
    public let dimension: Int

    /// Source type indicating origin of the embedding
    public let sourceType: SourceType

    /// Optional metadata for additional context
    /// Examples:
    /// - ["uri": "http://example.org/person/Alice"]
    /// - ["tripleId": "12345", "predicate": "knows"]
    /// - ["text": "Original input text", "hash": "sha256:..."]
    public let metadata: [String: String]?

    /// Creation timestamp (when the embedding was generated)
    public let createdAt: Date

    /// Optional update timestamp (for regeneration tracking)
    public let updatedAt: Date?

    public init(
        id: String,
        vector: [Float],
        model: String,
        dimension: Int,
        sourceType: SourceType,
        metadata: [String: String]? = nil,
        createdAt: Date = Date(),
        updatedAt: Date? = nil
    ) {
        self.id = id
        self.vector = vector
        self.model = model
        self.dimension = dimension
        self.sourceType = sourceType
        self.metadata = metadata
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}
```

#### Properties

| Property | Type | Description | Constraints |
|----------|------|-------------|-------------|
| `id` | String | Unique identifier | Non-empty, max 1024 chars |
| `vector` | [Float] | Embedding vector | Length = dimension, all finite values |
| `model` | String | Model identifier | Non-empty, registered in ModelManager |
| `dimension` | Int | Vector dimension | 1-8192, power of 2 recommended |
| `sourceType` | SourceType | Source category | Enum value |
| `metadata` | [String: String]? | Optional metadata | Max 10 keys, 256 chars per value |
| `createdAt` | Date | Creation time | Valid timestamp |
| `updatedAt` | Date? | Update time | Valid timestamp if present |

#### ID Format Conventions

```swift
// Triple embeddings
"triple:{tripleId}"          // e.g., "triple:12345"

// Ontology class embeddings
"class:{className}"          // e.g., "class:Person"

// Raw text embeddings
"text:{hash}"                // e.g., "text:sha256:abc123..."

// Entity embeddings (from URI)
"entity:{uriHash}"           // e.g., "entity:hash:def456..."

// Batch embeddings
"batch:{batchId}:{index}"    // e.g., "batch:job123:0"
```

#### Example Usage

```swift
// Create triple embedding
let tripleEmbedding = EmbeddingRecord(
    id: "triple:12345",
    vector: [0.123, -0.456, 0.789, ...], // 1024 dimensions
    model: "mlx-embed-1024-v1",
    dimension: 1024,
    sourceType: .triple,
    metadata: [
        "subject": "http://example.org/person/Alice",
        "predicate": "http://xmlns.com/foaf/0.1/knows",
        "object": "http://example.org/person/Bob"
    ],
    createdAt: Date()
)

// Create class embedding
let classEmbedding = EmbeddingRecord(
    id: "class:Person",
    vector: generateEmbedding(text: "Person: A human being with name, age, and relationships"),
    model: "text-embedding-3-large",
    dimension: 1536,
    sourceType: .entity,
    metadata: [
        "className": "Person",
        "description": "A human being"
    ]
)

// Create text embedding
let textHash = SHA256.hash(data: Data(query.utf8)).hexString
let textEmbedding = EmbeddingRecord(
    id: "text:\(textHash)",
    vector: generateEmbedding(text: query),
    model: "mlx-embed-1024-v1",
    dimension: 1024,
    sourceType: .text,
    metadata: ["text": query]
)
```

---

### EmbeddingModelMetadata

Defines characteristics and metadata for an embedding model.

```swift
public struct EmbeddingModelMetadata: Codable, Sendable {
    /// Unique model identifier (e.g., "mlx-embed-1024-v1")
    public let name: String

    /// Semantic version (e.g., "1.0.0", "2.1.3")
    public let version: String

    /// Output vector dimension
    public let dimension: Int

    /// Provider/source (e.g., "Apple MLX", "OpenAI", "Cohere", "Local")
    public let provider: String

    /// Model architecture/type (e.g., "BERT", "GPT", "Sentence-BERT", "Custom")
    public let modelType: String

    /// Human-readable description
    public let description: String?

    /// Whether vectors are L2-normalized (unit vectors)
    public let normalized: Bool

    /// Maximum input length (tokens or characters)
    public let maxInputLength: Int?

    /// Supported languages (ISO 639-1 codes)
    public let supportedLanguages: [String]?

    /// Model registration timestamp
    public let createdAt: Date

    /// Optional deprecation info
    public let deprecated: Bool
    public let deprecationReason: String?
    public let replacementModel: String?

    public init(
        name: String,
        version: String,
        dimension: Int,
        provider: String,
        modelType: String,
        description: String? = nil,
        normalized: Bool = true,
        maxInputLength: Int? = nil,
        supportedLanguages: [String]? = nil,
        createdAt: Date = Date(),
        deprecated: Bool = false,
        deprecationReason: String? = nil,
        replacementModel: String? = nil
    ) {
        self.name = name
        self.version = version
        self.dimension = dimension
        self.provider = provider
        self.modelType = modelType
        self.description = description
        self.normalized = normalized
        self.maxInputLength = maxInputLength
        self.supportedLanguages = supportedLanguages
        self.createdAt = createdAt
        self.deprecated = deprecated
        self.deprecationReason = deprecationReason
        self.replacementModel = replacementModel
    }
}
```

#### Properties

| Property | Type | Description | Constraints |
|----------|------|-------------|-------------|
| `name` | String | Unique identifier | Non-empty, alphanumeric + "-_" |
| `version` | String | Semantic version | Format: "major.minor.patch" |
| `dimension` | Int | Vector dimension | 1-8192 |
| `provider` | String | Model provider | Non-empty |
| `modelType` | String | Architecture type | Non-empty |
| `description` | String? | Description | Max 1024 chars |
| `normalized` | Bool | L2-normalized flag | Default: true |
| `maxInputLength` | Int? | Max input length | 1-1000000 if present |
| `supportedLanguages` | [String]? | Language codes | ISO 639-1 format |
| `deprecated` | Bool | Deprecation status | Default: false |

#### Example Usage

```swift
// MLX local model
let mlxModel = EmbeddingModelMetadata(
    name: "mlx-embed-1024-v1",
    version: "1.0.0",
    dimension: 1024,
    provider: "Apple MLX",
    modelType: "Sentence-BERT",
    description: "Local MLX-based sentence embedding model",
    normalized: true,
    maxInputLength: 512,
    supportedLanguages: ["en", "ja"],
    createdAt: Date()
)

// OpenAI model
let openAIModel = EmbeddingModelMetadata(
    name: "text-embedding-3-large",
    version: "3.0.0",
    dimension: 1536,
    provider: "OpenAI",
    modelType: "GPT-based",
    description: "OpenAI's large embedding model with 1536 dimensions",
    normalized: true,
    maxInputLength: 8191
)

// Deprecated model with replacement
let oldModel = EmbeddingModelMetadata(
    name: "mlx-embed-768-v1",
    version: "1.0.0",
    dimension: 768,
    provider: "Apple MLX",
    modelType: "BERT",
    deprecated: true,
    deprecationReason: "Superseded by higher dimension model",
    replacementModel: "mlx-embed-1024-v1"
)
```

---

### SearchResult

Represents a single result from similarity search.

```swift
public struct SearchResult: Sendable {
    /// Entity identifier matching the result
    public let id: String

    /// Similarity score (0.0-1.0, higher = more similar)
    /// - Cosine similarity: range [0.0, 1.0] (normalized)
    /// - Inner product: unnormalized dot product
    public let score: Float

    /// Optional: return the actual vector
    public let vector: [Float]?

    /// Optional: metadata from the embedding record
    public let metadata: [String: String]?

    /// Raw distance value (metric-dependent)
    /// - Cosine: 1.0 - score
    /// - Euclidean: L2 distance
    /// - Manhattan: L1 distance
    public let distance: Float

    /// Source type of the embedding
    public let sourceType: SourceType?

    /// Model used for the embedding
    public let model: String?

    public init(
        id: String,
        score: Float,
        vector: [Float]? = nil,
        metadata: [String: String]? = nil,
        distance: Float,
        sourceType: SourceType? = nil,
        model: String? = nil
    ) {
        self.id = id
        self.score = score
        self.vector = vector
        self.metadata = metadata
        self.distance = distance
        self.sourceType = sourceType
        self.model = model
    }
}
```

#### Properties

| Property | Type | Description | Range |
|----------|------|-------------|-------|
| `id` | String | Entity identifier | Non-empty |
| `score` | Float | Similarity score | 0.0-1.0 (cosine), unbounded (inner product) |
| `vector` | [Float]? | Optional vector | Finite values if present |
| `metadata` | [String: String]? | Optional metadata | - |
| `distance` | Float | Raw distance | Metric-dependent |
| `sourceType` | SourceType? | Source category | Enum value if present |
| `model` | String? | Model identifier | - |

#### Example Usage

```swift
// Search for similar entities
let results: [SearchResult] = try await searchEngine.search(
    queryVector: queryVector,
    topK: 10,
    model: "mlx-embed-1024-v1",
    similarityMetric: .cosine
)

// Process results
for result in results {
    print("ID: \(result.id)")
    print("Score: \(result.score)") // e.g., 0.95
    print("Distance: \(result.distance)") // e.g., 0.05

    if let metadata = result.metadata {
        print("Source: \(metadata["uri"] ?? "unknown")")
    }
}

// Result with vector included
let resultWithVector = SearchResult(
    id: "triple:12345",
    score: 0.95,
    vector: embeddingVector, // Include for reranking
    metadata: ["subject": "Alice", "predicate": "knows"],
    distance: 0.05,
    sourceType: .triple,
    model: "mlx-embed-1024-v1"
)
```

---

### SimilarityMetric

Defines the distance/similarity metric for search.

```swift
public enum SimilarityMetric: String, Codable, Sendable {
    /// Cosine similarity: dot(a, b) / (norm(a) * norm(b))
    /// Range: [-1, 1], typically [0, 1] for normalized vectors
    /// Use case: General-purpose semantic similarity
    case cosine

    /// Inner product (dot product): dot(a, b)
    /// Range: unbounded
    /// Use case: Normalized vectors, fast similarity
    case innerProduct

    /// Euclidean distance (L2): sqrt(sum((a - b)^2))
    /// Range: [0, inf)
    /// Use case: Geometric distance in embedding space
    case euclidean

    /// Manhattan distance (L1): sum(|a - b|)
    /// Range: [0, inf)
    /// Use case: Grid-like distance, robust to outliers
    case manhattan
}
```

#### Metric Comparison

| Metric | Formula | Range | Normalized Required | Use Case |
|--------|---------|-------|---------------------|----------|
| Cosine | cos(θ) = a·b / (\|\|a\|\| \|\|b\|\|) | [0, 1] | No (auto-normalized) | Semantic similarity |
| Inner Product | a·b | (-∞, +∞) | Yes (for bounded results) | Fast similarity on normalized vectors |
| Euclidean | \|\|a - b\|\|₂ | [0, +∞) | No | Geometric distance |
| Manhattan | \|\|a - b\|\|₁ | [0, +∞) | No | Robust distance metric |

#### Example Usage

```swift
// Cosine similarity (most common)
let results = try await searchEngine.search(
    queryVector: query,
    topK: 10,
    model: "mlx-embed-1024-v1",
    similarityMetric: .cosine
)

// Inner product (faster for normalized vectors)
let results = try await searchEngine.search(
    queryVector: normalizedQuery,
    topK: 10,
    model: "mlx-embed-1024-v1",
    similarityMetric: .innerProduct
)

// Euclidean distance
let results = try await searchEngine.search(
    queryVector: query,
    topK: 10,
    model: "mlx-embed-1024-v1",
    similarityMetric: .euclidean
)
```

#### Converting Between Metrics

```swift
// For normalized vectors: cosine similarity ≈ 1 - (euclidean^2 / 2)
func euclideanToCosine(_ euclideanDist: Float) -> Float {
    return 1.0 - (euclideanDist * euclideanDist / 2.0)
}

// For normalized vectors: inner product = cosine similarity
func innerProductToCosine(_ innerProduct: Float, normA: Float, normB: Float) -> Float {
    return innerProduct / (normA * normB)
}
```

---

### SourceType

Categorizes the origin of an embedding.

```swift
public enum SourceType: String, Codable, Sendable {
    /// Generated from an RDF triple (subject-predicate-object)
    /// ID format: "triple:{id}"
    case triple

    /// Generated from an ontology class or entity definition
    /// ID format: "class:{name}" or "entity:{uri}"
    case entity

    /// Generated from raw text input
    /// ID format: "text:{hash}"
    case text

    /// Part of a batch generation job
    /// ID format: "batch:{jobId}:{index}"
    case batch
}
```

#### Example Usage

```swift
// Filter search by source type
let filter = SearchFilter(
    sourceTypes: [.triple, .entity],
    metadata: nil,
    createdAfter: nil,
    createdBefore: nil
)

let results = try await searchEngine.searchWithFilter(
    queryVector: query,
    topK: 10,
    model: "mlx-embed-1024-v1",
    filter: filter
)

// Count embeddings by source type
let tripleCount = try await store.countBySourceType(.triple, model: "mlx-embed-1024-v1")
let entityCount = try await store.countBySourceType(.entity, model: "mlx-embed-1024-v1")
```

---

## Supporting Models

### EmbeddingBatch

Represents a batch operation for bulk processing.

```swift
public struct EmbeddingBatch: Codable, Sendable {
    /// Unique batch identifier
    public let id: String

    /// List of embedding records in this batch
    public let records: [EmbeddingRecord]

    /// Model used for all embeddings in batch
    public let model: String

    /// Total number of embeddings
    public var count: Int { records.count }

    /// Batch creation timestamp
    public let createdAt: Date

    /// Optional batch metadata
    public let metadata: [String: String]?

    public init(
        id: String,
        records: [EmbeddingRecord],
        model: String,
        metadata: [String: String]? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.records = records
        self.model = model
        self.metadata = metadata
        self.createdAt = createdAt
    }
}
```

#### Example Usage

```swift
// Create batch from triples
let triples = try await tripleStore.list(limit: 100)
let texts = triples.map { "\($0.subject) \($0.predicate) \($0.object)" }
let vectors = try await generator.generateBatch(texts: texts)

let records = zip(triples, vectors).map { (triple, vector) in
    EmbeddingRecord(
        id: "triple:\(triple.id)",
        vector: vector,
        model: "mlx-embed-1024-v1",
        dimension: 1024,
        sourceType: .triple
    )
}

let batch = EmbeddingBatch(
    id: UUID().uuidString,
    records: records,
    model: "mlx-embed-1024-v1",
    metadata: ["batchType": "tripleEmbedding"]
)

try await store.saveBatch(batch.records)
```

---

### VectorStatistics

Statistical information about a set of vectors.

```swift
public struct VectorStatistics: Codable, Sendable {
    /// Number of vectors analyzed
    public let count: Int

    /// Vector dimension
    public let dimension: Int

    /// Mean vector (component-wise average)
    public let mean: [Float]

    /// Standard deviation (component-wise)
    public let stdDev: [Float]

    /// Minimum values (component-wise)
    public let min: [Float]

    /// Maximum values (component-wise)
    public let max: [Float]

    /// L2 norm statistics
    public let normStats: NormStatistics

    public init(
        count: Int,
        dimension: Int,
        mean: [Float],
        stdDev: [Float],
        min: [Float],
        max: [Float],
        normStats: NormStatistics
    ) {
        self.count = count
        self.dimension = dimension
        self.mean = mean
        self.stdDev = stdDev
        self.min = min
        self.max = max
        self.normStats = normStats
    }
}

public struct NormStatistics: Codable, Sendable {
    public let mean: Float
    public let stdDev: Float
    public let min: Float
    public let max: Float

    public init(mean: Float, stdDev: Float, min: Float, max: Float) {
        self.mean = mean
        self.stdDev = stdDev
        self.min = min
        self.max = max
    }
}
```

#### Example Usage

```swift
// Compute statistics for a model's embeddings
let embeddings = try await store.listByModel("mlx-embed-1024-v1", limit: 10000)
let stats = computeStatistics(embeddings: embeddings)

print("Count: \(stats.count)")
print("Dimension: \(stats.dimension)")
print("Mean L2 norm: \(stats.normStats.mean)")
print("Norm std dev: \(stats.normStats.stdDev)")

// Check if vectors are normalized
if stats.normStats.mean > 0.99 && stats.normStats.mean < 1.01 {
    print("Vectors appear to be L2-normalized")
}
```

---

### ModelVersion

Tracks versioning information for embedding models.

```swift
public struct ModelVersion: Codable, Sendable {
    /// Model name
    public let name: String

    /// Version number (semantic versioning)
    public let version: String

    /// Release date
    public let releaseDate: Date

    /// Changes in this version
    public let changes: [String]

    /// Breaking changes flag
    public let breaking: Bool

    /// Migration guide URL
    public let migrationGuide: String?

    /// Previous version (if applicable)
    public let previousVersion: String?

    public init(
        name: String,
        version: String,
        releaseDate: Date = Date(),
        changes: [String],
        breaking: Bool = false,
        migrationGuide: String? = nil,
        previousVersion: String? = nil
    ) {
        self.name = name
        self.version = version
        self.releaseDate = releaseDate
        self.changes = changes
        self.breaking = breaking
        self.migrationGuide = migrationGuide
        self.previousVersion = previousVersion
    }
}
```

#### Example Usage

```swift
// Create version record
let version = ModelVersion(
    name: "mlx-embed-1024-v2",
    version: "2.0.0",
    releaseDate: Date(),
    changes: [
        "Increased dimension from 768 to 1024",
        "Improved multilingual support",
        "Better handling of technical terms"
    ],
    breaking: true,
    migrationGuide: "https://docs.example.com/migration/v2",
    previousVersion: "1.0.0"
)

// Track version history
let versions = try await modelManager.listVersions(modelName: "mlx-embed")
for v in versions {
    print("\(v.version) - \(v.releaseDate): \(v.changes.joined(separator: ", "))")
}
```

---

### RegenerationJob

Tracks background embedding regeneration jobs.

```swift
public struct RegenerationJob: Codable, Sendable {
    /// Unique job identifier
    public let id: String

    /// Source model to migrate from
    public let sourceModel: String

    /// Target model to migrate to
    public let targetModel: String

    /// Source type filter (nil = all types)
    public let sourceType: SourceType?

    /// Job status
    public var status: JobStatus

    /// Total entities to process
    public let totalCount: Int

    /// Entities processed so far
    public var processedCount: Int

    /// Entities failed
    public var failedCount: Int

    /// Job creation time
    public let createdAt: Date

    /// Job start time
    public var startedAt: Date?

    /// Job completion time
    public var completedAt: Date?

    /// Error message (if failed)
    public var errorMessage: String?

    public init(
        id: String = UUID().uuidString,
        sourceModel: String,
        targetModel: String,
        sourceType: SourceType? = nil,
        status: JobStatus = .pending,
        totalCount: Int,
        processedCount: Int = 0,
        failedCount: Int = 0,
        createdAt: Date = Date(),
        startedAt: Date? = nil,
        completedAt: Date? = nil,
        errorMessage: String? = nil
    ) {
        self.id = id
        self.sourceModel = sourceModel
        self.targetModel = targetModel
        self.sourceType = sourceType
        self.status = status
        self.totalCount = totalCount
        self.processedCount = processedCount
        self.failedCount = failedCount
        self.createdAt = createdAt
        self.startedAt = startedAt
        self.completedAt = completedAt
        self.errorMessage = errorMessage
    }
}

public enum JobStatus: String, Codable, Sendable {
    case pending
    case running
    case completed
    case failed
    case cancelled
}
```

#### Example Usage

```swift
// Create regeneration job
let job = RegenerationJob(
    sourceModel: "mlx-embed-768-v1",
    targetModel: "mlx-embed-1024-v1",
    sourceType: .triple,
    totalCount: 10000
)

// Track progress
actor JobManager {
    func updateProgress(jobId: String, processed: Int) async throws {
        var job = try await getJob(jobId)
        job.processedCount = processed

        if processed == job.totalCount {
            job.status = .completed
            job.completedAt = Date()
        }

        try await saveJob(job)
    }

    func getProgress(jobId: String) async throws -> Float {
        let job = try await getJob(jobId)
        return Float(job.processedCount) / Float(job.totalCount)
    }
}
```

---

## Error Types

### EmbeddingError

Comprehensive error types for embedding operations.

```swift
public enum EmbeddingError: Error, Sendable {
    /// Model not found in registry
    case modelNotFound(String)

    /// Vector dimension mismatch
    case dimensionMismatch(expected: Int, actual: Int)

    /// Vector not found for given ID
    case vectorNotFound(String)

    /// Invalid vector (NaN, Inf, wrong dimension)
    case invalidVector(String)

    /// Storage/database error
    case storageError(String)

    /// Encoding/decoding error
    case encodingError(String)

    /// Invalid model metadata
    case invalidModel(String)

    /// Search error
    case searchError(String)

    /// Generation error
    case generationError(String)

    /// Cache error
    case cacheError(String)

    /// Validation error
    case validationError(String)

    /// Migration error
    case migrationError(String)
}

extension EmbeddingError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Embedding model '\(name)' not found in registry"
        case .dimensionMismatch(let expected, let actual):
            return "Vector dimension mismatch: expected \(expected), got \(actual)"
        case .vectorNotFound(let id):
            return "Embedding vector not found for ID: \(id)"
        case .invalidVector(let reason):
            return "Invalid vector: \(reason)"
        case .storageError(let message):
            return "Storage error: \(message)"
        case .encodingError(let message):
            return "Encoding error: \(message)"
        case .invalidModel(let reason):
            return "Invalid model: \(reason)"
        case .searchError(let message):
            return "Search error: \(message)"
        case .generationError(let message):
            return "Generation error: \(message)"
        case .cacheError(let message):
            return "Cache error: \(message)"
        case .validationError(let message):
            return "Validation error: \(message)"
        case .migrationError(let message):
            return "Migration error: \(message)"
        }
    }
}
```

#### Example Usage

```swift
// Handle specific errors
do {
    try await store.save(record)
} catch EmbeddingError.modelNotFound(let name) {
    // Register the model first
    print("Model \(name) not registered, registering now...")
    try await modelManager.registerModel(modelMetadata)
    try await store.save(record)
} catch EmbeddingError.dimensionMismatch(let expected, let actual) {
    // Dimension mismatch - cannot recover
    print("Fatal: dimension mismatch \(expected) != \(actual)")
    throw error
} catch EmbeddingError.storageError {
    // Retry on storage error
    try await Task.sleep(nanoseconds: 1_000_000_000)
    try await store.save(record)
}
```

---

## Validation Rules

### EmbeddingRecord Validation

```swift
extension EmbeddingRecord {
    /// Validate the embedding record
    public func validate() throws {
        // ID validation
        guard !id.isEmpty else {
            throw EmbeddingError.validationError("ID cannot be empty")
        }
        guard id.count <= 1024 else {
            throw EmbeddingError.validationError("ID exceeds 1024 characters")
        }

        // Model validation
        guard !model.isEmpty else {
            throw EmbeddingError.validationError("Model name cannot be empty")
        }

        // Dimension validation
        guard dimension > 0 && dimension <= 8192 else {
            throw EmbeddingError.validationError("Dimension must be between 1 and 8192")
        }
        guard vector.count == dimension else {
            throw EmbeddingError.dimensionMismatch(expected: dimension, actual: vector.count)
        }

        // Vector validation
        guard vector.allSatisfy({ $0.isFinite }) else {
            throw EmbeddingError.invalidVector("Vector contains NaN or Inf values")
        }

        // Metadata validation
        if let metadata = metadata {
            guard metadata.count <= 10 else {
                throw EmbeddingError.validationError("Metadata cannot exceed 10 keys")
            }
            for (key, value) in metadata {
                guard key.count <= 256 && value.count <= 256 else {
                    throw EmbeddingError.validationError("Metadata key/value exceeds 256 characters")
                }
            }
        }
    }
}
```

### EmbeddingModelMetadata Validation

```swift
extension EmbeddingModelMetadata {
    /// Validate model metadata
    public func validate() throws {
        // Name validation
        guard !name.isEmpty else {
            throw EmbeddingError.validationError("Model name cannot be empty")
        }
        guard name.count <= 256 else {
            throw EmbeddingError.validationError("Model name exceeds 256 characters")
        }
        let validCharacters = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        guard name.unicodeScalars.allSatisfy({ validCharacters.contains($0) }) else {
            throw EmbeddingError.validationError("Model name contains invalid characters")
        }

        // Version validation (semantic versioning)
        let versionRegex = #"^\d+\.\d+\.\d+$"#
        guard version.range(of: versionRegex, options: .regularExpression) != nil else {
            throw EmbeddingError.validationError("Version must follow semantic versioning (x.y.z)")
        }

        // Dimension validation
        guard dimension > 0 && dimension <= 8192 else {
            throw EmbeddingError.validationError("Dimension must be between 1 and 8192")
        }

        // Max input length validation
        if let maxLength = maxInputLength {
            guard maxLength > 0 && maxLength <= 1_000_000 else {
                throw EmbeddingError.validationError("Max input length must be between 1 and 1,000,000")
            }
        }

        // Language codes validation
        if let languages = supportedLanguages {
            for code in languages {
                guard code.count == 2 else {
                    throw EmbeddingError.validationError("Language codes must be ISO 639-1 (2 characters)")
                }
            }
        }
    }
}
```

### Vector Normalization Validation

```swift
extension Array where Element == Float {
    /// Check if vector is L2-normalized (unit vector)
    public var isNormalized: Bool {
        let norm = sqrt(self.reduce(0) { $0 + $1 * $1 })
        return abs(norm - 1.0) < 0.01 // Allow 1% tolerance
    }

    /// Normalize vector to unit length
    public func normalized() -> [Float] {
        let norm = sqrt(self.reduce(0) { $0 + $1 * $1 })
        guard norm > 0 else { return self }
        return self.map { $0 / norm }
    }

    /// Validate vector contains only finite values
    public var isValid: Bool {
        return self.allSatisfy { $0.isFinite }
    }
}
```

---

## Relationships

### Triple Layer Integration

```
Triple (fdb-triple-layer)
    │
    ├─► ID: Int64
    ├─► Subject: Value
    ├─► Predicate: Value
    └─► Object: Value
         │
         └─► EmbeddingRecord (fdb-embedding-layer)
              ├─► id: "triple:{tripleId}"
              ├─► vector: [Float]
              ├─► sourceType: .triple
              └─► metadata: {subject, predicate, object URIs}
```

### Ontology Layer Integration

```
OntologyClass (fdb-ontology-layer)
    │
    ├─► name: String
    ├─► description: String
    └─► properties: [String]
         │
         └─► EmbeddingRecord (fdb-embedding-layer)
              ├─► id: "class:{className}"
              ├─► vector: [Float]
              ├─► sourceType: .entity
              └─► metadata: {className, description}
```

### Knowledge Layer Integration

```
Knowledge Layer Query
    │
    ├─► Structured Query (SPARQL-like)
    │    └─► fdb-triple-layer
    │
    ├─► Semantic Query (Vector-based)
    │    └─► fdb-embedding-layer
    │         ├─► SearchEngine.search()
    │         └─► Returns: [SearchResult]
    │
    └─► Hybrid Query (Structured + Semantic)
         ├─► Triple Layer: filter by constraints
         ├─► Embedding Layer: rank by similarity
         └─► Merge results
```

---

## Serialization

### JSON Encoding

```swift
// EmbeddingRecord JSON
let record = EmbeddingRecord(
    id: "triple:12345",
    vector: [0.1, 0.2, 0.3],
    model: "mlx-embed-1024-v1",
    dimension: 3,
    sourceType: .triple,
    metadata: ["subject": "Alice"],
    createdAt: Date()
)

let encoder = JSONEncoder()
encoder.dateEncodingStrategy = .iso8601
let json = try encoder.encode(record)

// Result:
{
  "id": "triple:12345",
  "vector": [0.1, 0.2, 0.3],
  "model": "mlx-embed-1024-v1",
  "dimension": 3,
  "sourceType": "triple",
  "metadata": {
    "subject": "Alice"
  },
  "createdAt": "2025-10-30T12:34:56Z"
}
```

### Binary Encoding (FoundationDB)

```swift
// Efficient binary encoding for FDB storage
struct VectorCodec {
    /// Encode vector as Float32 binary
    static func encode(_ vector: [Float]) -> FDB.Bytes {
        var bytes = FDB.Bytes()
        bytes.reserveCapacity(vector.count * 4)
        for value in vector {
            withUnsafeBytes(of: value.bitPattern.littleEndian) {
                bytes.append(contentsOf: $0)
            }
        }
        return bytes
    }

    /// Decode Float32 binary to vector
    static func decode(_ bytes: FDB.Bytes) -> [Float] {
        let count = bytes.count / 4
        return (0..<count).map { i in
            let offset = i * 4
            let bits = bytes[offset..<offset+4].withUnsafeBytes {
                $0.load(as: UInt32.self).littleEndian
            }
            return Float(bitPattern: bits)
        }
    }
}

// Storage example
let vectorBytes = VectorCodec.encode(record.vector)
let metadataJSON = try JSONEncoder().encode(record.metadata)

transaction.setValue(vectorBytes + metadataJSON, for: key)
```

---

## Migration Strategy

### Version Migration

```swift
actor MigrationManager {
    /// Migrate embeddings from one model to another
    func migrate(
        from oldModel: String,
        to newModel: String,
        generator: EmbeddingGenerator,
        batchSize: Int = 100
    ) async throws {
        // 1. Get all embeddings for old model
        let oldEmbeddings = try await store.listByModel(oldModel, limit: Int.max)
        let totalCount = oldEmbeddings.count

        // 2. Create migration job
        let job = RegenerationJob(
            sourceModel: oldModel,
            targetModel: newModel,
            totalCount: totalCount
        )
        try await jobManager.createJob(job)

        // 3. Process in batches
        for batch in oldEmbeddings.chunked(into: batchSize) {
            // Fetch source data
            let sourceData = try await fetchSourceData(for: batch)

            // Generate new embeddings
            let newVectors = try await generator.generateBatch(texts: sourceData)

            // Create new records
            let newRecords = zip(batch, newVectors).map { (old, vector) in
                EmbeddingRecord(
                    id: old.id,
                    vector: vector,
                    model: newModel,
                    dimension: vector.count,
                    sourceType: old.sourceType,
                    metadata: old.metadata,
                    createdAt: Date()
                )
            }

            // Save batch
            try await store.saveBatch(newRecords)

            // Update job progress
            try await jobManager.updateProgress(
                jobId: job.id,
                processed: newRecords.count
            )
        }

        // 4. Deprecate old model
        try await modelManager.deprecateModel(
            name: oldModel,
            reason: "Migrated to \(newModel)"
        )
    }
}
```

### Backward Compatibility

```swift
// Support multiple model versions simultaneously
actor MultiModelStore {
    func get(id: String, preferredModel: String) async throws -> EmbeddingRecord? {
        // Try preferred model first
        if let record = try await store.get(id: id, model: preferredModel) {
            return record
        }

        // Fall back to other models
        let allModels = try await modelManager.listModels()
        for model in allModels where model.name != preferredModel {
            if let record = try await store.get(id: id, model: model.name) {
                return record
            }
        }

        return nil
    }
}
```

---

## Usage Examples

### Complete Workflow

```swift
import FoundationDB
import EmbeddingLayer

// 1. Initialize
try await FDBClient.initialize()
let database = try FDBClient.openDatabase()

let store = EmbeddingStore(database: database, rootPrefix: "myapp")
let modelManager = ModelManager(database: database, rootPrefix: "myapp")
let searchEngine = SearchEngine(store: store)

// 2. Register model
let model = EmbeddingModelMetadata(
    name: "mlx-embed-1024-v1",
    version: "1.0.0",
    dimension: 1024,
    provider: "Apple MLX",
    modelType: "Sentence-BERT",
    normalized: true,
    maxInputLength: 512
)
try await modelManager.registerModel(model)

// 3. Generate and store embeddings
let generator = MLXEmbeddingGenerator(model: model)

// From triple
let triple = Triple(
    subject: .uri("http://example.org/person/Alice"),
    predicate: .uri("http://xmlns.com/foaf/0.1/knows"),
    object: .uri("http://example.org/person/Bob")
)
let tripleText = "\(triple.subject) \(triple.predicate) \(triple.object)"
let tripleVector = try await generator.generate(text: tripleText)

let tripleEmbedding = EmbeddingRecord(
    id: "triple:12345",
    vector: tripleVector,
    model: model.name,
    dimension: model.dimension,
    sourceType: .triple,
    metadata: [
        "subject": "Alice",
        "predicate": "knows",
        "object": "Bob"
    ]
)
try await store.save(tripleEmbedding)

// From ontology class
let personClass = OntologyClass(
    name: "Person",
    description: "A human being",
    properties: ["name", "age", "knows"]
)
let classText = "Class: \(personClass.name). \(personClass.description ?? "")"
let classVector = try await generator.generate(text: classText)

let classEmbedding = EmbeddingRecord(
    id: "class:Person",
    vector: classVector,
    model: model.name,
    dimension: model.dimension,
    sourceType: .entity,
    metadata: ["className": "Person"]
)
try await store.save(classEmbedding)

// 4. Semantic search
let query = "people who know each other"
let queryVector = try await generator.generate(text: query)

let results = try await searchEngine.search(
    queryVector: queryVector,
    topK: 10,
    model: model.name,
    similarityMetric: .cosine
)

for result in results {
    print("ID: \(result.id), Score: \(result.score)")
    if let metadata = result.metadata {
        print("  Metadata: \(metadata)")
    }
}

// 5. Batch operations
let triples = try await tripleStore.list(limit: 100)
let texts = triples.map { "\($0.subject) \($0.predicate) \($0.object)" }
let vectors = try await generator.generateBatch(texts: texts)

let records = zip(triples, vectors).map { (triple, vector) in
    EmbeddingRecord(
        id: "triple:\(triple.id)",
        vector: vector,
        model: model.name,
        dimension: model.dimension,
        sourceType: .triple
    )
}

try await store.saveBatch(records)

// 6. Filtered search
let filter = SearchFilter(
    sourceTypes: [.triple],
    metadata: nil,
    createdAfter: Date().addingTimeInterval(-86400), // Last 24 hours
    createdBefore: nil
)

let filteredResults = try await searchEngine.searchWithFilter(
    queryVector: queryVector,
    topK: 10,
    model: model.name,
    filter: filter
)
```

---

## Best Practices

### 1. Model Selection

```swift
// Choose appropriate dimension for your use case
// Small: 384 (fast, less accurate)
// Medium: 768 (balanced)
// Large: 1024-1536 (slow, more accurate)

let smallModel = EmbeddingModelMetadata(
    name: "small-embed",
    dimension: 384,
    // ... for fast search
)

let largeModel = EmbeddingModelMetadata(
    name: "large-embed",
    dimension: 1536,
    // ... for high accuracy
)
```

### 2. Vector Normalization

```swift
// Always normalize vectors for cosine similarity
let normalized = vector.normalized()

let record = EmbeddingRecord(
    id: id,
    vector: normalized,
    model: model.name,
    dimension: dimension,
    sourceType: sourceType
)
```

### 3. Batch Processing

```swift
// Process embeddings in batches for efficiency
let batchSize = 100

for batch in entities.chunked(into: batchSize) {
    let vectors = try await generator.generateBatch(texts: batch)
    let records = createRecords(batch, vectors)
    try await store.saveBatch(records)
}
```

### 4. Error Handling

```swift
// Handle errors gracefully
do {
    try await store.save(record)
} catch EmbeddingError.modelNotFound(let name) {
    // Register model and retry
    try await modelManager.registerModel(metadata)
    try await store.save(record)
} catch {
    // Log and continue
    logger.error("Failed to save embedding: \(error)")
}
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-30
**Status**: Complete
