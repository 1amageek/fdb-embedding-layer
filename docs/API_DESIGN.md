# fdb-embedding-layer API Design

## Overview

`fdb-embedding-layer` provides a comprehensive API for vector embedding storage, similarity search, and model management. The API is designed with the following principles:

- **Actor-based concurrency**: All public components are actors for thread safety
- **Type-safe**: Strong typing with Swift's type system and Sendable protocol
- **Async/await**: Modern Swift concurrency for all I/O operations
- **Composable**: Clean separation between storage, search, and model management
- **Scalable**: Batch operations and efficient caching for high throughput
- **Model-agnostic**: Support for multiple embedding models simultaneously

### Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
│            (Knowledge Layer, Custom Apps)                │
└─────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐  ┌─────────▼─────────┐  ┌───────▼────────┐
│ EmbeddingStore │  │  SearchEngine     │  │ ModelManager   │
│  (Storage)     │  │  (Similarity)     │  │  (Metadata)    │
└────────────────┘  └───────────────────┘  └────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  FoundationDB      │
                    │  (Distributed KVS) │
                    └────────────────────┘
```

---

## Table of Contents

1. [Core Components](#core-components)
   - [EmbeddingStore](#embeddingstore)
   - [ModelManager](#modelmanager)
   - [SearchEngine](#searchengine)
   - [EmbeddingGenerator](#embeddinggenerator)
2. [Method Signatures](#method-signatures)
3. [Usage Examples](#usage-examples)
4. [Error Handling](#error-handling)
5. [Performance Considerations](#performance-considerations)
6. [Best Practices](#best-practices)
7. [Integration Patterns](#integration-patterns)
8. [Advanced Features](#advanced-features)

---

## Core Components

### EmbeddingStore

The `EmbeddingStore` actor manages vector storage and retrieval in FoundationDB. It provides CRUD operations, batch processing, and LRU caching for optimal performance.

#### Initialization

```swift
public actor EmbeddingStore {
    /// Initialize the embedding store
    /// - Parameters:
    ///   - database: FoundationDB database instance
    ///   - rootPrefix: Namespace prefix (e.g., "myapp", "prod")
    ///   - cacheSize: Maximum number of vectors to cache (default: 10,000)
    public init(
        database: any DatabaseProtocol,
        rootPrefix: String,
        cacheSize: Int = 10_000
    ) async throws
}
```

**Example:**
```swift
import FoundationDB
import EmbeddingLayer

try await FDBClient.initialize()
let database = try FDBClient.openDatabase()

let store = try await EmbeddingStore(
    database: database,
    rootPrefix: "myapp",
    cacheSize: 10_000
)
```

#### CRUD Operations

##### save(_:)

Save a single embedding record to FoundationDB.

```swift
/// Save an embedding record
/// - Parameter record: The embedding record to save
/// - Throws: EmbeddingError if validation fails or storage error occurs
public func save(_ record: EmbeddingRecord) async throws
```

**Example:**
```swift
let record = EmbeddingRecord(
    id: "triple:12345",
    vector: [0.1, 0.2, 0.3, ...], // 1024 dimensions
    model: "mlx-embed-1024-v1",
    dimension: 1024,
    sourceType: .triple,
    metadata: ["subject": "Alice", "predicate": "knows"]
)

try await store.save(record)
```

**Performance:**
- Latency (p50): 5-10ms
- Latency (p99): 20-30ms
- Throughput: 1,000-5,000 ops/sec

**Error Conditions:**
- `EmbeddingError.modelNotFound`: Model not registered
- `EmbeddingError.dimensionMismatch`: Vector dimension mismatch
- `EmbeddingError.invalidVector`: Vector contains NaN/Inf
- `EmbeddingError.storageError`: FoundationDB error

##### get(id:model:)

Retrieve an embedding record by ID and model.

```swift
/// Get an embedding record
/// - Parameters:
///   - id: Entity identifier
///   - model: Model identifier
/// - Returns: The embedding record, or nil if not found
public func get(id: String, model: String) async throws -> EmbeddingRecord?
```

**Example:**
```swift
if let record = try await store.get(id: "triple:12345", model: "mlx-embed-1024-v1") {
    print("Vector dimension: \(record.dimension)")
    print("Vector norm: \(l2Norm(record.vector))")
} else {
    print("Embedding not found")
}
```

**Performance:**
- Cached: <1ms (p50), <5ms (p99)
- Uncached: 5-10ms (p50), 20-30ms (p99)

##### delete(id:model:)

Delete an embedding record.

```swift
/// Delete an embedding record
/// - Parameters:
///   - id: Entity identifier
///   - model: Model identifier
public func delete(id: String, model: String) async throws
```

**Example:**
```swift
try await store.delete(id: "triple:12345", model: "mlx-embed-1024-v1")
```

##### exists(id:model:)

Check if an embedding exists without retrieving the full vector.

```swift
/// Check if an embedding exists
/// - Parameters:
///   - id: Entity identifier
///   - model: Model identifier
/// - Returns: true if the embedding exists
public func exists(id: String, model: String) async throws -> Bool
```

**Example:**
```swift
if try await store.exists(id: "triple:12345", model: "mlx-embed-1024-v1") {
    print("Embedding exists")
}
```

#### Batch Operations

##### saveBatch(_:)

Save multiple embeddings in a single transaction (up to 1000 records).

```swift
/// Save multiple embeddings in batch
/// - Parameter records: Array of embedding records
/// - Throws: EmbeddingError if any record is invalid
/// - Note: Automatically splits into multiple transactions if > 1000 records
public func saveBatch(_ records: [EmbeddingRecord]) async throws
```

**Example:**
```swift
let records = (0..<100).map { i in
    EmbeddingRecord(
        id: "triple:\(i)",
        vector: generateVector(),
        model: "mlx-embed-1024-v1",
        dimension: 1024,
        sourceType: .triple
    )
}

try await store.saveBatch(records)
```

**Performance:**
- 100 vectors: 50-100ms (p50), 200-300ms (p99)
- Throughput: 500-1,000 batches/sec (50,000-100,000 vectors/sec)

##### getBatch(ids:model:)

Retrieve multiple embeddings in parallel.

```swift
/// Get multiple embeddings by IDs
/// - Parameters:
///   - ids: Array of entity identifiers
///   - model: Model identifier
/// - Returns: Dictionary mapping IDs to embedding records
public func getBatch(ids: [String], model: String) async throws -> [String: EmbeddingRecord]
```

**Example:**
```swift
let ids = ["triple:1", "triple:2", "triple:3"]
let embeddings = try await store.getBatch(ids: ids, model: "mlx-embed-1024-v1")

for (id, record) in embeddings {
    print("\(id): dimension \(record.dimension)")
}
```

**Performance:**
- 100 vectors: 20-50ms (p50), 100-150ms (p99)
- Parallel FDB reads for optimal throughput

##### deleteBatch(ids:model:)

Delete multiple embeddings in batch.

```swift
/// Delete multiple embeddings
/// - Parameters:
///   - ids: Array of entity identifiers
///   - model: Model identifier
public func deleteBatch(ids: [String], model: String) async throws
```

**Example:**
```swift
let ids = ["triple:1", "triple:2", "triple:3"]
try await store.deleteBatch(ids: ids, model: "mlx-embed-1024-v1")
```

#### Query Operations

##### getAllByModel(_:limit:)

List all embeddings for a specific model.

```swift
/// List embeddings by model
/// - Parameters:
///   - model: Model identifier
///   - limit: Maximum number of records to return (default: 1000)
/// - Returns: Array of embedding records
public func getAllByModel(_ model: String, limit: Int = 1000) async throws -> [EmbeddingRecord]
```

**Example:**
```swift
let embeddings = try await store.getAllByModel("mlx-embed-1024-v1", limit: 100)
print("Found \(embeddings.count) embeddings")
```

##### getAllBySourceType(_:model:limit:)

List all embeddings by source type and model.

```swift
/// List embeddings by source type
/// - Parameters:
///   - sourceType: Source type filter
///   - model: Model identifier
///   - limit: Maximum number of records to return
/// - Returns: Array of embedding records
public func getAllBySourceType(
    _ sourceType: SourceType,
    model: String,
    limit: Int = 1000
) async throws -> [EmbeddingRecord]
```

**Example:**
```swift
// Get all triple embeddings
let tripleEmbeddings = try await store.getAllBySourceType(
    .triple,
    model: "mlx-embed-1024-v1",
    limit: 500
)

// Get all entity embeddings
let entityEmbeddings = try await store.getAllBySourceType(
    .entity,
    model: "mlx-embed-1024-v1"
)
```

##### countByModel(_:)

Count embeddings for a model.

```swift
/// Count embeddings for a model
/// - Parameter model: Model identifier
/// - Returns: Number of embeddings
public func countByModel(_ model: String) async throws -> Int
```

**Example:**
```swift
let count = try await store.countByModel("mlx-embed-1024-v1")
print("Total embeddings: \(count)")
```

#### Cache Management

##### clearCache()

Clear the LRU cache.

```swift
/// Clear the entire cache
public func clearCache() async
```

**Example:**
```swift
await store.clearCache()
```

##### getCacheStats()

Get cache performance statistics.

```swift
/// Get cache statistics
/// - Returns: Cache statistics including hit rate and size
public func getCacheStats() async -> CacheStats
```

**Example:**
```swift
let stats = await store.getCacheStats()
print("Cache size: \(stats.currentSize) / \(stats.maxSize)")
print("Hit rate: \(stats.hitRate * 100)%")
print("Total hits: \(stats.hits)")
print("Total misses: \(stats.misses)")
```

---

### ModelManager

The `ModelManager` actor manages embedding model metadata, versioning, and lifecycle.

#### Initialization

```swift
public actor ModelManager {
    /// Initialize the model manager
    /// - Parameters:
    ///   - database: FoundationDB database instance
    ///   - rootPrefix: Namespace prefix
    public init(
        database: any DatabaseProtocol,
        rootPrefix: String
    ) async throws
}
```

**Example:**
```swift
let modelManager = try await ModelManager(
    database: database,
    rootPrefix: "myapp"
)
```

#### Model Registration

##### registerModel(_:)

Register a new embedding model.

```swift
/// Register an embedding model
/// - Parameter metadata: Model metadata
/// - Throws: EmbeddingError.invalidModel if validation fails
public func registerModel(_ metadata: EmbeddingModelMetadata) async throws
```

**Example:**
```swift
let model = EmbeddingModelMetadata(
    name: "mlx-embed-1024-v1",
    version: "1.0.0",
    dimension: 1024,
    provider: "Apple MLX",
    modelType: "Sentence-BERT",
    description: "Local MLX-based sentence embedding model",
    normalized: true,
    maxInputLength: 512,
    supportedLanguages: ["en", "ja"]
)

try await modelManager.registerModel(model)
```

##### getModel(name:)

Retrieve model metadata by name.

```swift
/// Get model metadata
/// - Parameter name: Model identifier
/// - Returns: Model metadata, or nil if not found
public func getModel(name: String) async throws -> EmbeddingModelMetadata?
```

**Example:**
```swift
if let model = try await modelManager.getModel(name: "mlx-embed-1024-v1") {
    print("Dimension: \(model.dimension)")
    print("Provider: \(model.provider)")
    print("Normalized: \(model.normalized)")
}
```

##### getAllModels()

List all registered models.

```swift
/// List all registered models
/// - Returns: Array of model metadata
public func getAllModels() async throws -> [EmbeddingModelMetadata]
```

**Example:**
```swift
let models = try await modelManager.getAllModels()
for model in models {
    print("\(model.name) (v\(model.version)): \(model.dimension) dims")
}
```

##### updateModel(_:)

Update model metadata (non-breaking changes only).

```swift
/// Update model metadata
/// - Parameter metadata: Updated model metadata
/// - Throws: EmbeddingError.modelNotFound if model doesn't exist
public func updateModel(_ metadata: EmbeddingModelMetadata) async throws
```

**Example:**
```swift
var model = try await modelManager.getModel(name: "mlx-embed-1024-v1")!
model.description = "Updated description"
try await modelManager.updateModel(model)
```

##### deleteModel(name:)

Delete a model and all its embeddings.

```swift
/// Delete a model
/// - Parameter name: Model identifier
/// - Warning: This deletes all embeddings for this model
public func deleteModel(name: String) async throws
```

**Example:**
```swift
try await modelManager.deleteModel(name: "old-model-v1")
```

#### Model Lifecycle

##### deprecateModel(name:reason:replacementModel:)

Mark a model as deprecated.

```swift
/// Deprecate a model
/// - Parameters:
///   - name: Model identifier
///   - reason: Deprecation reason
///   - replacementModel: Optional replacement model name
public func deprecateModel(
    name: String,
    reason: String,
    replacementModel: String? = nil
) async throws
```

**Example:**
```swift
try await modelManager.deprecateModel(
    name: "mlx-embed-768-v1",
    reason: "Superseded by higher dimension model",
    replacementModel: "mlx-embed-1024-v1"
)
```

##### validateModelCompatibility(_:_:)

Check if two models are compatible for comparison.

```swift
/// Validate model compatibility
/// - Parameters:
///   - modelA: First model name
///   - modelB: Second model name
/// - Returns: true if models are compatible (same dimension, provider, etc.)
public func validateModelCompatibility(
    _ modelA: String,
    _ modelB: String
) async throws -> Bool
```

**Example:**
```swift
let compatible = try await modelManager.validateModelCompatibility(
    "mlx-embed-1024-v1",
    "mlx-embed-1024-v2"
)

if compatible {
    print("Models can be compared")
}
```

#### Model Statistics

##### getModelStats(name:)

Get usage statistics for a model.

```swift
/// Get model statistics
/// - Parameter name: Model identifier
/// - Returns: Model statistics
public func getModelStats(name: String) async throws -> ModelStats
```

**Example:**
```swift
let stats = try await modelManager.getModelStats(name: "mlx-embed-1024-v1")
print("Vector count: \(stats.vectorCount)")
print("Avg generation time: \(stats.avgGenerationTime ?? 0)ms")
print("Last used: \(stats.lastUsed ?? Date.distantPast)")
```

---

### SearchEngine

The `SearchEngine` actor implements similarity search algorithms with support for multiple metrics and filtering.

#### Initialization

```swift
public actor SearchEngine {
    /// Initialize the search engine
    /// - Parameter store: Embedding store instance
    public init(store: EmbeddingStore)
}
```

**Example:**
```swift
let searchEngine = SearchEngine(store: store)
```

#### Single Query Search

##### search(queryVector:topK:model:metric:filter:)

Perform similarity search for a single query vector.

```swift
/// Search for similar embeddings
/// - Parameters:
///   - queryVector: Query vector (must match model dimension)
///   - topK: Number of results to return
///   - model: Model identifier
///   - metric: Similarity metric (default: .cosine)
///   - filter: Optional filter criteria
/// - Returns: Array of search results, sorted by similarity (descending)
public func search(
    queryVector: [Float],
    topK: Int,
    model: String,
    metric: SimilarityMetric = .cosine,
    filter: SearchFilter? = nil
) async throws -> [SearchResult]
```

**Example:**
```swift
let queryVector = try await generator.generate(text: "software engineer")

let results = try await searchEngine.search(
    queryVector: queryVector,
    topK: 10,
    model: "mlx-embed-1024-v1",
    metric: .cosine
)

for result in results {
    print("ID: \(result.id), Score: \(result.score)")
}
```

**Performance:**
- 10K corpus: 50-100ms (p50), 200-300ms (p99)
- 100K corpus: 200-500ms (p50), 1-2s (p99)
- 1M+ corpus (with ANN): 10-50ms (p50), 100-200ms (p99)

#### Batch Search

##### batchSearch(queryVectors:topK:model:metric:)

Perform search for multiple query vectors in parallel.

```swift
/// Batch search for multiple queries
/// - Parameters:
///   - queryVectors: Array of query vectors
///   - topK: Number of results per query
///   - model: Model identifier
///   - metric: Similarity metric
/// - Returns: Array of result arrays (one per query)
public func batchSearch(
    queryVectors: [[Float]],
    topK: Int,
    model: String,
    metric: SimilarityMetric = .cosine
) async throws -> [[SearchResult]]
```

**Example:**
```swift
let queries = [
    try await generator.generate(text: "software engineer"),
    try await generator.generate(text: "data scientist"),
    try await generator.generate(text: "product manager")
]

let batchResults = try await searchEngine.batchSearch(
    queryVectors: queries,
    topK: 5,
    model: "mlx-embed-1024-v1"
)

for (i, results) in batchResults.enumerated() {
    print("Query \(i): \(results.count) results")
}
```

#### Filtered Search

##### searchWithFilter(queryVector:topK:model:filter:)

Search with advanced filtering criteria.

```swift
/// Search with filters
/// - Parameters:
///   - queryVector: Query vector
///   - topK: Number of results
///   - model: Model identifier
///   - filter: Filter criteria
/// - Returns: Filtered and ranked results
public func searchWithFilter(
    queryVector: [Float],
    topK: Int,
    model: String,
    filter: SearchFilter
) async throws -> [SearchResult]
```

**Example:**
```swift
let filter = SearchFilter(
    sourceTypes: [.triple, .entity],
    metadata: ["language": "en"],
    createdAfter: Date().addingTimeInterval(-86400), // Last 24 hours
    createdBefore: nil
)

let results = try await searchEngine.searchWithFilter(
    queryVector: queryVector,
    topK: 10,
    model: "mlx-embed-1024-v1",
    filter: filter
)
```

#### Similarity Metrics

##### getSimilarityScore(vector1:vector2:metric:)

Compute similarity score between two vectors.

```swift
/// Compute similarity between two vectors
/// - Parameters:
///   - vector1: First vector
///   - vector2: Second vector
///   - metric: Similarity metric
/// - Returns: Similarity score
public func getSimilarityScore(
    vector1: [Float],
    vector2: [Float],
    metric: SimilarityMetric
) -> Float
```

**Example:**
```swift
let score = searchEngine.getSimilarityScore(
    vector1: embedding1.vector,
    vector2: embedding2.vector,
    metric: .cosine
)

print("Cosine similarity: \(score)")
```

---

### EmbeddingGenerator

The `EmbeddingGenerator` protocol defines the interface for embedding model integration.

```swift
public protocol EmbeddingGenerator: Sendable {
    /// Generate embedding from text
    func generate(text: String) async throws -> [Float]

    /// Generate embedding from triple
    func generate(triple: Triple) async throws -> [Float]

    /// Generate embedding from entity
    func generate(entity: OntologyClass) async throws -> [Float]

    /// Batch generate embeddings from texts
    func generateBatch(texts: [String]) async throws -> [[Float]]

    /// Batch generate embeddings from triples
    func generateBatch(triples: [Triple]) async throws -> [[Float]]

    /// Model metadata
    var modelMetadata: EmbeddingModelMetadata { get }
}
```

#### Implementation Examples

##### MLX Local Model

```swift
public actor MLXEmbeddingGenerator: EmbeddingGenerator {
    private let model: MLXModel
    private let tokenizer: Tokenizer

    public var modelMetadata: EmbeddingModelMetadata {
        EmbeddingModelMetadata(
            name: "mlx-embed-1024-v1",
            version: "1.0.0",
            dimension: 1024,
            provider: "Apple MLX",
            modelType: "Sentence-BERT",
            normalized: true,
            maxInputLength: 512
        )
    }

    public func generate(text: String) async throws -> [Float] {
        let tokens = tokenizer.encode(text)
        let output = try await model.forward(tokens)
        return normalize(output.pooled)
    }

    public func generateBatch(texts: [String]) async throws -> [[Float]] {
        let tokenBatches = texts.map { tokenizer.encode($0) }
        let outputs = try await model.forwardBatch(tokenBatches)
        return outputs.map { normalize($0.pooled) }
    }

    private func normalize(_ vector: [Float]) -> [Float] {
        let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        return vector.map { $0 / norm }
    }
}
```

##### OpenAI API

```swift
public actor OpenAIEmbeddingGenerator: EmbeddingGenerator {
    private let apiKey: String
    private let client: OpenAIClient

    public var modelMetadata: EmbeddingModelMetadata {
        EmbeddingModelMetadata(
            name: "text-embedding-3-large",
            version: "3.0.0",
            dimension: 1536,
            provider: "OpenAI",
            modelType: "GPT-based",
            normalized: true,
            maxInputLength: 8191
        )
    }

    public func generate(text: String) async throws -> [Float] {
        let response = try await client.createEmbedding(
            model: "text-embedding-3-large",
            input: text
        )
        return response.data[0].embedding
    }

    public func generateBatch(texts: [String]) async throws -> [[Float]] {
        let response = try await client.createEmbedding(
            model: "text-embedding-3-large",
            input: texts
        )
        return response.data.map { $0.embedding }
    }
}
```

##### Triple Embedding Strategy

```swift
extension EmbeddingGenerator {
    public func generate(triple: Triple) async throws -> [Float] {
        // Simple concatenation strategy
        let text = "\(triple.subject.value) \(triple.predicate.value) \(triple.object.value)"
        return try await generate(text: text)
    }

    public func generateContextual(triple: Triple, tripleStore: TripleStore) async throws -> [Float] {
        // Contextual strategy with 1-hop neighbors
        let neighbors = try await tripleStore.query(subject: triple.subject, predicate: nil, object: nil)
        let context = neighbors.map { "\($0.predicate) \($0.object)" }.joined(separator: ", ")

        let text = "\(triple.subject) \(triple.predicate) \(triple.object). Context: \(context)"
        return try await generate(text: text)
    }
}
```

---

## Method Signatures

### Complete API Reference

```swift
// MARK: - EmbeddingStore

public actor EmbeddingStore {
    public init(database: any DatabaseProtocol, rootPrefix: String, cacheSize: Int = 10_000) async throws

    // CRUD
    public func save(_ record: EmbeddingRecord) async throws
    public func get(id: String, model: String) async throws -> EmbeddingRecord?
    public func delete(id: String, model: String) async throws
    public func exists(id: String, model: String) async throws -> Bool

    // Batch
    public func saveBatch(_ records: [EmbeddingRecord]) async throws
    public func getBatch(ids: [String], model: String) async throws -> [String: EmbeddingRecord]
    public func deleteBatch(ids: [String], model: String) async throws

    // Query
    public func getAllByModel(_ model: String, limit: Int = 1000) async throws -> [EmbeddingRecord]
    public func getAllBySourceType(_ sourceType: SourceType, model: String, limit: Int = 1000) async throws -> [EmbeddingRecord]
    public func countByModel(_ model: String) async throws -> Int

    // Cache
    public func clearCache() async
    public func getCacheStats() async -> CacheStats
}

// MARK: - ModelManager

public actor ModelManager {
    public init(database: any DatabaseProtocol, rootPrefix: String) async throws

    // Registration
    public func registerModel(_ metadata: EmbeddingModelMetadata) async throws
    public func getModel(name: String) async throws -> EmbeddingModelMetadata?
    public func getAllModels() async throws -> [EmbeddingModelMetadata]
    public func updateModel(_ metadata: EmbeddingModelMetadata) async throws
    public func deleteModel(name: String) async throws

    // Lifecycle
    public func deprecateModel(name: String, reason: String, replacementModel: String? = nil) async throws
    public func validateModelCompatibility(_ modelA: String, _ modelB: String) async throws -> Bool

    // Statistics
    public func getModelStats(name: String) async throws -> ModelStats
}

// MARK: - SearchEngine

public actor SearchEngine {
    public init(store: EmbeddingStore)

    // Search
    public func search(
        queryVector: [Float],
        topK: Int,
        model: String,
        metric: SimilarityMetric = .cosine,
        filter: SearchFilter? = nil
    ) async throws -> [SearchResult]

    public func batchSearch(
        queryVectors: [[Float]],
        topK: Int,
        model: String,
        metric: SimilarityMetric = .cosine
    ) async throws -> [[SearchResult]]

    public func searchWithFilter(
        queryVector: [Float],
        topK: Int,
        model: String,
        filter: SearchFilter
    ) async throws -> [SearchResult]

    // Metrics
    public func getSimilarityScore(vector1: [Float], vector2: [Float], metric: SimilarityMetric) -> Float
}

// MARK: - EmbeddingGenerator

public protocol EmbeddingGenerator: Sendable {
    func generate(text: String) async throws -> [Float]
    func generate(triple: Triple) async throws -> [Float]
    func generate(entity: OntologyClass) async throws -> [Float]
    func generateBatch(texts: [String]) async throws -> [[Float]]
    func generateBatch(triples: [Triple]) async throws -> [[Float]]

    var modelMetadata: EmbeddingModelMetadata { get }
}
```

---

## Usage Examples

### Example 1: Basic Setup

```swift
import FoundationDB
import EmbeddingLayer

// 1. Initialize FoundationDB
try await FDBClient.initialize()
let database = try FDBClient.openDatabase()

// 2. Create components
let store = try await EmbeddingStore(
    database: database,
    rootPrefix: "myapp"
)

let modelManager = try await ModelManager(
    database: database,
    rootPrefix: "myapp"
)

let searchEngine = SearchEngine(store: store)

// 3. Register model
let model = EmbeddingModelMetadata(
    name: "mlx-embed-1024-v1",
    version: "1.0.0",
    dimension: 1024,
    provider: "Apple MLX",
    modelType: "Sentence-BERT",
    normalized: true
)
try await modelManager.registerModel(model)

// 4. Create generator
let generator = MLXEmbeddingGenerator(modelPath: "/path/to/model")
```

### Example 2: Single Embedding Workflow

```swift
// Generate embedding from text
let text = "Alice knows Bob"
let vector = try await generator.generate(text: text)

// Create record
let record = EmbeddingRecord(
    id: "triple:12345",
    vector: vector,
    model: "mlx-embed-1024-v1",
    dimension: 1024,
    sourceType: .triple,
    metadata: ["subject": "Alice", "predicate": "knows", "object": "Bob"]
)

// Save
try await store.save(record)

// Retrieve
if let retrieved = try await store.get(id: "triple:12345", model: "mlx-embed-1024-v1") {
    print("Vector dimension: \(retrieved.dimension)")
}

// Delete
try await store.delete(id: "triple:12345", model: "mlx-embed-1024-v1")
```

### Example 3: Batch Processing

```swift
// Generate batch from triples
let triples = try await tripleStore.query(subject: nil, predicate: nil, object: nil)
let tripleTexts = triples.map { "\($0.subject) \($0.predicate) \($0.object)" }

// Generate embeddings in batch
let vectors = try await generator.generateBatch(texts: tripleTexts)

// Create records
let records = zip(triples, vectors).map { (triple, vector) in
    EmbeddingRecord(
        id: "triple:\(triple.id)",
        vector: vector,
        model: "mlx-embed-1024-v1",
        dimension: 1024,
        sourceType: .triple,
        metadata: ["tripleId": "\(triple.id)"]
    )
}

// Save batch
try await store.saveBatch(records)

print("Saved \(records.count) embeddings")
```

### Example 4: Similarity Search

```swift
// Generate query embedding
let queryText = "person relationships"
let queryVector = try await generator.generate(text: queryText)

// Search for similar embeddings
let results = try await searchEngine.search(
    queryVector: queryVector,
    topK: 10,
    model: "mlx-embed-1024-v1",
    metric: .cosine
)

// Process results
for (rank, result) in results.enumerated() {
    print("\(rank + 1). ID: \(result.id)")
    print("   Score: \(result.score)")
    print("   Metadata: \(result.metadata ?? [:])")
}
```

### Example 5: Filtered Search

```swift
// Create filter for recent triple embeddings
let filter = SearchFilter(
    sourceTypes: [.triple],
    metadata: nil,
    createdAfter: Date().addingTimeInterval(-86400), // Last 24 hours
    createdBefore: nil
)

// Search with filter
let results = try await searchEngine.searchWithFilter(
    queryVector: queryVector,
    topK: 10,
    model: "mlx-embed-1024-v1",
    filter: filter
)

print("Found \(results.count) recent triple embeddings")
```

### Example 6: Batch Search

```swift
// Multiple queries
let queries = [
    "software engineer at tech company",
    "data scientist in research lab",
    "product manager for mobile apps"
]

// Generate query vectors
let queryVectors = try await generator.generateBatch(texts: queries)

// Batch search
let batchResults = try await searchEngine.batchSearch(
    queryVectors: queryVectors,
    topK: 5,
    model: "mlx-embed-1024-v1",
    metric: .cosine
)

// Process all results
for (i, results) in batchResults.enumerated() {
    print("Query: \(queries[i])")
    for result in results {
        print("  - \(result.id): \(result.score)")
    }
}
```

### Example 7: Model Management

```swift
// Register multiple models
let models = [
    EmbeddingModelMetadata(
        name: "mlx-embed-384-v1",
        version: "1.0.0",
        dimension: 384,
        provider: "Apple MLX",
        modelType: "MiniLM"
    ),
    EmbeddingModelMetadata(
        name: "mlx-embed-1024-v1",
        version: "1.0.0",
        dimension: 1024,
        provider: "Apple MLX",
        modelType: "Sentence-BERT"
    )
]

for model in models {
    try await modelManager.registerModel(model)
}

// List models
let registeredModels = try await modelManager.getAllModels()
for model in registeredModels {
    print("\(model.name): \(model.dimension) dims, \(model.provider)")
}

// Get model stats
let stats = try await modelManager.getModelStats(name: "mlx-embed-1024-v1")
print("Vector count: \(stats.vectorCount)")
```

### Example 8: Model Migration

```swift
// Migrate from old model to new model
let oldModel = "mlx-embed-768-v1"
let newModel = "mlx-embed-1024-v2"

// Get all embeddings from old model
let oldEmbeddings = try await store.getAllByModel(oldModel, limit: Int.max)

// Re-generate with new model
let newGenerator = MLXEmbeddingGenerator(modelPath: "/path/to/new/model")

for batch in oldEmbeddings.chunked(into: 100) {
    // Fetch source texts (from metadata or original source)
    let sourceTexts = batch.compactMap { $0.metadata?["text"] }

    // Generate new embeddings
    let newVectors = try await newGenerator.generateBatch(texts: sourceTexts)

    // Create new records
    let newRecords = zip(batch, newVectors).map { (old, vector) in
        EmbeddingRecord(
            id: old.id,
            vector: vector,
            model: newModel,
            dimension: vector.count,
            sourceType: old.sourceType,
            metadata: old.metadata
        )
    }

    // Save new embeddings
    try await store.saveBatch(newRecords)
}

// Deprecate old model
try await modelManager.deprecateModel(
    name: oldModel,
    reason: "Migrated to higher dimension model",
    replacementModel: newModel
)
```

### Example 9: Triple Embeddings Integration

```swift
import TripleLayer
import EmbeddingLayer

// Setup
let tripleStore = try await TripleStore(database: database, rootPrefix: "myapp")
let embeddingStore = try await EmbeddingStore(database: database, rootPrefix: "myapp")

// Insert triples and generate embeddings
let triples = [
    Triple(
        subject: .uri("http://example.org/Alice"),
        predicate: .uri("http://xmlns.com/foaf/0.1/knows"),
        object: .uri("http://example.org/Bob")
    ),
    Triple(
        subject: .uri("http://example.org/Alice"),
        predicate: .uri("http://xmlns.com/foaf/0.1/name"),
        object: .text("Alice Smith")
    )
]

// Save triples
try await tripleStore.insertBatch(triples)

// Generate embeddings
let texts = triples.map { "\($0.subject.value) \($0.predicate.value) \($0.object.value)" }
let vectors = try await generator.generateBatch(texts: texts)

let embeddings = zip(triples, vectors).enumerated().map { (index, pair) in
    let (triple, vector) = pair
    return EmbeddingRecord(
        id: "triple:\(index)",
        vector: vector,
        model: "mlx-embed-1024-v1",
        dimension: 1024,
        sourceType: .triple,
        metadata: [
            "subject": triple.subject.value,
            "predicate": triple.predicate.value,
            "object": triple.object.value
        ]
    )
}

try await embeddingStore.saveBatch(embeddings)
```

### Example 10: Ontology Class Embeddings

```swift
import OntologyLayer
import EmbeddingLayer

// Create class embeddings
let personClass = OntologyClass(
    name: "Person",
    description: "A human being with identity and relationships",
    properties: ["name", "age", "knows", "worksAt"]
)

// Generate rich description
let classDescription = """
Class: \(personClass.name)
Description: \(personClass.description ?? "")
Properties: \(personClass.properties.joined(separator: ", "))
"""

let classVector = try await generator.generate(text: classDescription)

let classEmbedding = EmbeddingRecord(
    id: "class:Person",
    vector: classVector,
    model: "mlx-embed-1024-v1",
    dimension: 1024,
    sourceType: .entity,
    metadata: [
        "className": personClass.name,
        "propertyCount": "\(personClass.properties.count)"
    ]
)

try await embeddingStore.save(classEmbedding)
```

### Example 11: Hybrid Search (Structured + Semantic)

```swift
// Combine triple query with semantic search

// 1. Structured query: Find all triples about Alice
let aliceTriples = try await tripleStore.query(
    subject: .uri("http://example.org/Alice"),
    predicate: nil,
    object: nil
)

// 2. Semantic query: Find similar concepts
let queryText = "person who works at a company"
let queryVector = try await generator.generate(text: queryText)

let semanticResults = try await searchEngine.search(
    queryVector: queryVector,
    topK: 50,
    model: "mlx-embed-1024-v1"
)

// 3. Merge results: semantic results that match Alice's triples
let aliceTripleIds = Set(aliceTriples.map { "triple:\($0.id)" })
let hybridResults = semanticResults.filter { aliceTripleIds.contains($0.id) }

print("Hybrid search found \(hybridResults.count) results")
```

### Example 12: Caching Strategy

```swift
// Pre-warm cache with frequently accessed embeddings
let frequentIds = ["triple:1", "triple:2", "triple:3", "class:Person", "class:Company"]

let cachedEmbeddings = try await store.getBatch(
    ids: frequentIds,
    model: "mlx-embed-1024-v1"
)

print("Pre-warmed cache with \(cachedEmbeddings.count) embeddings")

// Check cache stats
let stats = await store.getCacheStats()
print("Cache hit rate: \(stats.hitRate * 100)%")

// Clear cache when switching models
await store.clearCache()
```

### Example 13: Error Handling

```swift
do {
    let record = EmbeddingRecord(
        id: "test:1",
        vector: vector,
        model: "unknown-model",
        dimension: 1024,
        sourceType: .text
    )

    try await store.save(record)

} catch EmbeddingError.modelNotFound(let name) {
    print("Model '\(name)' not found, registering...")

    let model = EmbeddingModelMetadata(
        name: name,
        version: "1.0.0",
        dimension: 1024,
        provider: "Custom",
        modelType: "Unknown"
    )

    try await modelManager.registerModel(model)
    try await store.save(record)

} catch EmbeddingError.dimensionMismatch(let expected, let actual) {
    print("Fatal: dimension mismatch \(expected) != \(actual)")
    throw EmbeddingError.dimensionMismatch(expected: expected, actual: actual)

} catch EmbeddingError.invalidVector(let reason) {
    print("Invalid vector: \(reason)")
    // Log and skip

} catch {
    print("Unexpected error: \(error)")
    throw error
}
```

### Example 14: Model Version Comparison

```swift
// Compare two model versions
let modelV1 = try await modelManager.getModel(name: "mlx-embed-1024-v1")!
let modelV2 = try await modelManager.getModel(name: "mlx-embed-1024-v2")!

// Check compatibility
let compatible = try await modelManager.validateModelCompatibility(
    modelV1.name,
    modelV2.name
)

if compatible {
    print("Models are compatible for comparison")

    // Generate same embedding with both models
    let text = "software engineer"
    let vectorV1 = try await generatorV1.generate(text: text)
    let vectorV2 = try await generatorV2.generate(text: text)

    // Compare vectors
    let similarity = searchEngine.getSimilarityScore(
        vector1: vectorV1,
        vector2: vectorV2,
        metric: .cosine
    )

    print("Cross-version similarity: \(similarity)")
}
```

### Example 15: Background Regeneration Job

```swift
actor RegenerationJobManager {
    private let store: EmbeddingStore
    private let generator: EmbeddingGenerator

    func regenerateModel(
        sourceModel: String,
        targetModel: String,
        batchSize: Int = 100
    ) async throws {
        // Get all embeddings for source model
        let embeddings = try await store.getAllByModel(sourceModel, limit: Int.max)

        print("Regenerating \(embeddings.count) embeddings...")

        var processed = 0

        for batch in embeddings.chunked(into: batchSize) {
            // Fetch source data from metadata
            let texts = batch.compactMap { $0.metadata?["text"] }

            // Generate new vectors
            let newVectors = try await generator.generateBatch(texts: texts)

            // Create new records
            let newRecords = zip(batch, newVectors).map { (old, vector) in
                EmbeddingRecord(
                    id: old.id,
                    vector: vector,
                    model: targetModel,
                    dimension: vector.count,
                    sourceType: old.sourceType,
                    metadata: old.metadata
                )
            }

            // Save batch
            try await store.saveBatch(newRecords)

            processed += batch.count
            print("Progress: \(processed)/\(embeddings.count)")
        }

        print("Regeneration complete!")
    }
}

// Usage
let jobManager = RegenerationJobManager(store: store, generator: newGenerator)

Task {
    try await jobManager.regenerateModel(
        sourceModel: "mlx-embed-768-v1",
        targetModel: "mlx-embed-1024-v2"
    )
}
```

---

## Error Handling

### Error Types

```swift
public enum EmbeddingError: Error, Sendable {
    case modelNotFound(String)
    case vectorNotFound(String)
    case dimensionMismatch(expected: Int, actual: Int)
    case modelIncompatible(String, String)
    case generationFailed(String)
    case searchFailed(String)
    case cacheError(String)
    case storageError(String)
    case encodingError(String)
    case validationError(String)
    case migrationError(String)
}
```

### Error Handling Strategies

| Error | Recovery Strategy | Example |
|-------|------------------|---------|
| `modelNotFound` | Register model first | Auto-register on save |
| `vectorNotFound` | Generate on-demand | Create if missing |
| `dimensionMismatch` | Fail fast | Validate before save |
| `generationFailed` | Retry with backoff | 3 retries max |
| `searchFailed` | Degrade gracefully | Fall back to smaller corpus |
| `cacheError` | Bypass cache | Read from FDB |
| `storageError` | Retry transaction | FDB auto-retry |

### Complete Error Handling Example

```swift
func saveWithRetry(
    _ record: EmbeddingRecord,
    maxRetries: Int = 3
) async throws {
    var attempt = 0

    while attempt < maxRetries {
        do {
            try await store.save(record)
            return // Success

        } catch EmbeddingError.modelNotFound(let name) {
            // One-time recovery: register model
            if attempt == 0 {
                print("Registering missing model: \(name)")
                try await modelManager.registerModel(record.modelMetadata)
                attempt += 1
                continue
            }
            throw EmbeddingError.modelNotFound(name)

        } catch EmbeddingError.storageError(let message) {
            // Retry with exponential backoff
            attempt += 1
            if attempt < maxRetries {
                let delay = TimeInterval(pow(2.0, Double(attempt)))
                print("Storage error, retrying in \(delay)s: \(message)")
                try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            } else {
                throw EmbeddingError.storageError("Max retries exceeded: \(message)")
            }

        } catch let error as EmbeddingError {
            // Non-retryable errors
            print("Non-retryable error: \(error)")
            throw error

        } catch {
            print("Unexpected error: \(error)")
            throw error
        }
    }
}
```

---

## Performance Considerations

### Latency Targets

| Operation | Target (p50) | Target (p99) | Optimization |
|-----------|--------------|--------------|--------------|
| **Save Embedding** | 5-10ms | 20-30ms | Batch operations |
| **Get Embedding (cached)** | <1ms | <5ms | LRU cache |
| **Get Embedding (uncached)** | 5-10ms | 20-30ms | Pre-fetch |
| **Save Batch (100)** | 50-100ms | 200-300ms | Transaction batching |
| **Get Batch (100)** | 20-50ms | 100-150ms | Parallel reads |
| **Search (10K corpus)** | 50-100ms | 200-300ms | SIMD optimization |
| **Search (100K corpus)** | 200-500ms | 1-2s | Consider ANN |
| **Search (1M+ with ANN)** | 10-50ms | 100-200ms | HNSW/IVF index |

### Throughput Expectations

| Operation | Throughput | Concurrency | Notes |
|-----------|------------|-------------|-------|
| Save Embedding | 1,000-5,000 ops/sec | 10-50 actors | Limited by FDB writes |
| Get Embedding (cached) | 100,000+ ops/sec | Memory bound | In-process cache |
| Batch Save (100 vectors) | 500-1,000 batches/sec | 50,000-100,000 vectors/sec | Transaction batching |
| Similarity Search (brute-force) | 100-500 queries/sec | CPU bound | SIMD acceleration |
| Similarity Search (ANN) | 1,000-5,000 queries/sec | Memory + CPU | Index-based |
| Similarity Search (VectorDB) | 5,000-10,000 queries/sec | Network bound | External service |

### Optimization Tips

#### 1. Batch Operations

```swift
// BAD: Save individually (slow)
for record in records {
    try await store.save(record) // 1000 x 10ms = 10s
}

// GOOD: Save in batch (fast)
try await store.saveBatch(records) // ~100ms
```

**Speedup**: 50-100x for large batches

#### 2. Caching Strategy

```swift
// Configure cache size based on memory
let store = try await EmbeddingStore(
    database: database,
    rootPrefix: "myapp",
    cacheSize: 50_000 // ~200MB for 1024-dim
)

// Pre-warm cache with frequently accessed vectors
let hotIds = getFrequentlyAccessedIDs()
let _ = try await store.getBatch(ids: hotIds, model: modelName)
```

**Speedup**: 10-100x for cache hits

#### 3. Search Optimization

```swift
// For small corpus (<10K): Brute-force is fastest
let results = try await searchEngine.search(
    queryVector: query,
    topK: 10,
    model: "mlx-embed-1024-v1",
    metric: .cosine // SIMD-optimized
)

// For large corpus (>100K): Use filtering to reduce corpus
let filter = SearchFilter(
    sourceTypes: [.triple], // Filter by type
    createdAfter: recentDate // Recent only
)

let results = try await searchEngine.searchWithFilter(
    queryVector: query,
    topK: 10,
    model: "mlx-embed-1024-v1",
    filter: filter
)
```

#### 4. Parallel Batch Processing

```swift
// Process multiple batches in parallel
await withTaskGroup(of: Void.self) { group in
    for batch in triples.chunked(into: 100) {
        group.addTask {
            let texts = batch.map { tripleToText($0) }
            let vectors = try await generator.generateBatch(texts: texts)
            let records = createRecords(batch, vectors)
            try await store.saveBatch(records)
        }
    }
}
```

**Speedup**: Linear with CPU cores (up to 8-16x)

#### 5. Model Selection

```swift
// Small model for speed
let fastModel = "mlx-embed-384-v1" // 384 dims, 2-3x faster

// Large model for accuracy
let accurateModel = "mlx-embed-1536-v1" // 1536 dims, higher quality

// Choose based on use case
let model = needsSpeed ? fastModel : accurateModel
```

**Trade-off**: 2-3x speed vs 5-10% accuracy

---

## Best Practices

### 1. When to Use Batch Operations

**Use batch operations for:**
- Initial bulk import (>100 embeddings)
- Periodic regeneration jobs
- Model migration
- Background processing

**Use single operations for:**
- Real-time user requests
- Low-latency requirements
- Incremental updates

**Example:**
```swift
// Batch: Initial import
let allTriples = try await tripleStore.query(subject: nil, predicate: nil, object: nil)
try await batchGenerateEmbeddings(triples: allTriples)

// Single: Real-time insert
let newTriple = Triple(...)
try await tripleStore.insert(newTriple)
let embedding = try await generateEmbedding(triple: newTriple)
try await store.save(embedding)
```

### 2. Caching Strategies

**Pre-warming:**
```swift
// Warm cache at startup
let frequentIds = ["class:Person", "class:Company", ...]
let _ = try await store.getBatch(ids: frequentIds, model: modelName)
```

**Cache invalidation:**
```swift
// Clear cache after model update
await store.clearCache()

// Monitor cache performance
let stats = await store.getCacheStats()
if stats.hitRate < 0.7 { // <70% hit rate
    // Increase cache size or adjust access patterns
}
```

### 3. Transaction Patterns

**Single transaction (fast):**
```swift
try await store.save(record)
```

**Batch transaction (faster for bulk):**
```swift
try await store.saveBatch(records) // Auto-batched into chunks of 1000
```

**Manual chunking (for very large batches):**
```swift
for chunk in records.chunked(into: 1000) {
    try await store.saveBatch(chunk)
}
```

### 4. Search Optimization

**Filter before search:**
```swift
// BAD: Search all, then filter
let allResults = try await searchEngine.search(query, topK: 1000, model: model)
let filtered = allResults.filter { $0.sourceType == .triple }

// GOOD: Filter during search
let filter = SearchFilter(sourceTypes: [.triple], ...)
let results = try await searchEngine.searchWithFilter(query, topK: 10, model: model, filter: filter)
```

**Choose appropriate topK:**
```swift
// For UI display: topK = 10-20
let displayResults = try await searchEngine.search(query, topK: 10, model: model)

// For re-ranking: topK = 50-100
let candidates = try await searchEngine.search(query, topK: 100, model: model)
let reranked = rerankResults(candidates)
```

### 5. Model Management

**Version naming convention:**
```swift
// GOOD: Include version in name
"mlx-embed-1024-v1"
"mlx-embed-1024-v2"

// BAD: Generic names
"mlx-embed"
"my-model"
```

**Deprecation workflow:**
```swift
// 1. Register new model
try await modelManager.registerModel(newModel)

// 2. Migrate embeddings
try await migrateEmbeddings(from: oldModel, to: newModel)

// 3. Deprecate old model
try await modelManager.deprecateModel(
    name: oldModel,
    reason: "Superseded by v2",
    replacementModel: newModel
)
```

### 6. Error Handling Best Practices

**Graceful degradation:**
```swift
do {
    return try await searchEngine.search(query, topK: 10, model: model)
} catch {
    // Fall back to smaller corpus or cached results
    print("Search failed, using fallback")
    return try await getFallbackResults(query)
}
```

**Retry with backoff:**
```swift
func saveWithRetry(_ record: EmbeddingRecord) async throws {
    var retries = 0
    while retries < 3 {
        do {
            return try await store.save(record)
        } catch EmbeddingError.storageError {
            retries += 1
            try await Task.sleep(nanoseconds: UInt64(pow(2.0, Double(retries)) * 1_000_000_000))
        }
    }
    throw EmbeddingError.storageError("Max retries exceeded")
}
```

---

## Integration Patterns

### Triple Layer Integration

**Automatic embedding generation on triple insert:**

```swift
actor TripleEmbeddingManager {
    private let tripleStore: TripleStore
    private let embeddingStore: EmbeddingStore
    private let generator: EmbeddingGenerator

    func insertTripleWithEmbedding(_ triple: Triple) async throws {
        // 1. Save triple
        try await tripleStore.insert(triple)

        // 2. Generate embedding
        let text = "\(triple.subject.value) \(triple.predicate.value) \(triple.object.value)"
        let vector = try await generator.generate(text: text)

        // 3. Save embedding
        let embedding = EmbeddingRecord(
            id: "triple:\(triple.id)",
            vector: vector,
            model: generator.modelMetadata.name,
            dimension: vector.count,
            sourceType: .triple,
            metadata: [
                "subject": triple.subject.value,
                "predicate": triple.predicate.value,
                "object": triple.object.value
            ]
        )

        try await embeddingStore.save(embedding)
    }

    func semanticTripleQuery(query: String, topK: Int = 10) async throws -> [Triple] {
        // 1. Generate query embedding
        let queryVector = try await generator.generate(text: query)

        // 2. Search embeddings
        let searchEngine = SearchEngine(store: embeddingStore)
        let results = try await searchEngine.search(
            queryVector: queryVector,
            topK: topK,
            model: generator.modelMetadata.name
        )

        // 3. Fetch corresponding triples
        var triples: [Triple] = []
        for result in results {
            // Extract triple ID from embedding ID
            if let tripleIdStr = result.id.components(separatedBy: ":").last,
               let tripleId = Int64(tripleIdStr) {
                // Fetch triple by ID
                if let triple = try await tripleStore.getByID(tripleId) {
                    triples.append(triple)
                }
            }
        }

        return triples
    }
}
```

### Ontology Layer Integration

**Entity embedding generation:**

```swift
actor OntologyEmbeddingManager {
    private let ontologyStore: OntologyStore
    private let embeddingStore: EmbeddingStore
    private let generator: EmbeddingGenerator

    func generateClassEmbedding(_ cls: OntologyClass) async throws {
        // Create rich text representation
        var parts: [String] = []
        parts.append("Class: \(cls.name)")

        if let description = cls.description {
            parts.append("Description: \(description)")
        }

        if let parent = cls.parent {
            parts.append("Inherits from: \(parent)")
        }

        if !cls.properties.isEmpty {
            parts.append("Properties: \(cls.properties.joined(separator: ", "))")
        }

        let text = parts.joined(separator: ". ")

        // Generate embedding
        let vector = try await generator.generate(text: text)

        // Save
        let embedding = EmbeddingRecord(
            id: "class:\(cls.name)",
            vector: vector,
            model: generator.modelMetadata.name,
            dimension: vector.count,
            sourceType: .entity,
            metadata: [
                "className": cls.name,
                "propertyCount": "\(cls.properties.count)"
            ]
        )

        try await embeddingStore.save(embedding)
    }

    func findSimilarClasses(_ className: String, topK: Int = 5) async throws -> [OntologyClass] {
        // Get class embedding
        guard let embedding = try await embeddingStore.get(
            id: "class:\(className)",
            model: generator.modelMetadata.name
        ) else {
            throw EmbeddingError.vectorNotFound("class:\(className)")
        }

        // Search for similar classes
        let searchEngine = SearchEngine(store: embeddingStore)
        let filter = SearchFilter(sourceTypes: [.entity], metadata: nil, createdAfter: nil, createdBefore: nil)

        let results = try await searchEngine.searchWithFilter(
            queryVector: embedding.vector,
            topK: topK + 1, // +1 to exclude self
            model: generator.modelMetadata.name,
            filter: filter
        )

        // Fetch classes
        var classes: [OntologyClass] = []
        for result in results {
            // Skip self
            if result.id == "class:\(className)" { continue }

            if let name = result.metadata?["className"],
               let cls = try await ontologyStore.getClass(name: name) {
                classes.append(cls)
            }
        }

        return classes
    }
}
```

### Knowledge Layer Integration

**Unified semantic + structured query:**

```swift
actor KnowledgeLayerAPI {
    private let tripleStore: TripleStore
    private let embeddingStore: EmbeddingStore
    private let generator: EmbeddingGenerator

    /// Hybrid query: semantic search with structured constraints
    func hybridQuery(
        semanticQuery: String,
        structuredFilter: (subject: Value?, predicate: Value?, object: Value?),
        topK: Int = 10
    ) async throws -> [Triple] {
        // 1. Structured query to get candidate triples
        let candidateTriples = try await tripleStore.query(
            subject: structuredFilter.subject,
            predicate: structuredFilter.predicate,
            object: structuredFilter.object
        )

        // 2. Generate embeddings for candidates (if not exist)
        var candidateEmbeddings: [String: EmbeddingRecord] = [:]
        for triple in candidateTriples {
            let embeddingID = "triple:\(triple.id)"
            if let embedding = try await embeddingStore.get(id: embeddingID, model: generator.modelMetadata.name) {
                candidateEmbeddings[embeddingID] = embedding
            } else {
                // Generate on-demand
                let text = "\(triple.subject.value) \(triple.predicate.value) \(triple.object.value)"
                let vector = try await generator.generate(text: text)
                let embedding = EmbeddingRecord(
                    id: embeddingID,
                    vector: vector,
                    model: generator.modelMetadata.name,
                    dimension: vector.count,
                    sourceType: .triple
                )
                try await embeddingStore.save(embedding)
                candidateEmbeddings[embeddingID] = embedding
            }
        }

        // 3. Semantic ranking
        let queryVector = try await generator.generate(text: semanticQuery)
        let searchEngine = SearchEngine(store: embeddingStore)

        // Compute scores for candidates
        var scoredTriples: [(Triple, Float)] = []
        for (embeddingID, embedding) in candidateEmbeddings {
            let score = searchEngine.getSimilarityScore(
                vector1: queryVector,
                vector2: embedding.vector,
                metric: .cosine
            )

            if let tripleIdStr = embeddingID.components(separatedBy: ":").last,
               let tripleId = Int64(tripleIdStr),
               let triple = candidateTriples.first(where: { $0.id == tripleId }) {
                scoredTriples.append((triple, score))
            }
        }

        // 4. Sort and return top K
        scoredTriples.sort { $0.1 > $1.1 }
        return scoredTriples.prefix(topK).map { $0.0 }
    }
}
```

---

## Advanced Features

### 1. Filtered Search

**Multi-criteria filtering:**

```swift
let filter = SearchFilter(
    sourceTypes: [.triple, .entity], // Only triples and entities
    metadata: ["language": "en"],     // English only
    createdAfter: Date().addingTimeInterval(-7 * 86400), // Last week
    createdBefore: nil
)

let results = try await searchEngine.searchWithFilter(
    queryVector: queryVector,
    topK: 20,
    model: "mlx-embed-1024-v1",
    filter: filter
)
```

### 2. Hybrid Search (Semantic + Keyword)

**Combine vector search with keyword matching:**

```swift
func hybridSearch(
    query: String,
    keywords: [String],
    topK: Int
) async throws -> [SearchResult] {
    // 1. Semantic search
    let queryVector = try await generator.generate(text: query)
    let semanticResults = try await searchEngine.search(
        queryVector: queryVector,
        topK: topK * 2, // Over-fetch
        model: "mlx-embed-1024-v1"
    )

    // 2. Keyword filter
    let keywordMatched = semanticResults.filter { result in
        guard let metadata = result.metadata else { return false }
        return keywords.allSatisfy { keyword in
            metadata.values.contains { $0.localizedCaseInsensitiveContains(keyword) }
        }
    }

    // 3. Return top K
    return Array(keywordMatched.prefix(topK))
}
```

### 3. Model Switching and Migration

**Zero-downtime model migration:**

```swift
actor ModelMigrationManager {
    private let store: EmbeddingStore
    private let modelManager: ModelManager

    func migrateWithZeroDowntime(
        from oldModel: String,
        to newModel: String,
        generator: EmbeddingGenerator
    ) async throws {
        // 1. Register new model
        try await modelManager.registerModel(generator.modelMetadata)

        // 2. Generate new embeddings in parallel with old model serving
        let oldEmbeddings = try await store.getAllByModel(oldModel, limit: Int.max)

        for batch in oldEmbeddings.chunked(into: 100) {
            // Fetch source data
            let texts = batch.compactMap { $0.metadata?["text"] }

            // Generate with new model
            let newVectors = try await generator.generateBatch(texts: texts)

            // Save with new model name
            let newRecords = zip(batch, newVectors).map { (old, vector) in
                EmbeddingRecord(
                    id: old.id,
                    vector: vector,
                    model: newModel, // New model
                    dimension: vector.count,
                    sourceType: old.sourceType,
                    metadata: old.metadata
                )
            }

            try await store.saveBatch(newRecords)
        }

        // 3. Switch applications to use new model
        // (Application-level configuration change)

        // 4. Deprecate old model
        try await modelManager.deprecateModel(
            name: oldModel,
            reason: "Migrated to \(newModel)",
            replacementModel: newModel
        )

        // 5. Optional: Delete old embeddings after grace period
        // for id in oldEmbeddings.map({ $0.id }) {
        //     try await store.delete(id: id, model: oldModel)
        // }
    }
}
```

### 4. Background Regeneration Jobs

**Async job processing:**

```swift
actor BackgroundJobProcessor {
    private let store: EmbeddingStore
    private let generator: EmbeddingGenerator

    func scheduleRegeneration(
        model: String,
        sourceType: SourceType? = nil
    ) async throws -> String {
        let jobID = UUID().uuidString

        // Start background task
        Task.detached {
            do {
                try await self.executeRegeneration(
                    jobID: jobID,
                    model: model,
                    sourceType: sourceType
                )
            } catch {
                print("Job \(jobID) failed: \(error)")
            }
        }

        return jobID
    }

    private func executeRegeneration(
        jobID: String,
        model: String,
        sourceType: SourceType?
    ) async throws {
        print("Job \(jobID) started")

        // Get embeddings to regenerate
        let embeddings: [EmbeddingRecord]
        if let sourceType = sourceType {
            embeddings = try await store.getAllBySourceType(sourceType, model: model, limit: Int.max)
        } else {
            embeddings = try await store.getAllByModel(model, limit: Int.max)
        }

        var processed = 0
        let total = embeddings.count

        // Process in batches
        for batch in embeddings.chunked(into: 100) {
            // Fetch source texts
            let texts = batch.compactMap { $0.metadata?["text"] }

            // Regenerate
            let newVectors = try await generator.generateBatch(texts: texts)

            // Update
            let updatedRecords = zip(batch, newVectors).map { (old, vector) in
                EmbeddingRecord(
                    id: old.id,
                    vector: vector,
                    model: model,
                    dimension: vector.count,
                    sourceType: old.sourceType,
                    metadata: old.metadata,
                    updatedAt: Date()
                )
            }

            try await store.saveBatch(updatedRecords)

            processed += batch.count
            print("Job \(jobID): \(processed)/\(total)")
        }

        print("Job \(jobID) completed")
    }
}
```

---

## Summary

The `fdb-embedding-layer` API provides:

1. **EmbeddingStore**: Efficient vector storage with caching
2. **ModelManager**: Model lifecycle and metadata management
3. **SearchEngine**: Similarity search with multiple metrics
4. **EmbeddingGenerator**: Pluggable model integration interface

### Key Features

- Actor-based concurrency for thread safety
- Batch operations for high throughput
- LRU caching for low latency
- Multiple similarity metrics
- Flexible filtering
- Model versioning and migration
- Integration with triple and ontology layers

### Performance Highlights

- Single save: 5-10ms
- Batch save (100): 50-100ms
- Cached get: <1ms
- Search (10K corpus): 50-100ms
- Search (1M+ with ANN): 10-50ms

### Best Practices

1. Use batch operations for bulk processing
2. Pre-warm cache for frequently accessed vectors
3. Filter before search to reduce corpus size
4. Choose appropriate model dimension for use case
5. Implement graceful error handling and retries
6. Monitor cache hit rates and adjust size accordingly

---

**Document Version**: 1.0
**Last Updated**: 2025-10-30
**Status**: Complete
