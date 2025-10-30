# fdb-embedding-layer Architecture

## Overview

`fdb-embedding-layer` is a semantic vector layer built on FoundationDB that provides vector embedding storage, similarity search, and model management capabilities for knowledge management systems. It transforms structured knowledge (triples, ontology classes, text) into dense vector representations, enabling semantic search, concept similarity, and clustering operations.

This layer serves as the semantic search foundation for the knowledge management stack:
- Receives structured data from `fdb-triple-layer` and `fdb-ontology-layer`
- Generates and stores vector embeddings for semantic search
- Provides similarity search capabilities to `fdb-knowledge-layer`
- Supports multiple embedding models with versioning

## Design Philosophy

### Core Principles

1. **Model Agnostic**: Support multiple embedding models (MLX, OpenAI, Cohere, custom)
2. **Efficient Storage**: Optimize FoundationDB for high-dimensional vector storage
3. **Scalable Search**: Provide graduated search strategies from brute-force to ANN to external vector DBs
4. **Actor-Based Concurrency**: Ensure thread safety through Swift's actor model
5. **Separation of Concerns**: Distinct components for storage, search, and model management
6. **Caching Strategy**: LRU caching for frequently accessed vectors
7. **Batch Operations**: Support bulk generation and search for performance

### Non-Goals

- Custom embedding model training (use external models)
- Real-time model fine-tuning (batch updates only)
- Complex graph traversal (delegate to knowledge layer)
- Full-text search (use dedicated search engines)

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   fdb-knowledge-layer                       │
│        (Unified API, hybrid search, reasoning)              │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐  ┌─────────▼─────────┐  ┌───────▼────────┐
│ fdb-triple-    │  │ fdb-ontology-     │  │ fdb-embedding- │
│ layer          │─►│ layer             │─►│ layer          │
│ (Triple Store) │  │ (Schema/Semantic) │  │ (Vector Store) │
└────────────────┘  └───────────────────┘  └────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   FoundationDB     │
                    │ (Distributed KVS)  │
                    └────────────────────┘

External Integration (Phase 2+):
┌────────────────┐
│  Vector DBs    │  ← Milvus, Weaviate, Pinecone
│  (ANN Search)  │
└────────────────┘
```

### Layer Responsibilities

| Layer | Responsibility | Uses Embedding For |
|-------|----------------|-------------------|
| **fdb-embedding-layer** | Store & search vectors | Self-contained |
| **fdb-triple-layer** | Store triples | Semantic search over triples |
| **fdb-ontology-layer** | Define semantic structure | Class/concept embeddings |
| **fdb-knowledge-layer** | Unified knowledge API | Hybrid search (structured + semantic) |

## Component Design

### 1. Core Data Models

#### EmbeddingRecord
- Represents a single vector embedding
- Links embedding to source entity (triple ID, class URI, text hash)
- Tracks model, dimension, creation time
- Stores dense float32 vector

```swift
public struct EmbeddingRecord: Codable, Sendable {
    public let id: String                   // Entity ID (e.g., "triple:12345", "class:Person")
    public let vector: [Float]              // Dense embedding vector
    public let model: String                // Model identifier (e.g., "mlx-embed-1024")
    public let dimension: Int               // Vector dimension (e.g., 1024)
    public let sourceType: EmbeddingSourceType  // .triple, .entity, .text
    public let metadata: [String: String]?  // Optional metadata (source URI, etc.)
    public let createdAt: Date              // Creation timestamp
}

public enum EmbeddingSourceType: String, Codable, Sendable {
    case triple         // Generated from RDF triple
    case entity         // Generated from ontology class
    case text           // Generated from raw text
    case batch          // Part of batch generation
}
```

#### EmbeddingModelMetadata
- Defines embedding model characteristics
- Tracks model version and provider
- Specifies vector dimension and normalization

```swift
public struct EmbeddingModelMetadata: Codable, Sendable {
    public let name: String                 // Unique model identifier
    public let version: String              // Model version (e.g., "1.0.0")
    public let dimension: Int               // Output vector dimension
    public let provider: String             // Provider (e.g., "Apple MLX", "OpenAI")
    public let modelType: String            // Type (e.g., "BERT", "GPT", "Custom")
    public let description: String?         // Human-readable description
    public let normalized: Bool             // Whether vectors are L2-normalized
    public let maxInputLength: Int?         // Max input tokens/chars
    public let createdAt: Date              // Registration timestamp
}
```

#### SearchResult
- Represents a similarity search result
- Contains entity ID, similarity score, and optional metadata

```swift
public struct SearchResult: Sendable {
    public let id: String                   // Entity ID
    public let score: Float                 // Similarity score (0.0-1.0)
    public let vector: [Float]?             // Optional: return vector
    public let metadata: [String: String]?  // Optional: metadata
    public let distance: Float              // Raw distance (depends on metric)
}
```

### 2. Storage Layer

#### EmbeddingStore (Actor)
- Primary storage interface for embeddings
- Manages FoundationDB transactions
- Implements caching for frequently accessed vectors
- Supports batch operations

**Core APIs**:
```swift
public actor EmbeddingStore {
    // CRUD Operations
    func save(_ record: EmbeddingRecord) async throws
    func get(id: String, model: String) async throws -> EmbeddingRecord?
    func delete(id: String, model: String) async throws
    func exists(id: String, model: String) async throws -> Bool

    // Batch Operations
    func saveBatch(_ records: [EmbeddingRecord]) async throws
    func getBatch(ids: [String], model: String) async throws -> [String: EmbeddingRecord]
    func deleteBatch(ids: [String], model: String) async throws

    // Query Operations
    func listByModel(_ model: String, limit: Int) async throws -> [EmbeddingRecord]
    func listBySourceType(_ sourceType: EmbeddingSourceType, model: String) async throws -> [EmbeddingRecord]

    // Cache Management
    func clearCache()
    func getCacheStats() -> CacheStats
}
```

**Caching Strategy**:
- LRU cache for frequently accessed embeddings
- Default cache size: 10,000 vectors (~40MB for 1024-dim float32)
- Cache key: `"\(model):\(id)"`
- Configurable cache size and TTL

#### SubspaceManager
- Manages FoundationDB subspace organization
- Provides isolated namespaces for embeddings, models, and indexes
- Handles key encoding/decoding

**Subspace Structure**:
```
<rootPrefix>/
  embedding/
    vector/<model>/<id>          → EmbeddingRecord (compressed)
    model/<name>                 → EmbeddingModelMetadata JSON
    index/by_source/<sourceType>/<model>/<id>  → Empty (index)
    metadata/vector_count        → UInt64 counter
    metadata/last_updated        → Unix timestamp
    cache/<model>/<id>           → Cached vector (transient)
```

### 3. Model Management

#### ModelManager (Actor)
- Manages embedding model metadata
- Tracks model versions and lifecycles
- Validates model compatibility

**Core APIs**:
```swift
public actor ModelManager {
    // Model Registration
    func registerModel(_ metadata: EmbeddingModelMetadata) async throws
    func getModel(name: String) async throws -> EmbeddingModelMetadata?
    func listModels() async throws -> [EmbeddingModelMetadata]
    func deleteModel(name: String) async throws

    // Model Lifecycle
    func deprecateModel(name: String, reason: String) async throws
    func validateModelCompatibility(_ modelA: String, _ modelB: String) throws -> Bool

    // Model Statistics
    func getModelStats(name: String) async throws -> ModelStats
}

public struct ModelStats: Sendable {
    public let modelName: String
    public let vectorCount: Int
    public let avgGenerationTime: TimeInterval?
    public let lastUsed: Date?
}
```

**Model Versioning Strategy**:
- Each model has unique name (e.g., "mlx-embed-1024-v1")
- Version changes require new model registration
- Old embeddings remain valid (no automatic re-generation)
- Applications handle migration via batch re-generation

### 4. Similarity Search

#### SearchEngine (Actor)
- Implements similarity search algorithms
- Supports multiple distance metrics
- Provides batch search capabilities

**Core APIs**:
```swift
public actor SearchEngine {
    // Single Query
    func search(
        queryVector: [Float],
        topK: Int,
        model: String,
        similarityMetric: SimilarityMetric,
        filter: SearchFilter?
    ) async throws -> [SearchResult]

    // Batch Search
    func batchSearch(
        queryVectors: [[Float]],
        topK: Int,
        model: String,
        similarityMetric: SimilarityMetric
    ) async throws -> [[SearchResult]]

    // Filtered Search
    func searchWithFilter(
        queryVector: [Float],
        topK: Int,
        model: String,
        filter: SearchFilter
    ) async throws -> [SearchResult]
}

public enum SimilarityMetric: String, Sendable {
    case cosine         // Cosine similarity (normalized dot product)
    case innerProduct   // Inner product (dot product)
    case euclidean      // Euclidean distance (L2)
    case manhattan      // Manhattan distance (L1)
}

public struct SearchFilter: Sendable {
    public let sourceTypes: [EmbeddingSourceType]?
    public let metadata: [String: String]?
    public let createdAfter: Date?
    public let createdBefore: Date?
}
```

**Search Algorithm Selection**:
```
Phase 1 (Current): Brute-Force
├─ Corpus Size: < 10K vectors
├─ Latency: 50-100ms
└─ Algorithm: Linear scan with SIMD

Phase 2 (Future): Hybrid
├─ Corpus Size: 10K - 100K vectors
├─ Latency: 10-50ms
└─ Algorithm: Quantization + ANN (HNSW, IVF)

Phase 3 (Advanced): External VectorDB
├─ Corpus Size: > 100K vectors
├─ Latency: 5-20ms
└─ Algorithm: Milvus/Weaviate integration
```

### 5. Embedding Generation

#### EmbeddingGenerator (Protocol)
- Interface for embedding model integration
- Supports text, triple, and entity embeddings
- Provides batch generation

```swift
public protocol EmbeddingGenerator: Sendable {
    // Single Generation
    func generate(text: String) async throws -> [Float]
    func generate(triple: Triple) async throws -> [Float]
    func generate(entity: OntologyClass) async throws -> [Float]

    // Batch Generation
    func generateBatch(texts: [String]) async throws -> [[Float]]
    func generateBatch(triples: [Triple]) async throws -> [[Float]]

    // Model Info
    var modelMetadata: EmbeddingModelMetadata { get }
}
```

**Implementation Examples**:
```swift
// MLX-based local model
public actor MLXEmbeddingGenerator: EmbeddingGenerator {
    private let model: MLXModel

    public func generate(text: String) async throws -> [Float] {
        let tokens = tokenize(text)
        let output = try await model.forward(tokens)
        return normalize(output.pooled)
    }
}

// OpenAI API
public actor OpenAIEmbeddingGenerator: EmbeddingGenerator {
    private let apiKey: String

    public func generate(text: String) async throws -> [Float] {
        let response = try await openAI.createEmbedding(
            model: "text-embedding-3-large",
            input: text
        )
        return response.data[0].embedding
    }
}
```

## Embedding Generation Strategies

### 1. Text Embeddings
- Direct text → vector mapping
- Used for free-form queries and descriptions

```swift
let generator: EmbeddingGenerator = MLXEmbeddingGenerator()
let text = "A person who works at a software company"
let vector = try await generator.generate(text: text)

let record = EmbeddingRecord(
    id: "text:\(text.hashValue)",
    vector: vector,
    model: generator.modelMetadata.name,
    dimension: vector.count,
    sourceType: .text,
    metadata: ["text": text],
    createdAt: Date()
)
try await embeddingStore.save(record)
```

### 2. Triple Embeddings
- Convert RDF triple to text representation
- Options: subject-predicate-object concatenation, graph structure, contextual

**Strategies**:

**Simple Concatenation**:
```swift
func generateTripleEmbedding(triple: Triple) async throws -> [Float] {
    // Convert: (<subject> <predicate> <object>)
    let text = "\(triple.subject.value) \(triple.predicate.value) \(triple.object.value)"
    return try await generator.generate(text: text)
}
```

**Contextual (with labels)**:
```swift
func generateContextualTripleEmbedding(triple: Triple) async throws -> [Float] {
    // Fetch human-readable labels
    let subjectLabel = try await dictionaryStore.getLabel(uri: triple.subject)
    let predicateLabel = try await ontologyStore.getPredicateLabel(triple.predicate)
    let objectLabel = try await dictionaryStore.getLabel(uri: triple.object)

    let text = "\(subjectLabel) \(predicateLabel) \(objectLabel)"
    return try await generator.generate(text: text)
}
```

**Graph Structure (advanced)**:
```swift
func generateGraphTripleEmbedding(triple: Triple) async throws -> [Float] {
    // Include 1-hop neighborhood
    let neighbors = try await tripleStore.getNeighbors(subject: triple.subject)
    let context = neighbors.map { "\($0.predicate) \($0.object)" }.joined(separator: ", ")

    let text = "\(triple.subject) \(triple.predicate) \(triple.object). Context: \(context)"
    return try await generator.generate(text: text)
}
```

### 3. Entity Embeddings (from Ontology Classes)
- Convert ontology class definition to text
- Include class hierarchy and properties

```swift
func generateEntityEmbedding(cls: OntologyClass) async throws -> [Float] {
    // Generate rich text representation
    var parts: [String] = []
    parts.append("Class: \(cls.name)")

    if let parent = cls.parent {
        parts.append("Inherits from: \(parent)")
    }

    if let description = cls.description {
        parts.append("Description: \(description)")
    }

    if !cls.properties.isEmpty {
        parts.append("Properties: \(cls.properties.joined(separator: ", "))")
    }

    let text = parts.joined(separator: ". ")
    return try await generator.generate(text: text)
}
```

### 4. Batch Generation
- Optimize throughput via batch processing
- Reduce model overhead (tokenization, model loading)

```swift
func batchGenerateTripleEmbeddings(triples: [Triple]) async throws {
    // Convert triples to text batch
    let texts = triples.map { triple in
        "\(triple.subject.value) \(triple.predicate.value) \(triple.object.value)"
    }

    // Generate embeddings in batch
    let vectors = try await generator.generateBatch(texts: texts)

    // Create embedding records
    let records = zip(triples, vectors).map { (triple, vector) in
        EmbeddingRecord(
            id: "triple:\(triple.id)",
            vector: vector,
            model: generator.modelMetadata.name,
            dimension: vector.count,
            sourceType: .triple,
            metadata: ["tripleId": "\(triple.id)"],
            createdAt: Date()
        )
    }

    // Save batch
    try await embeddingStore.saveBatch(records)
}
```

## Storage Strategy

### FoundationDB Key Layout

All embedding data uses Tuple encoding for efficient range queries:

```
Key Structure:
<rootPrefix> / "embedding" / "vector" / <model> / <id>  → Compressed EmbeddingRecord
<rootPrefix> / "embedding" / "model" / <name>           → ModelMetadata JSON
<rootPrefix> / "embedding" / "index" / "by_source" / <sourceType> / <model> / <id>  → Empty
<rootPrefix> / "embedding" / "metadata" / "vector_count"  → UInt64
<rootPrefix> / "embedding" / "metadata" / "last_updated"  → Int64 timestamp
```

**Example Keys**:
```
// Vector storage
Tuple("myapp", "embedding", "vector", "mlx-embed-1024", "triple:12345")
  → [compressed float32 array] + metadata

// Model registration
Tuple("myapp", "embedding", "model", "mlx-embed-1024")
  → {"name":"mlx-embed-1024","dimension":1024,...}

// Source type index
Tuple("myapp", "embedding", "index", "by_source", "triple", "mlx-embed-1024", "triple:12345")
  → (empty, for indexing)
```

### Value Encoding

#### Vector Compression
To stay within FDB's 100KB value limit and optimize storage:

**Option 1: Float32 (default)**:
- 4 bytes per dimension
- 1024-dim = 4KB + metadata (~100 bytes) = ~4.1KB
- No compression, full precision

**Option 2: Float16 (half-precision)**:
- 2 bytes per dimension
- 1024-dim = 2KB + metadata = ~2.1KB
- 50% size reduction, minimal precision loss for search

**Option 3: Quantization (int8)**:
- 1 byte per dimension
- 1024-dim = 1KB + metadata = ~1.1KB
- 75% size reduction, requires calibration

**Implementation**:
```swift
struct VectorEncoding {
    static func encodeFloat32(_ vector: [Float]) -> FDB.Bytes {
        var bytes = FDB.Bytes()
        bytes.reserveCapacity(vector.count * 4)
        for value in vector {
            withUnsafeBytes(of: value.bitPattern) { bytes.append(contentsOf: $0) }
        }
        return bytes
    }

    static func decodeFloat32(_ bytes: FDB.Bytes) -> [Float] {
        let count = bytes.count / 4
        return (0..<count).map { i in
            let offset = i * 4
            let bits = bytes[offset..<offset+4].withUnsafeBytes { $0.load(as: UInt32.self) }
            return Float(bitPattern: bits)
        }
    }

    static func encodeFloat16(_ vector: [Float]) -> FDB.Bytes {
        // Convert Float32 → Float16
        return vector.map { Float16($0) }.flatMap {
            withUnsafeBytes(of: $0.bitPattern) { Array($0) }
        }
    }
}
```

### Caching Strategy

**LRU Cache Design**:
```swift
actor EmbeddingCache {
    private var cache: [String: CacheEntry] = [:]
    private var accessOrder: [String] = []
    private let maxSize: Int

    struct CacheEntry {
        let record: EmbeddingRecord
        var lastAccessed: Date
        var hitCount: Int
    }

    func get(key: String) -> EmbeddingRecord? {
        guard var entry = cache[key] else { return nil }

        // Update access metadata
        entry.lastAccessed = Date()
        entry.hitCount += 1
        cache[key] = entry

        // Move to end (most recently used)
        accessOrder.removeAll { $0 == key }
        accessOrder.append(key)

        return entry.record
    }

    func put(key: String, record: EmbeddingRecord) {
        // Evict least recently used if at capacity
        if cache.count >= maxSize {
            if let evictKey = accessOrder.first {
                cache.removeValue(forKey: evictKey)
                accessOrder.removeFirst()
            }
        }

        cache[key] = CacheEntry(
            record: record,
            lastAccessed: Date(),
            hitCount: 0
        )
        accessOrder.append(key)
    }
}
```

**Cache Configuration**:
```swift
struct CacheConfig {
    var maxVectors: Int = 10_000          // ~40MB for 1024-dim float32
    var ttl: TimeInterval? = 3600         // 1 hour TTL
    var evictionPolicy: EvictionPolicy = .lru
}

enum EvictionPolicy {
    case lru        // Least Recently Used
    case lfu        // Least Frequently Used
    case ttl        // Time To Live
}
```

## Similarity Search

### Phase 1: Brute-Force Search (Current)

**Target**: Up to 10K-100K vectors
**Latency**: 50-100ms for 10K vectors
**Algorithm**: Linear scan with SIMD optimization

```swift
func bruteForceSearch(
    queryVector: [Float],
    corpus: [EmbeddingRecord],
    topK: Int,
    metric: SimilarityMetric
) -> [SearchResult] {
    // Compute similarity scores
    var results: [(id: String, score: Float, record: EmbeddingRecord)] = []

    for record in corpus {
        let score = computeSimilarity(queryVector, record.vector, metric: metric)
        results.append((record.id, score, record))
    }

    // Sort by score (descending)
    results.sort { $0.score > $1.score }

    // Return top K
    return results.prefix(topK).map { result in
        SearchResult(
            id: result.id,
            score: result.score,
            vector: result.record.vector,
            metadata: result.record.metadata,
            distance: computeDistance(queryVector, result.record.vector, metric: metric)
        )
    }
}
```

**SIMD Optimization**:
```swift
import Accelerate

func computeCosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    assert(a.count == b.count)

    var dotProduct: Float = 0
    var normA: Float = 0
    var normB: Float = 0

    vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))
    vDSP_svesq(a, 1, &normA, vDSP_Length(a.count))
    vDSP_svesq(b, 1, &normB, vDSP_Length(b.count))

    return dotProduct / (sqrt(normA) * sqrt(normB))
}

func computeInnerProduct(_ a: [Float], _ b: [Float]) -> Float {
    var result: Float = 0
    vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
    return result
}

func computeEuclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
    var diff = [Float](repeating: 0, count: a.count)
    vDSP_vsub(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))

    var sumSquares: Float = 0
    vDSP_svesq(diff, 1, &sumSquares, vDSP_Length(diff.count))

    return sqrt(sumSquares)
}
```

### Phase 2: Approximate Nearest Neighbor (ANN)

**Target**: 100K-1M vectors
**Latency**: 10-50ms
**Algorithms**: HNSW, IVF, Product Quantization

**HNSW (Hierarchical Navigable Small World)**:
```
Advantages:
- Fast search: O(log N)
- High recall (>95%)
- Incremental updates

Disadvantages:
- Memory-intensive (graph structure)
- Complex implementation
```

**IVF (Inverted File Index)**:
```
Advantages:
- Fast search with quantization
- Lower memory footprint
- Good for large-scale

Disadvantages:
- Requires training
- Lower recall than HNSW
```

**Integration Strategy**:
```swift
protocol ANNIndex {
    func add(id: String, vector: [Float]) async throws
    func search(query: [Float], k: Int) async throws -> [SearchResult]
    func remove(id: String) async throws
    func rebuild() async throws
}

actor HNSWIndex: ANNIndex {
    private var graph: [String: Node] = [:]
    private let efConstruction: Int = 200
    private let M: Int = 16

    struct Node {
        let id: String
        let vector: [Float]
        var neighbors: [String]
    }

    func search(query: [Float], k: Int) async throws -> [SearchResult] {
        // HNSW search algorithm
        var visited = Set<String>()
        var candidates = PriorityQueue<SearchCandidate>()

        // Start from random entry point
        guard let entryPoint = graph.values.randomElement() else {
            return []
        }

        candidates.insert(SearchCandidate(
            id: entryPoint.id,
            distance: computeDistance(query, entryPoint.vector)
        ))

        while let candidate = candidates.popMin() {
            if visited.contains(candidate.id) { continue }
            visited.insert(candidate.id)

            // Explore neighbors
            for neighborId in graph[candidate.id]?.neighbors ?? [] {
                if !visited.contains(neighborId), let neighbor = graph[neighborId] {
                    candidates.insert(SearchCandidate(
                        id: neighbor.id,
                        distance: computeDistance(query, neighbor.vector)
                    ))
                }
            }
        }

        // Return top K
        return Array(candidates.prefix(k)).map { candidate in
            SearchResult(
                id: candidate.id,
                score: 1.0 - candidate.distance,
                vector: nil,
                metadata: nil,
                distance: candidate.distance
            )
        }
    }
}
```

### Phase 3: External Vector Database Integration

**Target**: >1M vectors, production scale
**Latency**: 5-20ms
**Options**: Milvus, Weaviate, Pinecone, Qdrant

**Integration Architecture**:
```
┌──────────────────┐
│  fdb-embedding-  │
│     layer        │
└────────┬─────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌────────┐  ┌──────────┐
│  FDB   │  │ Milvus / │
│(Metadata)  │ Weaviate │
└────────┘  └──────────┘
             (Vectors)
```

**Dual-Storage Strategy**:
```swift
actor HybridEmbeddingStore {
    private let fdbStore: EmbeddingStore       // Metadata
    private let vectorDB: VectorDBClient       // Vectors

    func save(_ record: EmbeddingRecord) async throws {
        // Store metadata in FDB
        let metadata = EmbeddingMetadata(
            id: record.id,
            model: record.model,
            dimension: record.dimension,
            sourceType: record.sourceType,
            createdAt: record.createdAt
        )
        try await fdbStore.saveMetadata(metadata)

        // Store vector in external DB
        try await vectorDB.insert(
            id: record.id,
            vector: record.vector,
            metadata: record.metadata
        )
    }

    func search(
        queryVector: [Float],
        topK: Int,
        model: String
    ) async throws -> [SearchResult] {
        // Search in vector DB
        let vectorResults = try await vectorDB.search(
            vector: queryVector,
            limit: topK,
            filter: ["model": model]
        )

        // Enrich with FDB metadata
        let ids = vectorResults.map { $0.id }
        let metadataMap = try await fdbStore.getMetadataBatch(ids: ids)

        return vectorResults.map { result in
            SearchResult(
                id: result.id,
                score: result.score,
                vector: result.vector,
                metadata: metadataMap[result.id]?.metadata,
                distance: result.distance
            )
        }
    }
}
```

**Vector DB Clients**:
```swift
protocol VectorDBClient: Sendable {
    func insert(id: String, vector: [Float], metadata: [String: String]?) async throws
    func search(vector: [Float], limit: Int, filter: [String: String]?) async throws -> [VectorDBResult]
    func delete(id: String) async throws
    func createCollection(name: String, dimension: Int) async throws
}

struct MilvusClient: VectorDBClient {
    private let host: String
    private let port: Int

    func search(vector: [Float], limit: Int, filter: [String: String]?) async throws -> [VectorDBResult] {
        // Milvus gRPC API
        let request = SearchRequest(
            collectionName: "embeddings",
            vectors: [vector],
            topK: limit,
            expr: buildFilterExpression(filter)
        )
        let response = try await milvusClient.search(request)
        return response.results.map { VectorDBResult($0) }
    }
}
```

## Performance Characteristics

### Latency Targets

| Operation | Target (p50) | Target (p99) | Notes |
|-----------|--------------|--------------|-------|
| Save Embedding | 5-10ms | 20-30ms | Single write to FDB |
| Get Embedding (cached) | <1ms | <5ms | In-memory cache hit |
| Get Embedding (uncached) | 5-10ms | 20-30ms | FDB read |
| Save Batch (100 vectors) | 50-100ms | 200-300ms | Batch write optimization |
| Get Batch (100 vectors) | 20-50ms | 100-150ms | Parallel FDB reads |
| Similarity Search (10K corpus) | 50-100ms | 200-300ms | Brute-force SIMD |
| Similarity Search (100K corpus) | 200-500ms | 1-2s | Brute-force (degraded) |
| Similarity Search (1M+ corpus, ANN) | 10-50ms | 100-200ms | HNSW/IVF index |
| Similarity Search (External VectorDB) | 5-20ms | 50-100ms | Milvus/Weaviate |

### Throughput Expectations

| Operation | Throughput | Concurrency |
|-----------|------------|-------------|
| Save Embedding | 1,000-5,000 ops/sec | 10-50 actors |
| Get Embedding (cached) | 100,000+ ops/sec | Memory bound |
| Batch Save (100 vectors) | 500-1,000 batches/sec | 5,000-10,000 vectors/sec |
| Similarity Search (brute-force) | 100-500 queries/sec | CPU bound |
| Similarity Search (ANN) | 1,000-5,000 queries/sec | Memory + CPU |
| Similarity Search (VectorDB) | 5,000-10,000 queries/sec | Network + VectorDB capacity |

### Scalability Considerations

**Storage Scalability**:
```
Vector Count vs Storage Size:
- 10K vectors × 1024-dim × 4 bytes = ~40MB
- 100K vectors = ~400MB
- 1M vectors = ~4GB
- 10M vectors = ~40GB

FDB Limit: Petabyte scale
Bottleneck: Query latency, not storage
```

**Search Scalability**:
```
Brute-Force Complexity: O(N × D)
- N = number of vectors
- D = dimension

10K vectors × 1024-dim = 10M dot products
100K vectors × 1024-dim = 100M dot products (too slow)

Solution: ANN or VectorDB
```

**Concurrent Access**:
- **Read Scalability**: Near-linear with caching
- **Write Scalability**: Limited by FDB transaction throughput (~10K/sec)
- **Search Scalability**: CPU-bound for brute-force, memory-bound for ANN

## Concurrency Model

### Actor-Based Isolation

All components are actors for thread safety:

```swift
public actor EmbeddingStore {
    private let database: any DatabaseProtocol
    private var cache: EmbeddingCache
    private let subspaceManager: SubspaceManager
}

public actor ModelManager {
    private let database: any DatabaseProtocol
    private var modelCache: [String: EmbeddingModelMetadata]
}

public actor SearchEngine {
    private let store: EmbeddingStore
    private var indexCache: [String: ANNIndex]
}
```

**Benefits**:
- No locks or semaphores needed
- Serial access to mutable state
- Concurrent reads via `async` methods
- Sendable types for data transfer

### Cache Consistency

**Cache Invalidation Strategy**:
```swift
actor EmbeddingStore {
    private var cache: EmbeddingCache

    func save(_ record: EmbeddingRecord) async throws {
        // 1. Write to FDB
        try await database.withTransaction { transaction in
            let key = subspaceManager.vectorKey(model: record.model, id: record.id)
            let value = try encodeRecord(record)
            transaction.setValue(value, for: key)
        }

        // 2. Invalidate cache
        await cache.remove(key: "\(record.model):\(record.id)")

        // Or update cache immediately (write-through)
        await cache.put(key: "\(record.model):\(record.id)", record: record)
    }
}
```

## Versioning Strategy

### Model Versioning

**Version Identifier**:
- Include version in model name: `"mlx-embed-1024-v1"`
- Track creation date and deprecation status

**Version Migration**:
```swift
actor EmbeddingMigrationManager {
    func migrateModel(
        fromModel oldModel: String,
        toModel newModel: String,
        generator: EmbeddingGenerator
    ) async throws {
        // 1. Get all embeddings for old model
        let oldEmbeddings = try await store.listByModel(oldModel, limit: Int.max)

        // 2. Re-generate embeddings
        for batch in oldEmbeddings.chunked(into: 100) {
            let texts = try await fetchSourceTexts(for: batch)
            let newVectors = try await generator.generateBatch(texts: texts)

            let newRecords = zip(batch, newVectors).map { (old, newVector) in
                EmbeddingRecord(
                    id: old.id,
                    vector: newVector,
                    model: newModel,
                    dimension: newVector.count,
                    sourceType: old.sourceType,
                    metadata: old.metadata,
                    createdAt: Date()
                )
            }

            try await store.saveBatch(newRecords)
        }

        // 3. Deprecate old model
        try await modelManager.deprecateModel(name: oldModel, reason: "Migrated to \(newModel)")
    }
}
```

### Embedding Regeneration

**Triggers for Re-generation**:
1. Model upgrade
2. Source data change (triple/entity modified)
3. Embedding quality issues
4. Dimension change

**Regeneration Strategy**:
```swift
func regenerateEmbeddings(
    sourceType: EmbeddingSourceType,
    model: String,
    generator: EmbeddingGenerator
) async throws {
    switch sourceType {
    case .triple:
        let triples = try await tripleStore.listAll()
        try await batchGenerateTripleEmbeddings(triples: triples)

    case .entity:
        let classes = try await ontologyStore.listClasses()
        for cls in classes {
            let vector = try await generateEntityEmbedding(cls: cls)
            let record = EmbeddingRecord(
                id: "class:\(cls.name)",
                vector: vector,
                model: model,
                dimension: vector.count,
                sourceType: .entity,
                metadata: ["className": cls.name],
                createdAt: Date()
            )
            try await embeddingStore.save(record)
        }

    case .text:
        // Application-specific regeneration
        break

    case .batch:
        // No-op
        break
    }
}
```

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
}
```

### Recovery Strategies

| Error | Strategy | Action |
|-------|----------|--------|
| `modelNotFound` | Fail fast | Register model first |
| `vectorNotFound` | Return nil | Generate on-demand |
| `dimensionMismatch` | Reject | Validate before save |
| `generationFailed` | Retry | Exponential backoff |
| `searchFailed` | Degrade | Fall back to smaller corpus |
| `cacheError` | Bypass cache | Read from FDB |
| `storageError` | Retry | FDB auto-retry |

### Error Handling Example

```swift
func saveWithRetry(_ record: EmbeddingRecord, maxRetries: Int = 3) async throws {
    var attempt = 0
    while attempt < maxRetries {
        do {
            try await save(record)
            return
        } catch let error as EmbeddingError {
            switch error {
            case .storageError:
                // Retry with exponential backoff
                attempt += 1
                if attempt < maxRetries {
                    let delay = TimeInterval(pow(2.0, Double(attempt)))
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                } else {
                    throw error
                }
            default:
                // Non-retryable error
                throw error
            }
        }
    }
}
```

## Future Enhancements

### Phase 2: Advanced Indexing

1. **Product Quantization (PQ)**:
   - Compress vectors to 8-64 bytes
   - Fast distance computation
   - 10-20x memory reduction

2. **Inverted File Index (IVF)**:
   - Cluster vectors into partitions
   - Search only relevant partitions
   - 10-100x speedup for large corpora

3. **Hybrid Search**:
   - Combine structured filters with semantic search
   - Example: "Find Person entities similar to 'software engineer' created after 2025-01-01"

### Phase 3: Distributed Search

1. **Sharded Vector Storage**:
   - Partition vectors across FDB cluster
   - Parallel search across shards
   - Merge results

2. **Distributed ANN**:
   - FAISS distributed mode
   - Horizontal scaling

3. **Caching Layer**:
   - Redis for hot vectors
   - Multi-tier caching (memory → Redis → FDB)

### Phase 4: GPU Acceleration

1. **GPU-based Search**:
   - Use Metal/CUDA for similarity computation
   - 100x speedup for large batches

2. **GPU-based Indexing**:
   - Accelerate HNSW/IVF construction
   - Online index updates

### Phase 5: Advanced Features

1. **Multi-Modal Embeddings**:
   - Text + Image embeddings
   - Cross-modal search

2. **Dynamic Embeddings**:
   - Update embeddings incrementally
   - Online learning

3. **Embedding Analytics**:
   - Cluster visualization
   - Concept drift detection
   - Quality metrics

4. **Query Optimization**:
   - Query caching
   - Pre-computed similarity graphs
   - Materialized views

## Testing Strategy

### Unit Tests
- Model encoding/decoding
- Vector compression/decompression
- Similarity metric correctness
- Cache eviction logic

### Integration Tests
- FDB transaction behavior
- Batch operations
- Cache consistency
- Model registration

### Performance Tests
- Latency benchmarks
- Throughput measurements
- Cache hit rate analysis
- Search accuracy (recall@K)

### Accuracy Tests
- Embedding quality
- Search relevance
- ANN recall rate

## Monitoring and Observability

### Metrics

```swift
struct EmbeddingMetrics {
    var vectorCount: Int
    var modelCount: Int
    var cacheHitRate: Double
    var avgSearchLatency: TimeInterval
    var searchThroughput: Double
    var storageSize: Int64
    var generationLatency: Histogram
}
```

### Logging

- **DEBUG**: Cache hits/misses, query details
- **INFO**: Save/delete operations, search queries
- **WARNING**: Cache evictions, slow queries
- **ERROR**: Generation failures, storage errors

---

**Document Version**: 1.0
**Last Updated**: 2025-10-30
**Status**: Draft
