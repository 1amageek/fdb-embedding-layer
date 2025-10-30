# fdb-embedding-layer Storage Layout

## Overview

This document provides a comprehensive specification of the FoundationDB storage layout for `fdb-embedding-layer`. The embedding layer stores high-dimensional vector embeddings alongside metadata, supporting efficient similarity search, batch operations, and model management.

The storage design optimizes for:
- **Fast vector retrieval**: Single-key lookups for embedding vectors
- **Efficient range queries**: Index-based filtering by model, source type, and timestamp
- **Atomic statistics updates**: Lock-free counters for metadata tracking
- **Scalability**: Support for millions of embeddings with minimal overhead
- **Compression**: Optional Float16/int8 quantization for space efficiency

All data is stored using FoundationDB's Tuple encoding to maintain lexicographic ordering and enable efficient range scans.

---

## Table of Contents

1. [Storage Architecture](#storage-architecture)
2. [Namespace Organization](#namespace-organization)
3. [Key Structure Design](#key-structure-design)
4. [Detailed Key Structures](#detailed-key-structures)
5. [Value Encoding](#value-encoding)
6. [Storage Patterns](#storage-patterns)
7. [Transaction Patterns](#transaction-patterns)
8. [Performance Optimizations](#performance-optimizations)
9. [Space Efficiency](#space-efficiency)
10. [Migration Strategies](#migration-strategies)
11. [Concrete Examples](#concrete-examples)

---

## Storage Architecture

### Hierarchical Key Space

```
<rootPrefix>/
├── embedding/
│   ├── vector/<model>/<id>                           → Vector bytes + metadata
│   ├── model/<modelName>                            → ModelMetadata JSON
│   ├── index/
│   │   ├── source/<sourceType>/<model>/<id>         → Empty (index entry)
│   │   └── timestamp/<timestamp>/<id>               → Empty (index entry)
│   ├── stats/
│   │   ├── <model>/vector_count                     → UInt64 counter
│   │   ├── <model>/total_dimension                  → UInt64 sum
│   │   └── <model>/last_updated                     → Int64 timestamp
│   └── job/<jobId>                                  → RegenerationJob JSON
```

### Design Principles

1. **Tuple-based keys**: All keys use FDB Tuple encoding for lexicographic ordering
2. **Hierarchical namespacing**: Logical separation via subspace prefixes
3. **Index separation**: Secondary indexes stored separately from primary data
4. **Atomic counters**: Statistics tracked using FDB atomic operations
5. **JSON metadata**: Model and job metadata encoded as JSON for flexibility

---

## Namespace Organization

### Primary Namespaces

| Namespace | Purpose | Example Key |
|-----------|---------|-------------|
| `embedding/vector` | Store embedding vectors and metadata | `(root, "embedding", "vector", model, id)` |
| `embedding/model` | Store model metadata | `(root, "embedding", "model", modelName)` |
| `embedding/index` | Secondary indexes for queries | `(root, "embedding", "index", type, ...)` |
| `embedding/stats` | Model-level statistics | `(root, "embedding", "stats", model, stat)` |
| `embedding/job` | Background job tracking | `(root, "embedding", "job", jobId)` |

### Subspace Isolation

Each namespace is isolated using a root prefix to support multiple embedding layers in the same FDB cluster:

```swift
// Example: Multiple applications on same cluster
let app1Store = EmbeddingStore(database: db, rootPrefix: "app1")
let app2Store = EmbeddingStore(database: db, rootPrefix: "app2")

// Keys are isolated:
// app1: ("app1", "embedding", "vector", "mlx-embed", "id1")
// app2: ("app2", "embedding", "vector", "mlx-embed", "id1")
```

---

## Key Structure Design

### Tuple Encoding Principles

All keys use FDB Tuple encoding, which:
- Preserves lexicographic ordering
- Supports heterogeneous types (String, Int64, Bool, etc.)
- Enables efficient prefix scans
- Provides compact binary representation

### Key Component Types

| Component | Type | Purpose | Example |
|-----------|------|---------|---------|
| Root Prefix | String | Application isolation | `"myapp"` |
| Namespace | String | Logical grouping | `"embedding"` |
| Subspace | String | Data category | `"vector"`, `"model"`, `"index"` |
| Model Name | String | Model identifier | `"mlx-embed-1024-v1"` |
| Entity ID | String | Unique entity identifier | `"triple:12345"` |
| Timestamp | Int64 | Unix timestamp (milliseconds) | `1730332800000` |
| Source Type | String | Embedding source | `"triple"`, `"entity"`, `"text"` |

### Lexicographic Ordering

Tuple encoding guarantees that range queries work correctly:

```
Key Order (ascending):
("myapp", "embedding", "vector", "mlx-embed", "triple:00001")
("myapp", "embedding", "vector", "mlx-embed", "triple:00002")
("myapp", "embedding", "vector", "mlx-embed", "triple:00003")
...
("myapp", "embedding", "vector", "mlx-embed", "triple:99999")
```

This enables efficient prefix scans:

```swift
// Get all embeddings for a specific model
let beginKey = Tuple(rootPrefix, "embedding", "vector", model).encode()
let endKey = Tuple(rootPrefix, "embedding", "vector", model, "\u{FFFF}").encode()

for try await (key, value) in transaction.getRange(
    beginSelector: .firstGreaterOrEqual(beginKey),
    endSelector: .firstGreaterThan(endKey)
) {
    // Process each embedding for this model
}
```

---

## Detailed Key Structures

### 1. Embedding Vector Keys

**Purpose**: Store vector embeddings with associated metadata

**Key Structure**:
```
(rootPrefix, "embedding", "vector", model, id) → Value
```

**Components**:
- `rootPrefix`: Application namespace (String)
- `"embedding"`: Fixed namespace literal
- `"vector"`: Subspace identifier
- `model`: Model name (String, e.g., `"mlx-embed-1024-v1"`)
- `id`: Entity identifier (String, e.g., `"triple:12345"`)

**Example**:
```swift
// Triple embedding
let key = Tuple("myapp", "embedding", "vector", "mlx-embed-1024-v1", "triple:12345")
let keyBytes = key.encode()
// Encoded: [0x02, 'm', 'y', 'a', 'p', 'p', 0x00, 0x02, 'e', 'm', 'b', ...]

// Class embedding
let key = Tuple("myapp", "embedding", "vector", "text-embedding-3-large", "class:Person")
let keyBytes = key.encode()
```

**Value Structure**: See [Value Encoding](#value-encoding)

### 2. Model Metadata Keys

**Purpose**: Store embedding model configuration and metadata

**Key Structure**:
```
(rootPrefix, "embedding", "model", modelName) → EmbeddingModelMetadata JSON
```

**Components**:
- `rootPrefix`: Application namespace
- `"embedding"`: Fixed namespace literal
- `"model"`: Subspace identifier
- `modelName`: Unique model identifier (String)

**Example**:
```swift
let key = Tuple("myapp", "embedding", "model", "mlx-embed-1024-v1")
let keyBytes = key.encode()

// Value: JSON-encoded EmbeddingModelMetadata
let metadata = EmbeddingModelMetadata(
    name: "mlx-embed-1024-v1",
    version: "1.0.0",
    dimension: 1024,
    provider: "Apple MLX",
    modelType: "Sentence-BERT",
    normalized: true
)
let value = try JSONEncoder().encode(metadata)
```

### 3. Index Keys

#### 3.1 Source Type Index

**Purpose**: Query embeddings by source type

**Key Structure**:
```
(rootPrefix, "embedding", "index", "source", sourceType, model, id) → Empty
```

**Components**:
- `sourceType`: Source category (`"triple"`, `"entity"`, `"text"`, `"batch"`)
- `model`: Model name
- `id`: Entity identifier

**Example**:
```swift
// Index entry for triple embedding
let key = Tuple(
    "myapp",
    "embedding",
    "index",
    "source",
    "triple",              // sourceType
    "mlx-embed-1024-v1",   // model
    "triple:12345"         // id
)

// Value: Empty (this is an index entry only)
let value = FDB.Bytes()
```

**Query Pattern**:
```swift
// Get all triple embeddings for a model
let beginKey = Tuple(
    rootPrefix,
    "embedding",
    "index",
    "source",
    "triple",
    model
).encode()

let endKey = Tuple(
    rootPrefix,
    "embedding",
    "index",
    "source",
    "triple",
    model,
    "\u{FFFF}"
).encode()

for try await (key, _) in transaction.getRange(
    beginSelector: .firstGreaterOrEqual(beginKey),
    endSelector: .firstGreaterThan(endKey),
    snapshot: true
) {
    // Decode key to get entity ID
    let elements = try Tuple.decode(from: key)
    let entityId = elements[6] as! String  // Index 6 is the ID component

    // Fetch actual embedding
    let embeddingKey = Tuple(rootPrefix, "embedding", "vector", model, entityId).encode()
    let embeddingData = try await transaction.getValue(for: embeddingKey)
}
```

#### 3.2 Timestamp Index

**Purpose**: Query embeddings by creation time

**Key Structure**:
```
(rootPrefix, "embedding", "index", "timestamp", timestamp, model, id) → Empty
```

**Components**:
- `timestamp`: Unix timestamp in milliseconds (Int64)
- `model`: Model name
- `id`: Entity identifier

**Example**:
```swift
// Index entry with timestamp
let timestamp = Int64(Date().timeIntervalSince1970 * 1000)  // Milliseconds
let key = Tuple(
    "myapp",
    "embedding",
    "index",
    "timestamp",
    timestamp,              // Int64 timestamp
    "mlx-embed-1024-v1",
    "triple:12345"
)
```

**Query Pattern**:
```swift
// Get embeddings created in last 24 hours
let now = Int64(Date().timeIntervalSince1970 * 1000)
let oneDayAgo = now - (24 * 60 * 60 * 1000)

let beginKey = Tuple(
    rootPrefix,
    "embedding",
    "index",
    "timestamp",
    oneDayAgo,
    model
).encode()

let endKey = Tuple(
    rootPrefix,
    "embedding",
    "index",
    "timestamp",
    now + 1,
    model,
    "\u{FFFF}"
).encode()

for try await (key, _) in transaction.getRange(
    beginSelector: .firstGreaterOrEqual(beginKey),
    endSelector: .firstGreaterThan(endKey)
) {
    // Process recent embeddings
}
```

### 4. Statistics Keys

**Purpose**: Track model-level aggregate statistics using atomic operations

**Key Structures**:
```
(rootPrefix, "embedding", "stats", model, "vector_count")    → UInt64
(rootPrefix, "embedding", "stats", model, "total_dimension") → UInt64
(rootPrefix, "embedding", "stats", model, "last_updated")    → Int64
```

**Example**:
```swift
// Vector count key
let countKey = Tuple(
    "myapp",
    "embedding",
    "stats",
    "mlx-embed-1024-v1",
    "vector_count"
).encode()

// Value: UInt64 encoded as little-endian bytes
let count: UInt64 = 12345
let value = withUnsafeBytes(of: count.littleEndian) { Array($0) }
```

**Atomic Update**:
```swift
// Increment vector count atomically
let increment = withUnsafeBytes(of: UInt64(1).littleEndian) { Array($0) }
transaction.atomicOp(key: countKey, param: increment, mutationType: .add)

// Update last_updated timestamp
let timestamp = Int64(Date().timeIntervalSince1970 * 1000)
let timestampBytes = withUnsafeBytes(of: timestamp.littleEndian) { Array($0) }
let lastUpdatedKey = Tuple(rootPrefix, "embedding", "stats", model, "last_updated").encode()
transaction.setValue(timestampBytes, for: lastUpdatedKey)
```

### 5. Job Keys

**Purpose**: Track background regeneration jobs

**Key Structure**:
```
(rootPrefix, "embedding", "job", jobId) → RegenerationJob JSON
```

**Example**:
```swift
let key = Tuple("myapp", "embedding", "job", "job-uuid-12345").encode()

let job = RegenerationJob(
    id: "job-uuid-12345",
    sourceModel: "mlx-embed-768-v1",
    targetModel: "mlx-embed-1024-v1",
    sourceType: .triple,
    status: .running,
    totalCount: 10000,
    processedCount: 5000
)

let value = try JSONEncoder().encode(job)
transaction.setValue(value, for: key)
```

---

## Value Encoding

### Embedding Vector Values

Embedding values combine vector data with metadata in a compact binary format.

#### Format Structure

```
┌────────────────────────────────────────────────────────────┐
│ Version (1 byte) │ Flags (1 byte) │ Metadata Length (2 bytes) │
├────────────────────────────────────────────────────────────┤
│ Vector Data (dimension × bytes_per_element)                │
├────────────────────────────────────────────────────────────┤
│ Metadata JSON (variable length)                            │
└────────────────────────────────────────────────────────────┘

Version: Format version (currently 0x01)
Flags: Bit flags for encoding options
  - Bit 0-1: Vector encoding (00=Float32, 01=Float16, 10=Int8)
  - Bit 2: Normalized flag
  - Bit 3-7: Reserved
Metadata Length: Length of JSON metadata in bytes (UInt16, little-endian)
Vector Data: Binary encoded vector (see below)
Metadata JSON: Optional metadata as UTF-8 JSON
```

#### Vector Encoding Options

**Option 1: Float32 (Default)**

4 bytes per dimension, full precision:

```swift
struct VectorCodec {
    static func encodeFloat32(_ vector: [Float]) -> FDB.Bytes {
        var bytes = FDB.Bytes()
        bytes.reserveCapacity(vector.count * 4)

        for value in vector {
            withUnsafeBytes(of: value.bitPattern.littleEndian) { bytes.append(contentsOf: $0) }
        }

        return bytes
    }

    static func decodeFloat32(_ bytes: FDB.Bytes, dimension: Int) -> [Float] {
        return (0..<dimension).map { i in
            let offset = i * 4
            let bits = bytes[offset..<offset+4].withUnsafeBytes {
                $0.load(as: UInt32.self).littleEndian
            }
            return Float(bitPattern: bits)
        }
    }
}
```

**Storage Size**:
- 1024-dim vector: 4096 bytes (4 KB)
- 1536-dim vector: 6144 bytes (6 KB)
- Overhead: ~100 bytes (header + metadata)
- Total: ~4.1 - 6.1 KB per embedding

**Binary Example** (3-dimensional vector [0.5, -1.0, 0.25]):
```
Float32 encoding:
0x3F000000  (0.5 in IEEE 754)
0xBF800000  (-1.0 in IEEE 754)
0x3E800000  (0.25 in IEEE 754)

Total: 12 bytes
```

**Option 2: Float16 (Half-Precision)**

2 bytes per dimension, 50% space savings:

```swift
extension VectorCodec {
    static func encodeFloat16(_ vector: [Float]) -> FDB.Bytes {
        return vector.flatMap { value in
            let float16 = Float16(value)
            return withUnsafeBytes(of: float16.bitPattern.littleEndian) { Array($0) }
        }
    }

    static func decodeFloat16(_ bytes: FDB.Bytes, dimension: Int) -> [Float] {
        return (0..<dimension).map { i in
            let offset = i * 2
            let bits = bytes[offset..<offset+2].withUnsafeBytes {
                $0.load(as: UInt16.self).littleEndian
            }
            return Float(Float16(bitPattern: bits))
        }
    }
}
```

**Storage Size**:
- 1024-dim vector: 2048 bytes (2 KB)
- 1536-dim vector: 3072 bytes (3 KB)
- Space savings: 50% vs Float32
- Precision loss: ~0.001 for normalized vectors

**Option 3: Int8 Quantization**

1 byte per dimension, 75% space savings:

```swift
extension VectorCodec {
    // Quantize to [-128, 127] range
    static func encodeInt8(_ vector: [Float], scale: Float = 127.0) -> (bytes: FDB.Bytes, scale: Float) {
        let bytes = vector.map { value in
            let quantized = Int8(max(-128, min(127, value * scale)))
            return UInt8(bitPattern: quantized)
        }
        return (bytes, scale)
    }

    static func decodeInt8(_ bytes: FDB.Bytes, scale: Float) -> [Float] {
        return bytes.map { byte in
            Float(Int8(bitPattern: byte)) / scale
        }
    }
}
```

**Storage Size**:
- 1024-dim vector: 1024 bytes (1 KB)
- 1536-dim vector: 1536 bytes (1.5 KB)
- Space savings: 75% vs Float32
- Precision loss: ~0.008 for normalized vectors
- Requires calibration (compute scale factor)

#### Complete Value Encoding Example

```swift
actor EmbeddingValueEncoder {
    enum VectorEncoding: UInt8 {
        case float32 = 0b00
        case float16 = 0b01
        case int8    = 0b10
    }

    func encode(
        vector: [Float],
        metadata: [String: String]?,
        encoding: VectorEncoding = .float32,
        normalized: Bool = true
    ) throws -> FDB.Bytes {
        var bytes = FDB.Bytes()

        // Version byte
        bytes.append(0x01)

        // Flags byte
        var flags: UInt8 = encoding.rawValue
        if normalized {
            flags |= 0b00000100  // Set bit 2
        }
        bytes.append(flags)

        // Metadata JSON
        let metadataJSON: Data
        if let metadata = metadata {
            metadataJSON = try JSONEncoder().encode(metadata)
        } else {
            metadataJSON = Data("{}".utf8)
        }

        // Metadata length (UInt16, little-endian)
        let metadataLength = UInt16(metadataJSON.count)
        withUnsafeBytes(of: metadataLength.littleEndian) { bytes.append(contentsOf: $0) }

        // Vector data
        let vectorBytes: FDB.Bytes
        switch encoding {
        case .float32:
            vectorBytes = VectorCodec.encodeFloat32(vector)
        case .float16:
            vectorBytes = VectorCodec.encodeFloat16(vector)
        case .int8:
            let (quantized, _) = VectorCodec.encodeInt8(vector)
            vectorBytes = quantized
        }
        bytes.append(contentsOf: vectorBytes)

        // Metadata JSON
        bytes.append(contentsOf: metadataJSON)

        return bytes
    }

    func decode(_ bytes: FDB.Bytes) throws -> (vector: [Float], metadata: [String: String]?, dimension: Int) {
        guard bytes.count >= 4 else {
            throw EmbeddingError.encodingError("Value too short")
        }

        let version = bytes[0]
        guard version == 0x01 else {
            throw EmbeddingError.encodingError("Unsupported version: \(version)")
        }

        let flags = bytes[1]
        let encoding = VectorEncoding(rawValue: flags & 0b00000011)!
        let normalized = (flags & 0b00000100) != 0

        let metadataLength = bytes[2..<4].withUnsafeBytes {
            $0.load(as: UInt16.self).littleEndian
        }

        // Calculate vector data range
        let vectorStart = 4
        let metadataStart = bytes.count - Int(metadataLength)
        let vectorBytes = FDB.Bytes(bytes[vectorStart..<metadataStart])

        // Decode vector
        let bytesPerElement: Int
        switch encoding {
        case .float32: bytesPerElement = 4
        case .float16: bytesPerElement = 2
        case .int8: bytesPerElement = 1
        }

        let dimension = vectorBytes.count / bytesPerElement

        let vector: [Float]
        switch encoding {
        case .float32:
            vector = VectorCodec.decodeFloat32(vectorBytes, dimension: dimension)
        case .float16:
            vector = VectorCodec.decodeFloat16(vectorBytes, dimension: dimension)
        case .int8:
            // Note: Scale factor should be stored in metadata
            vector = VectorCodec.decodeInt8(vectorBytes, scale: 127.0)
        }

        // Decode metadata
        let metadataBytes = bytes[metadataStart...]
        let metadata = try? JSONDecoder().decode([String: String].self, from: Data(metadataBytes))

        return (vector, metadata, dimension)
    }
}
```

### Model Metadata Values

Model metadata is stored as JSON for flexibility:

```swift
let metadata = EmbeddingModelMetadata(
    name: "mlx-embed-1024-v1",
    version: "1.0.0",
    dimension: 1024,
    provider: "Apple MLX",
    modelType: "Sentence-BERT",
    normalized: true,
    maxInputLength: 512
)

let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys]  // Deterministic encoding
let value = try encoder.encode(metadata)
```

**JSON Example**:
```json
{
  "name": "mlx-embed-1024-v1",
  "version": "1.0.0",
  "dimension": 1024,
  "provider": "Apple MLX",
  "modelType": "Sentence-BERT",
  "description": "Local MLX-based sentence embedding model",
  "normalized": true,
  "maxInputLength": 512,
  "supportedLanguages": ["en", "ja"],
  "createdAt": "2025-10-30T12:00:00Z",
  "deprecated": false
}
```

### Statistics Values

Statistics are stored as binary-encoded integers for atomic operations:

```swift
// UInt64 counter
let count: UInt64 = 12345
let value = withUnsafeBytes(of: count.littleEndian) { Array($0) }
// Binary: [0x39, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]

// Int64 timestamp
let timestamp = Int64(Date().timeIntervalSince1970 * 1000)
let value = withUnsafeBytes(of: timestamp.littleEndian) { Array($0) }
```

---

## Storage Patterns

### 1. Single Embedding Save

**Pattern**: Save one embedding with indexes and statistics update

```swift
actor EmbeddingStore {
    func save(_ record: EmbeddingRecord) async throws {
        try await database.withTransaction { transaction in
            // 1. Encode vector value
            let value = try encoder.encode(
                vector: record.vector,
                metadata: record.metadata,
                encoding: .float32,
                normalized: modelMetadata.normalized
            )

            // 2. Primary key
            let vectorKey = Tuple(
                rootPrefix,
                "embedding",
                "vector",
                record.model,
                record.id
            ).encode()
            transaction.setValue(value, for: vectorKey)

            // 3. Source type index
            let sourceIndexKey = Tuple(
                rootPrefix,
                "embedding",
                "index",
                "source",
                record.sourceType.rawValue,
                record.model,
                record.id
            ).encode()
            transaction.setValue(FDB.Bytes(), for: sourceIndexKey)

            // 4. Timestamp index
            let timestamp = Int64(record.createdAt.timeIntervalSince1970 * 1000)
            let timestampIndexKey = Tuple(
                rootPrefix,
                "embedding",
                "index",
                "timestamp",
                timestamp,
                record.model,
                record.id
            ).encode()
            transaction.setValue(FDB.Bytes(), for: timestampIndexKey)

            // 5. Update statistics (atomic)
            let countKey = Tuple(
                rootPrefix,
                "embedding",
                "stats",
                record.model,
                "vector_count"
            ).encode()
            let increment = withUnsafeBytes(of: UInt64(1).littleEndian) { Array($0) }
            transaction.atomicOp(key: countKey, param: increment, mutationType: .add)

            // 6. Update last_updated timestamp
            let now = Int64(Date().timeIntervalSince1970 * 1000)
            let timestampBytes = withUnsafeBytes(of: now.littleEndian) { Array($0) }
            let lastUpdatedKey = Tuple(
                rootPrefix,
                "embedding",
                "stats",
                record.model,
                "last_updated"
            ).encode()
            transaction.setValue(timestampBytes, for: lastUpdatedKey)
        }
    }
}
```

### 2. Batch Embedding Save (Atomic)

**Pattern**: Save multiple embeddings in a single transaction

```swift
actor EmbeddingStore {
    func saveBatch(_ records: [EmbeddingRecord]) async throws {
        // Split into chunks to stay under 10MB transaction limit
        let maxBatchSize = 1000  // ~4MB for 1024-dim vectors

        for batch in records.chunked(into: maxBatchSize) {
            try await database.withTransaction { transaction in
                for record in batch {
                    // Encode value
                    let value = try encoder.encode(
                        vector: record.vector,
                        metadata: record.metadata,
                        encoding: .float32,
                        normalized: true
                    )

                    // Primary key
                    let vectorKey = Tuple(
                        rootPrefix, "embedding", "vector",
                        record.model, record.id
                    ).encode()
                    transaction.setValue(value, for: vectorKey)

                    // Indexes
                    let sourceIndexKey = Tuple(
                        rootPrefix, "embedding", "index", "source",
                        record.sourceType.rawValue, record.model, record.id
                    ).encode()
                    transaction.setValue(FDB.Bytes(), for: sourceIndexKey)

                    let timestamp = Int64(record.createdAt.timeIntervalSince1970 * 1000)
                    let timestampIndexKey = Tuple(
                        rootPrefix, "embedding", "index", "timestamp",
                        timestamp, record.model, record.id
                    ).encode()
                    transaction.setValue(FDB.Bytes(), for: timestampIndexKey)
                }

                // Update statistics (single atomic operation for entire batch)
                let countKey = Tuple(
                    rootPrefix, "embedding", "stats",
                    batch[0].model, "vector_count"
                ).encode()
                let increment = withUnsafeBytes(of: UInt64(batch.count).littleEndian) { Array($0) }
                transaction.atomicOp(key: countKey, param: increment, mutationType: .add)

                // Update last_updated
                let now = Int64(Date().timeIntervalSince1970 * 1000)
                let timestampBytes = withUnsafeBytes(of: now.littleEndian) { Array($0) }
                let lastUpdatedKey = Tuple(
                    rootPrefix, "embedding", "stats",
                    batch[0].model, "last_updated"
                ).encode()
                transaction.setValue(timestampBytes, for: lastUpdatedKey)
            }
        }
    }
}
```

### 3. Range Query by Source Type

**Pattern**: Retrieve all embeddings of a specific source type

```swift
actor EmbeddingStore {
    func listBySourceType(
        _ sourceType: SourceType,
        model: String,
        limit: Int = 1000
    ) async throws -> [EmbeddingRecord] {
        return try await database.withTransaction { transaction in
            var records: [EmbeddingRecord] = []

            // Query source type index
            let beginKey = Tuple(
                rootPrefix, "embedding", "index", "source",
                sourceType.rawValue, model
            ).encode()

            let endKey = Tuple(
                rootPrefix, "embedding", "index", "source",
                sourceType.rawValue, model, "\u{FFFF}"
            ).encode()

            let sequence = transaction.getRange(
                beginSelector: .firstGreaterOrEqual(beginKey),
                endSelector: .firstGreaterThan(endKey),
                limit: limit,
                snapshot: true
            )

            // Fetch actual embeddings
            for try await (key, _) in sequence {
                let elements = try Tuple.decode(from: key)
                let entityId = elements[6] as! String

                // Fetch embedding data
                let vectorKey = Tuple(
                    rootPrefix, "embedding", "vector", model, entityId
                ).encode()

                if let valueBytes = try await transaction.getValue(for: vectorKey, snapshot: true) {
                    let (vector, metadata, dimension) = try encoder.decode(valueBytes)

                    let record = EmbeddingRecord(
                        id: entityId,
                        vector: vector,
                        model: model,
                        dimension: dimension,
                        sourceType: sourceType,
                        metadata: metadata
                    )
                    records.append(record)
                }
            }

            return records
        }
    }
}
```

### 4. Atomic Counter Updates for Statistics

**Pattern**: Increment/decrement counters without read-modify-write

```swift
actor StatisticsManager {
    func incrementVectorCount(model: String, by delta: Int = 1) async throws {
        try await database.withTransaction { transaction in
            let key = Tuple(
                rootPrefix, "embedding", "stats", model, "vector_count"
            ).encode()

            let increment = withUnsafeBytes(of: UInt64(delta).littleEndian) { Array($0) }
            transaction.atomicOp(key: key, param: increment, mutationType: .add)
        }
    }

    func getVectorCount(model: String) async throws -> UInt64 {
        return try await database.withTransaction { transaction in
            let key = Tuple(
                rootPrefix, "embedding", "stats", model, "vector_count"
            ).encode()

            guard let bytes = try await transaction.getValue(for: key, snapshot: true) else {
                return 0
            }

            return bytes.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
        }
    }
}
```

---

## Transaction Patterns

### 1. withTransaction Usage (Recommended)

**Pattern**: Automatic retry logic for transient failures

```swift
// Save embedding with automatic retries
try await database.withTransaction { transaction in
    let value = try encoder.encode(vector: record.vector, metadata: record.metadata)
    let key = Tuple(rootPrefix, "embedding", "vector", model, id).encode()
    transaction.setValue(value, for: key)

    // Update indexes and stats
    // ...
}
```

**Benefits**:
- Automatic retry on `transaction_too_old` and `not_committed`
- Transaction lifecycle management (create, commit, cancel)
- Up to 100 retries with exponential backoff

### 2. Snapshot Reads for Search

**Pattern**: Read-only queries without conflicts

```swift
func search(queryVector: [Float], topK: Int, model: String) async throws -> [SearchResult] {
    return try await database.withTransaction { transaction in
        var results: [(id: String, score: Float)] = []

        // Read all vectors for model (snapshot read to avoid conflicts)
        let beginKey = Tuple(rootPrefix, "embedding", "vector", model).encode()
        let endKey = Tuple(rootPrefix, "embedding", "vector", model, "\u{FFFF}").encode()

        let sequence = transaction.getRange(
            beginSelector: .firstGreaterOrEqual(beginKey),
            endSelector: .firstGreaterThan(endKey),
            snapshot: true  // No read conflicts
        )

        for try await (key, valueBytes) in sequence {
            let (vector, metadata, _) = try encoder.decode(valueBytes)
            let score = computeCosineSimilarity(queryVector, vector)

            let elements = try Tuple.decode(from: key)
            let id = elements[4] as! String

            results.append((id, score))
        }

        // Sort and return top K
        results.sort { $0.score > $1.score }
        return results.prefix(topK).map { result in
            SearchResult(id: result.id, score: result.score, distance: 1.0 - result.score)
        }
    }
}
```

### 3. Atomic Operations for Counters

**Pattern**: Lock-free concurrent updates

```swift
// Multiple actors can increment counter concurrently without conflicts
actor Actor1 {
    func recordEmbedding() async throws {
        try await database.withTransaction { transaction in
            // ... save embedding ...

            let increment = withUnsafeBytes(of: UInt64(1).littleEndian) { Array($0) }
            transaction.atomicOp(key: counterKey, param: increment, mutationType: .add)
        }
    }
}

actor Actor2 {
    func recordEmbedding() async throws {
        try await database.withTransaction { transaction in
            // ... save embedding ...

            let increment = withUnsafeBytes(of: UInt64(1).littleEndian) { Array($0) }
            transaction.atomicOp(key: counterKey, param: increment, mutationType: .add)
        }
    }
}

// Both increment the same counter without conflicts
```

### 4. Batch Operations with Size Limits

**Pattern**: Stay within FDB transaction limits

```swift
// FDB Limits:
// - Transaction size: 10MB
// - Key size: 10KB
// - Value size: 100KB
// - Transaction duration: 5 seconds

func saveLargeBatch(_ records: [EmbeddingRecord]) async throws {
    let maxBatchSize = 1000  // ~4MB for 1024-dim Float32 vectors

    for batch in records.chunked(into: maxBatchSize) {
        try await database.withTransaction { transaction in
            for record in batch {
                // Ensure value size < 100KB
                let value = try encoder.encode(
                    vector: record.vector,
                    metadata: record.metadata
                )

                guard value.count < 100_000 else {
                    throw EmbeddingError.storageError("Value exceeds 100KB limit")
                }

                let key = Tuple(rootPrefix, "embedding", "vector", record.model, record.id).encode()
                transaction.setValue(value, for: key)
            }
        }
    }
}
```

---

## Performance Optimizations

### 1. Key Ordering for Range Scans

**Optimization**: Design keys for sequential access

```swift
// Good: All embeddings for a model are contiguous
Tuple(rootPrefix, "embedding", "vector", model, id)

// Result: Efficient prefix scan
let beginKey = Tuple(rootPrefix, "embedding", "vector", "mlx-embed-1024-v1").encode()
let endKey = Tuple(rootPrefix, "embedding", "vector", "mlx-embed-1024-v1", "\u{FFFF}").encode()
// Scans only relevant keys in sequential order

// Bad: Model name at end of key
Tuple(rootPrefix, "embedding", "vector", id, model)
// Result: Must scan all embeddings to filter by model
```

### 2. Index Selection Strategies

**Decision Tree**:

```
Query Pattern:
├─ By model only? → Use primary key prefix scan
├─ By model + source type? → Use source type index
├─ By model + time range? → Use timestamp index
└─ By model + source type + time range? → Use timestamp index + filter
```

**Example**:

```swift
// Query: Get triple embeddings for model X created in last week
// Strategy: Use timestamp index (most selective), filter by source type in memory

let oneWeekAgo = Int64(Date().timeIntervalSince1970 * 1000) - (7 * 24 * 60 * 60 * 1000)
let now = Int64(Date().timeIntervalSince1970 * 1000)

let beginKey = Tuple(
    rootPrefix, "embedding", "index", "timestamp", oneWeekAgo, model
).encode()

let endKey = Tuple(
    rootPrefix, "embedding", "index", "timestamp", now, model, "\u{FFFF}"
).encode()

for try await (key, _) in transaction.getRange(
    beginSelector: .firstGreaterOrEqual(beginKey),
    endSelector: .firstGreaterThan(endKey)
) {
    // Fetch embedding and filter by source type
}
```

### 3. Caching Layer Integration

**Pattern**: LRU cache for frequently accessed vectors

```swift
actor EmbeddingStore {
    private var cache: LRUCache<String, EmbeddingRecord>

    func get(id: String, model: String) async throws -> EmbeddingRecord? {
        let cacheKey = "\(model):\(id)"

        // Check cache first
        if let cached = cache.get(cacheKey) {
            return cached
        }

        // Fetch from FDB
        let record = try await database.withTransaction { transaction in
            let key = Tuple(rootPrefix, "embedding", "vector", model, id).encode()
            guard let value = try await transaction.getValue(for: key, snapshot: true) else {
                return nil
            }

            let (vector, metadata, dimension) = try encoder.decode(value)
            return EmbeddingRecord(
                id: id, vector: vector, model: model,
                dimension: dimension, sourceType: .triple, metadata: metadata
            )
        }

        // Update cache
        if let record = record {
            cache.put(cacheKey, record)
        }

        return record
    }
}

// Cache configuration
struct CacheConfig {
    let maxVectors: Int = 10_000       // ~40MB for 1024-dim
    let ttl: TimeInterval = 3600       // 1 hour
}
```

### 4. Batch Size Tuning

**Guidelines**:

| Vector Dimension | Encoding | Vectors per Transaction | Transaction Size | Latency |
|------------------|----------|-------------------------|------------------|---------|
| 384 | Float32 | 2000 | ~3MB | 50-100ms |
| 768 | Float32 | 1500 | ~4.5MB | 75-150ms |
| 1024 | Float32 | 1000 | ~4MB | 100-200ms |
| 1536 | Float32 | 700 | ~4.2MB | 150-300ms |
| 1024 | Float16 | 2000 | ~4MB | 100-200ms |
| 1024 | Int8 | 4000 | ~4MB | 100-200ms |

**Adaptive batching**:

```swift
func calculateOptimalBatchSize(dimension: Int, encoding: VectorEncoding) -> Int {
    let bytesPerVector: Int
    switch encoding {
    case .float32: bytesPerVector = dimension * 4 + 100  // 100 bytes overhead
    case .float16: bytesPerVector = dimension * 2 + 100
    case .int8: bytesPerVector = dimension * 1 + 100
    }

    let maxTransactionSize = 4_000_000  // 4MB (leave 6MB buffer)
    return maxTransactionSize / bytesPerVector
}
```

---

## Space Efficiency

### Storage Estimates per Embedding

**Formula**:
```
Total Size = Vector Size + Metadata Size + Index Overhead + Statistics Overhead

Where:
  Vector Size = (Header + Vector Data + Metadata JSON)
  Index Overhead = (Source Index Entry + Timestamp Index Entry) × Key Size
  Statistics Overhead = (amortized counter updates)
```

**Concrete Examples**:

| Configuration | Vector Size | Index Overhead | Total per Embedding |
|---------------|-------------|----------------|---------------------|
| 1024-dim Float32 | 4196 bytes | 200 bytes | ~4.4 KB |
| 1024-dim Float16 | 2148 bytes | 200 bytes | ~2.4 KB |
| 1024-dim Int8 | 1124 bytes | 200 bytes | ~1.3 KB |
| 1536-dim Float32 | 6244 bytes | 200 bytes | ~6.4 KB |
| 1536-dim Float16 | 3172 bytes | 200 bytes | ~3.4 KB |

**Breakdown** (1024-dim Float32):
```
Vector Key: ~60 bytes
  - Tuple("myapp", "embedding", "vector", "mlx-embed-1024-v1", "triple:12345")

Vector Value: ~4196 bytes
  - Header: 4 bytes (version, flags, metadata length)
  - Vector data: 4096 bytes (1024 × 4 bytes)
  - Metadata JSON: ~96 bytes ({"subject": "uri", "predicate": "uri", "object": "uri"})

Source Index Key: ~80 bytes
  - Tuple("myapp", "embedding", "index", "source", "triple", "mlx-embed-1024-v1", "triple:12345")

Timestamp Index Key: ~72 bytes
  - Tuple("myapp", "embedding", "index", "timestamp", 1730332800000, "mlx-embed-1024-v1", "triple:12345")

Total: ~4408 bytes (~4.3 KB)
```

### Corpus Size Estimates

| Embedding Count | Float32 (1024-dim) | Float16 (1024-dim) | Int8 (1024-dim) |
|-----------------|--------------------|--------------------|-----------------|
| 10,000 | 44 MB | 24 MB | 13 MB |
| 100,000 | 440 MB | 240 MB | 130 MB |
| 1,000,000 | 4.4 GB | 2.4 GB | 1.3 GB |
| 10,000,000 | 44 GB | 24 GB | 13 GB |

**FDB Capacity**: Petabyte scale (storage is not the bottleneck)

### Compression Tradeoffs

| Encoding | Space Savings | Precision Loss | Search Speed | Use Case |
|----------|---------------|----------------|--------------|----------|
| Float32 | Baseline (0%) | None | Baseline | Production (default) |
| Float16 | 50% | Minimal (~0.001) | 1.2x faster | Large corpora (>1M vectors) |
| Int8 | 75% | Moderate (~0.008) | 2x faster | Very large corpora (>10M vectors) |

**Precision Loss Example** (cosine similarity):

```
Original (Float32):
  Query: [0.5234, -0.8172, 0.2341, ...]
  Corpus: [0.5189, -0.8203, 0.2298, ...]
  Similarity: 0.9823

Float16:
  Query: [0.5234, -0.8174, 0.2341, ...]
  Corpus: [0.5190, -0.8203, 0.2299, ...]
  Similarity: 0.9821  (delta: -0.0002)

Int8 (scale=127):
  Query: [0.5197, -0.8189, 0.2362, ...]
  Corpus: [0.5197, -0.8189, 0.2283, ...]
  Similarity: 0.9815  (delta: -0.0008)
```

### Cleanup Strategies for Old Versions

**Strategy 1: Lazy Deletion**

```swift
actor EmbeddingCleanup {
    func cleanupDeprecatedModel(modelName: String) async throws {
        try await database.withTransaction { transaction in
            // Delete all embeddings for model
            let beginKey = Tuple(rootPrefix, "embedding", "vector", modelName).encode()
            let endKey = Tuple(rootPrefix, "embedding", "vector", modelName, "\u{FFFF}").encode()
            transaction.clearRange(beginKey: beginKey, endKey: endKey)

            // Delete source index
            let sourceBegin = Tuple(rootPrefix, "embedding", "index", "source").encode()
            let sourceEnd = Tuple(rootPrefix, "embedding", "index", "source", "\u{FFFF}").encode()
            // Note: Need to filter by model in application logic

            // Delete timestamp index
            let tsBegin = Tuple(rootPrefix, "embedding", "index", "timestamp").encode()
            let tsEnd = Tuple(rootPrefix, "embedding", "index", "timestamp", "\u{FFFF}").encode()

            // Delete statistics
            let statsBegin = Tuple(rootPrefix, "embedding", "stats", modelName).encode()
            let statsEnd = Tuple(rootPrefix, "embedding", "stats", modelName, "\u{FFFF}").encode()
            transaction.clearRange(beginKey: statsBegin, endKey: statsEnd)
        }
    }
}
```

**Strategy 2: Incremental Deletion**

```swift
// Delete in batches to avoid long transactions
func incrementalDelete(modelName: String, batchSize: Int = 1000) async throws {
    var hasMore = true

    while hasMore {
        hasMore = try await database.withTransaction { transaction in
            var count = 0
            let beginKey = Tuple(rootPrefix, "embedding", "vector", modelName).encode()
            let endKey = Tuple(rootPrefix, "embedding", "vector", modelName, "\u{FFFF}").encode()

            let sequence = transaction.getRange(
                beginSelector: .firstGreaterOrEqual(beginKey),
                endSelector: .firstGreaterThan(endKey),
                limit: batchSize
            )

            for try await (key, _) in sequence {
                transaction.clear(key: key)
                count += 1
            }

            return count == batchSize  // More to delete
        }
    }
}
```

---

## Migration Strategies

### Version Upgrades

**Scenario**: Migrate from Float32 to Float16 encoding

```swift
actor EncodingMigration {
    func migrateToFloat16(model: String) async throws {
        let beginKey = Tuple(rootPrefix, "embedding", "vector", model).encode()
        let endKey = Tuple(rootPrefix, "embedding", "vector", model, "\u{FFFF}").encode()

        var batch: [(key: FDB.Bytes, value: FDB.Bytes)] = []
        let batchSize = 500

        try await database.withTransaction { transaction in
            let sequence = transaction.getRange(
                beginSelector: .firstGreaterOrEqual(beginKey),
                endSelector: .firstGreaterThan(endKey),
                snapshot: true
            )

            for try await (key, oldValue) in sequence {
                // Decode Float32
                let (vector, metadata, _) = try encoder.decode(oldValue)

                // Re-encode as Float16
                let newValue = try encoder.encode(
                    vector: vector,
                    metadata: metadata,
                    encoding: .float16,
                    normalized: true
                )

                batch.append((key, newValue))

                // Commit batch
                if batch.count >= batchSize {
                    for (k, v) in batch {
                        transaction.setValue(v, for: k)
                    }
                    batch.removeAll()
                }
            }

            // Commit remaining
            for (k, v) in batch {
                transaction.setValue(v, for: k)
            }
        }
    }
}
```

### Data Migration Between Models

**Scenario**: Migrate embeddings from v1 to v2 model

```swift
actor ModelMigration {
    func migrate(
        from oldModel: String,
        to newModel: String,
        generator: EmbeddingGenerator
    ) async throws {
        // 1. Get all embeddings for old model
        let oldEmbeddings = try await listByModel(oldModel, limit: Int.max)

        // 2. Create migration job
        let job = RegenerationJob(
            sourceModel: oldModel,
            targetModel: newModel,
            totalCount: oldEmbeddings.count
        )
        let jobKey = Tuple(rootPrefix, "embedding", "job", job.id).encode()
        try await database.withTransaction { transaction in
            let jobValue = try JSONEncoder().encode(job)
            transaction.setValue(jobValue, for: jobKey)
        }

        // 3. Process in batches
        for batch in oldEmbeddings.chunked(into: 100) {
            // Fetch source data (triple text, class description, etc.)
            let sourceTexts = try await fetchSourceTexts(for: batch)

            // Generate new embeddings
            let newVectors = try await generator.generateBatch(texts: sourceTexts)

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

            // Save batch
            try await saveBatch(newRecords)

            // Update job progress
            try await updateJobProgress(jobId: job.id, increment: batch.count)
        }

        // 4. Mark job complete
        try await completeJob(jobId: job.id)
    }
}
```

### Schema Evolution

**Scenario**: Add new index type without downtime

```swift
// Old schema: Only source type index
// New schema: Add timestamp index

actor SchemaEvolution {
    func addTimestampIndex(model: String) async throws {
        let beginKey = Tuple(rootPrefix, "embedding", "vector", model).encode()
        let endKey = Tuple(rootPrefix, "embedding", "vector", model, "\u{FFFF}").encode()

        try await database.withTransaction { transaction in
            let sequence = transaction.getRange(
                beginSelector: .firstGreaterOrEqual(beginKey),
                endSelector: .firstGreaterThan(endKey),
                snapshot: true
            )

            for try await (key, value) in sequence {
                let elements = try Tuple.decode(from: key)
                let id = elements[4] as! String

                // Decode to get creation timestamp
                let (_, metadata, _) = try encoder.decode(value)

                // Assume timestamp stored in metadata
                guard let timestampStr = metadata?["createdAt"],
                      let timestamp = Int64(timestampStr) else {
                    continue
                }

                // Create timestamp index entry
                let indexKey = Tuple(
                    rootPrefix, "embedding", "index", "timestamp",
                    timestamp, model, id
                ).encode()
                transaction.setValue(FDB.Bytes(), for: indexKey)
            }
        }
    }
}
```

---

## Concrete Examples

### Example 1: Complete Storage Flow

**Scenario**: Store a triple embedding from generation to persistence

```swift
// 1. Generate embedding
let triple = Triple(
    subject: .uri("http://example.org/person/Alice"),
    predicate: .uri("http://xmlns.com/foaf/0.1/knows"),
    object: .uri("http://example.org/person/Bob")
)

let text = "Alice knows Bob"
let vector = try await generator.generate(text: text)  // [Float] with 1024 dimensions

// 2. Create embedding record
let record = EmbeddingRecord(
    id: "triple:12345",
    vector: vector,
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

// 3. Save to FDB
try await database.withTransaction { transaction in
    // Encode value (header + vector + metadata)
    var valueBytes = FDB.Bytes()
    valueBytes.append(0x01)  // Version
    valueBytes.append(0b00000100)  // Flags: Float32, normalized

    let metadataJSON = try JSONEncoder().encode(record.metadata)
    let metadataLength = UInt16(metadataJSON.count)
    withUnsafeBytes(of: metadataLength.littleEndian) { valueBytes.append(contentsOf: $0) }

    // Vector data (Float32)
    for value in vector {
        withUnsafeBytes(of: value.bitPattern.littleEndian) { valueBytes.append(contentsOf: $0) }
    }

    valueBytes.append(contentsOf: metadataJSON)

    // Primary key
    let vectorKey = Tuple(
        "myapp",                    // rootPrefix
        "embedding",                // namespace
        "vector",                   // subspace
        "mlx-embed-1024-v1",       // model
        "triple:12345"             // id
    ).encode()
    // Binary: [0x02,'m','y','a','p','p',0x00,0x02,'e','m','b','e','d','d','i','n','g',0x00,...]

    transaction.setValue(valueBytes, for: vectorKey)

    // Source type index
    let sourceIndexKey = Tuple(
        "myapp", "embedding", "index", "source",
        "triple", "mlx-embed-1024-v1", "triple:12345"
    ).encode()
    transaction.setValue(FDB.Bytes(), for: sourceIndexKey)

    // Timestamp index
    let timestamp = Int64(record.createdAt.timeIntervalSince1970 * 1000)
    let timestampIndexKey = Tuple(
        "myapp", "embedding", "index", "timestamp",
        timestamp, "mlx-embed-1024-v1", "triple:12345"
    ).encode()
    transaction.setValue(FDB.Bytes(), for: timestampIndexKey)

    // Update vector count (atomic)
    let countKey = Tuple("myapp", "embedding", "stats", "mlx-embed-1024-v1", "vector_count").encode()
    let increment = withUnsafeBytes(of: UInt64(1).littleEndian) { Array($0) }
    transaction.atomicOp(key: countKey, param: increment, mutationType: .add)

    // Update last_updated
    let now = Int64(Date().timeIntervalSince1970 * 1000)
    let timestampBytes = withUnsafeBytes(of: now.littleEndian) { Array($0) }
    let lastUpdatedKey = Tuple("myapp", "embedding", "stats", "mlx-embed-1024-v1", "last_updated").encode()
    transaction.setValue(timestampBytes, for: lastUpdatedKey)
}
```

**Resulting FDB State**:

```
Keys written:
1. ("myapp", "embedding", "vector", "mlx-embed-1024-v1", "triple:12345")
   → [0x01, 0x04, 0x60, 0x00, <4096 bytes vector>, <96 bytes JSON>]

2. ("myapp", "embedding", "index", "source", "triple", "mlx-embed-1024-v1", "triple:12345")
   → []

3. ("myapp", "embedding", "index", "timestamp", 1730332800000, "mlx-embed-1024-v1", "triple:12345")
   → []

Atomic updates:
4. ("myapp", "embedding", "stats", "mlx-embed-1024-v1", "vector_count")
   → [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]  (counter incremented to 1)

5. ("myapp", "embedding", "stats", "mlx-embed-1024-v1", "last_updated")
   → [0x00, 0xB8, 0x4F, 0x93, 0x8F, 0x01, 0x00, 0x00]  (timestamp updated)
```

### Example 2: Batch Storage with Float16 Compression

**Scenario**: Store 100 embeddings with Float16 encoding

```swift
// Generate 100 triple embeddings
let triples = try await tripleStore.list(limit: 100)
let texts = triples.map { "\($0.subject) \($0.predicate) \($0.object)" }
let vectors = try await generator.generateBatch(texts: texts)  // [[Float]]

let records = zip(triples, vectors).map { (triple, vector) in
    EmbeddingRecord(
        id: "triple:\(triple.id)",
        vector: vector,
        model: "mlx-embed-1024-v1",
        dimension: 1024,
        sourceType: .triple
    )
}

// Save batch with Float16 encoding
try await database.withTransaction { transaction in
    for record in records {
        // Encode as Float16 (2 bytes per dimension)
        var valueBytes = FDB.Bytes()
        valueBytes.append(0x01)  // Version
        valueBytes.append(0b00000101)  // Flags: Float16, normalized
        valueBytes.append(contentsOf: [0x02, 0x00])  // Metadata length = 2 ("{}")

        // Convert Float32 → Float16
        for value in record.vector {
            let float16 = Float16(value)
            withUnsafeBytes(of: float16.bitPattern.littleEndian) {
                valueBytes.append(contentsOf: $0)
            }
        }

        valueBytes.append(contentsOf: [UInt8]("{}".utf8))  // Empty metadata

        // Value size: 4 + (1024 × 2) + 2 = 2054 bytes (50% of Float32)

        let key = Tuple(
            "myapp", "embedding", "vector", record.model, record.id
        ).encode()
        transaction.setValue(valueBytes, for: key)

        // Indexes...
    }

    // Atomic update: increment by 100
    let countKey = Tuple("myapp", "embedding", "stats", "mlx-embed-1024-v1", "vector_count").encode()
    let increment = withUnsafeBytes(of: UInt64(100).littleEndian) { Array($0) }
    transaction.atomicOp(key: countKey, param: increment, mutationType: .add)
}

// Total transaction size: 100 × 2.1KB = ~210KB (well under 10MB limit)
```

### Example 3: Range Query with Timestamp Filter

**Scenario**: Find all embeddings created in last 7 days

```swift
let now = Int64(Date().timeIntervalSince1970 * 1000)
let sevenDaysAgo = now - (7 * 24 * 60 * 60 * 1000)

let embeddings = try await database.withTransaction { transaction in
    var results: [EmbeddingRecord] = []

    let beginKey = Tuple(
        "myapp", "embedding", "index", "timestamp",
        sevenDaysAgo, "mlx-embed-1024-v1"
    ).encode()

    let endKey = Tuple(
        "myapp", "embedding", "index", "timestamp",
        now + 1, "mlx-embed-1024-v1", "\u{FFFF}"
    ).encode()

    let sequence = transaction.getRange(
        beginSelector: .firstGreaterOrEqual(beginKey),
        endSelector: .firstGreaterThan(endKey),
        snapshot: true
    )

    for try await (indexKey, _) in sequence {
        // Decode index key to get entity ID
        let elements = try Tuple.decode(from: indexKey)
        // elements = ["myapp", "embedding", "index", "timestamp", <timestamp>, "mlx-embed-1024-v1", "triple:12345"]
        let entityId = elements[6] as! String

        // Fetch actual embedding
        let vectorKey = Tuple(
            "myapp", "embedding", "vector", "mlx-embed-1024-v1", entityId
        ).encode()

        guard let valueBytes = try await transaction.getValue(for: vectorKey, snapshot: true) else {
            continue
        }

        // Decode value
        let version = valueBytes[0]
        let flags = valueBytes[1]
        let encoding = flags & 0b00000011

        let metadataLength = valueBytes[2..<4].withUnsafeBytes {
            $0.load(as: UInt16.self).littleEndian
        }

        let vectorStart = 4
        let metadataStart = valueBytes.count - Int(metadataLength)
        let vectorBytes = FDB.Bytes(valueBytes[vectorStart..<metadataStart])

        let vector: [Float]
        if encoding == 0b00 {  // Float32
            vector = (0..<1024).map { i in
                let offset = i * 4
                let bits = vectorBytes[offset..<offset+4].withUnsafeBytes {
                    $0.load(as: UInt32.self).littleEndian
                }
                return Float(bitPattern: bits)
            }
        } else {  // Float16
            vector = (0..<1024).map { i in
                let offset = i * 2
                let bits = vectorBytes[offset..<offset+2].withUnsafeBytes {
                    $0.load(as: UInt16.self).littleEndian
                }
                return Float(Float16(bitPattern: bits))
            }
        }

        let record = EmbeddingRecord(
            id: entityId,
            vector: vector,
            model: "mlx-embed-1024-v1",
            dimension: 1024,
            sourceType: .triple
        )
        results.append(record)
    }

    return results
}

print("Found \(embeddings.count) embeddings created in last 7 days")
```

---

## Summary

### Key Design Decisions

1. **Tuple Encoding**: All keys use FDB Tuple encoding for lexicographic ordering
2. **Hierarchical Namespacing**: Logical separation via subspace prefixes
3. **Secondary Indexes**: Separate index keys for efficient filtering
4. **Atomic Counters**: Lock-free statistics tracking
5. **Flexible Value Encoding**: Support for Float32/Float16/Int8 compression
6. **Batch Operations**: Optimize for throughput with large batches

### Storage Characteristics

| Metric | Value |
|--------|-------|
| Key size (primary) | ~60 bytes |
| Value size (1024-dim Float32) | ~4.2 KB |
| Index overhead per embedding | ~200 bytes |
| Total storage per embedding | ~4.4 KB |
| Max embeddings per transaction | ~1000 (for 1024-dim Float32) |
| Transaction latency (100 embeddings) | 100-200ms |
| Max corpus size in FDB | Unlimited (petabyte scale) |

### Performance Expectations

| Operation | Latency (p50) | Latency (p99) |
|-----------|---------------|---------------|
| Single embedding save | 5-10ms | 20-30ms |
| Batch save (100 vectors) | 50-100ms | 200-300ms |
| Get embedding (cached) | <1ms | <5ms |
| Get embedding (uncached) | 5-10ms | 20-30ms |
| Range query (1000 results) | 50-100ms | 200-300ms |
| Statistics update (atomic) | 2-5ms | 10-15ms |

---

**Document Version**: 1.0
**Last Updated**: 2025-10-30
**Status**: Complete
