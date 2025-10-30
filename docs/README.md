# fdb-embedding-layer Documentation

## Overview

`fdb-embedding-layer` は、知識構造（Triple）やエンティティ（OntologyClass）をベクトル空間にマッピングし、類似性・意味距離・クラスタリングを実現するセマンティックレイヤーです。

このレイヤーは、fdb-triple-layer と fdb-ontology-layer で定義された構造・意味情報を数値的な「意味空間（Embedding）」に変換・保持し、fdb-knowledge-layer に類似性・意味距離情報を提供します。

---

## Documentation Structure

| Document | Description |
|----------|-------------|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | システムアーキテクチャ、コンポーネント設計、パフォーマンス特性 |
| **[DATA_MODEL.md](DATA_MODEL.md)** | データ構造（EmbeddingRecord, ModelMetadata, SearchResult等） |
| **[STORAGE_LAYOUT.md](STORAGE_LAYOUT.md)** | FoundationDBキーバリューレイアウトとエンコーディング |
| **[API_DESIGN.md](API_DESIGN.md)** | 完全なAPI仕様（EmbeddingStore, ModelManager, SearchEngine） |
| **[INTEGRATION.md](INTEGRATION.md)** | 他レイヤー（Triple, Ontology, Knowledge）との統合パターン |
| **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** | 段階的な実装ロードマップ |

---

## Quick Links

### Getting Started
- [Architecture Overview](ARCHITECTURE.md#overview)
- [Data Models](DATA_MODEL.md#core-models)
- [API Quick Start](API_DESIGN.md#usage-examples)

### Implementation
- [Storage Design](STORAGE_LAYOUT.md#key-structures)
- [Embedding Generation](ARCHITECTURE.md#embedding-generation)
- [Implementation Phases](IMPLEMENTATION_PLAN.md#phase-1-foundation)

### Integration
- [Triple Layer Integration](INTEGRATION.md#triple-layer-integration)
- [Ontology Layer Integration](INTEGRATION.md#ontology-layer-integration)
- [Knowledge Layer Integration](INTEGRATION.md#knowledge-layer-integration)

### Advanced Topics
- [Similarity Search Algorithms](ARCHITECTURE.md#similarity-search)
- [Model Management](API_DESIGN.md#model-management)
- [Performance Optimization](ARCHITECTURE.md#performance)

---

## Key Concepts

### 1. Embedding Components

```
Embedding Layer
├── EmbeddingStore (Storage & Retrieval)
│   ├── Save/Get/Delete embeddings
│   └── Batch operations
├── ModelManager (Model Metadata)
│   ├── Register/Get models
│   └── Model versioning
└── SearchEngine (Similarity Search)
    ├── Cosine similarity
    ├── Inner product
    └── Batch search
```

### 2. Core Operations

| Operation | Description | Document |
|-----------|-------------|----------|
| **Generate** | テキスト/Triple/Entityからベクトル生成 | [API_DESIGN.md](API_DESIGN.md#embedding-generation) |
| **Store** | ベクトルをFDBに永続化 | [API_DESIGN.md](API_DESIGN.md#embedding-storage) |
| **Search** | 類似性検索（K-NN） | [API_DESIGN.md](API_DESIGN.md#similarity-search) |
| **Update** | モデル更新時の再生成 | [API_DESIGN.md](API_DESIGN.md#update-operations) |

### 3. Integration Points

```
fdb-knowledge-layer
      │
      ├─► fdb-triple-layer (triple embeddings)
      ├─► fdb-ontology-layer (entity embeddings)
      └─► fdb-embedding-layer (semantic search)
```

---

## Example Workflow

### 1. Register Embedding Model

```swift
import EmbeddingLayer

let modelManager = ModelManager(database: database, rootPrefix: "myapp")

// Register model
let model = EmbeddingModelMetadata(
    name: "mlx-embed",
    dimension: 1024,
    provider: "Apple MLX",
    description: "Local embedding model"
)
try await modelManager.registerModel(model)
```

### 2. Generate and Store Embeddings

```swift
let embeddingStore = EmbeddingStore(
    database: database,
    rootPrefix: "myapp"
)

// Generate embedding for entity
let entityId = "Person:123"
let vector = try await generateEmbedding(for: "John works at Acme Corp")

// Store embedding
let record = EmbeddingRecord(
    id: entityId,
    vector: vector,
    model: "mlx-embed",
    dimension: 1024,
    sourceType: .entity,
    createdAt: Date()
)
try await embeddingStore.save(record)
```

### 3. Similarity Search

```swift
let searchEngine = SearchEngine(store: embeddingStore)

// Find similar entities
let queryVector = try await generateEmbedding(for: "software engineer")
let results = try await searchEngine.search(
    queryVector: queryVector,
    topK: 10,
    model: "mlx-embed",
    similarityMetric: .cosine
)

for result in results {
    print("ID: \(result.id), Similarity: \(result.score)")
}
```

### 4. Batch Operations

```swift
// Batch embedding generation
let entities = ["Person:123", "Person:456", "Person:789"]
let embeddings = try await embeddingStore.getBatch(ids: entities, model: "mlx-embed")

// Batch search
let queries = [vector1, vector2, vector3]
let batchResults = try await searchEngine.batchSearch(
    queryVectors: queries,
    topK: 5,
    model: "mlx-embed"
)
```

---

## Design Principles

### 1. Model Agnostic
Support multiple embedding models (local MLX, OpenAI, Cohere, etc.)

### 2. Efficient Storage
Optimize FDB storage for high-dimensional vectors

### 3. Scalable Search
Support both in-memory and external vector DB integration

### 4. Actor-Based Concurrency
All components are actors for thread safety

### 5. Separation of Concerns
- **EmbeddingStore**: Storage and retrieval
- **ModelManager**: Model metadata
- **SearchEngine**: Similarity search logic

---

## Performance Targets

| Operation | Target Latency (p99) | Throughput |
|-----------|---------------------|------------|
| Save Embedding | 10-20ms | 1,000-5,000 ops/sec |
| Get Embedding (cached) | <5ms | 100,000+ ops/sec |
| Similarity Search (1000 vectors) | 50-100ms | 100-500 queries/sec |
| Batch Save (100 vectors) | 100-200ms | 500-1,000 batches/sec |

**Note**: Search performance depends on corpus size. For large-scale similarity search (>100K vectors), integration with specialized vector databases (Milvus, Weaviate, etc.) is recommended.

---

## Architecture Highlights

### Phase 1: Foundation (FoundationDB Native)
- Store embeddings in FDB
- Simple brute-force similarity search
- Support up to ~10K-100K vectors

### Phase 2: Hybrid Approach (Recommended)
- FDB for metadata and small corpus
- External vector DB for large-scale ANN search
- Maintain consistency between both systems

### Phase 3: Advanced (Optional)
- Approximate Nearest Neighbor (ANN) algorithms
- Index structures (HNSW, IVF)
- Distributed search

---

## Requirements

- **macOS 15.0+**
- **Swift 6.0+** (Swift 5 language mode)
- **FoundationDB 7.1.0+**
- **Embedding Model** (mlx-embed, OpenAI API, etc.)

---

## Project Status

**Current Phase**: Design (v0.1)

**Roadmap**:
- ✅ Phase 0: Design and documentation
- ⏳ Phase 1: Core foundation (EmbeddingRecord, ModelMetadata)
- ⏳ Phase 2: Storage layer (EmbeddingStore)
- ⏳ Phase 3: Model management (ModelManager)
- ⏳ Phase 4: Search engine (SearchEngine with brute-force)
- ⏳ Phase 5: Integration and testing
- ⏳ Phase 6: Advanced features (ANN, external VectorDB integration)

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed roadmap.

---

## Reading Order

### For Users (Application Developers)

1. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand the system
2. [API_DESIGN.md](API_DESIGN.md) - Learn the API
3. [INTEGRATION.md](INTEGRATION.md) - Integrate with your stack

### For Contributors (Library Developers)

1. [ARCHITECTURE.md](ARCHITECTURE.md) - System design
2. [DATA_MODEL.md](DATA_MODEL.md) - Data structures
3. [STORAGE_LAYOUT.md](STORAGE_LAYOUT.md) - FDB layout
4. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Development phases

### For Researchers (ML/IR)

1. [ARCHITECTURE.md](ARCHITECTURE.md#similarity-search) - Search algorithms
2. [API_DESIGN.md](API_DESIGN.md#search-engine) - Search API
3. [INTEGRATION.md](INTEGRATION.md) - External vector DB integration

---

## Key Features

### Embedding Generation
- Support for multiple models
- Text, Triple, Entity embeddings
- Batch generation support

### Storage
- Efficient FDB encoding
- Compression options
- Vector indexing

### Search
- Cosine similarity
- Inner product (dot product)
- Euclidean distance
- Configurable top-K retrieval

### Model Management
- Model registration and versioning
- Model metadata tracking
- Multi-model support

### Integration
- Seamless integration with Triple Layer
- Ontology-aware embeddings
- Knowledge Layer API

---

## Contributing

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for development roadmap and contribution guidelines.

---

## License

MIT License (to be added)

---

## References

### External Documentation
- [FoundationDB Documentation](https://apple.github.io/foundationdb/)
- [FAISS: Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)
- [Milvus: Open-Source Vector Database](https://milvus.io/)

### Related Projects
- [fdb-triple-layer](../../fdb-triple-layer/) - Triple storage
- [fdb-ontology-layer](../../fdb-ontology-layer/) - Ontology management
- [fdb-knowledge-layer](../../fdb-knowledge-layer/) - Unified knowledge API

### Research Papers
- "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs" (HNSW)
- "Product Quantization for Nearest Neighbor Search" (PQ)
- "Learning to Route in Similarity Graphs" (NSW)

---

**Documentation Version**: 1.0
**Last Updated**: 2025-10-30
**Status**: Draft
