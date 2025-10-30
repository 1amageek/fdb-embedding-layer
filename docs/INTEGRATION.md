# fdb-embedding-layer Integration Guide

## Overview

This document provides comprehensive integration patterns for `fdb-embedding-layer` with other layers in the knowledge management stack. The embedding layer transforms structured knowledge into dense vector representations, enabling semantic search, similarity analysis, and clustering operations across the entire knowledge base.

**Integration Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                   fdb-knowledge-layer                       │
│  (Unified API, hybrid search, complex reasoning)            │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼────────┐ ┌──────▼──────┐ ┌───────▼────────┐
│ fdb-triple-    │ │ fdb-ontology-│ │ fdb-embedding- │
│ layer          │─┤ layer        │◄┤ layer          │
│ (Triple Store) │ │ (Schema)     │ │ (Vector Store) │
└────────────────┘ └──────────────┘ └────────────────┘
                          │
                ┌─────────▼──────────┐
                │   FoundationDB     │
                │ (Distributed KVS)  │
                └────────────────────┘
```

---

## Table of Contents

1. [Integration with fdb-triple-layer](#1-integration-with-fdb-triple-layer)
2. [Integration with fdb-ontology-layer](#2-integration-with-fdb-ontology-layer)
3. [Integration with fdb-knowledge-layer](#3-integration-with-fdb-knowledge-layer)
4. [External Vector Database Integration](#4-external-vector-database-integration)
5. [Embedding Model Integration](#5-embedding-model-integration)
6. [Real-World Use Cases](#6-real-world-use-cases)
7. [Best Practices](#7-best-practices)
8. [Migration Patterns](#8-migration-patterns)

---

## 1. Integration with fdb-triple-layer

The triple layer stores RDF-like triples (subject-predicate-object). The embedding layer generates vector embeddings for triples to enable semantic search over the knowledge graph.

### 1.1 Triple-to-Text Conversion Strategies

#### Strategy A: Simple Concatenation

Convert triple components to plain text and generate embedding.

```swift
import TripleLayer
import EmbeddingLayer

actor TripleEmbeddingGenerator {
    private let tripleStore: TripleStore
    private let embeddingStore: EmbeddingStore
    private let generator: EmbeddingGenerator

    init(
        database: any DatabaseProtocol,
        rootPrefix: String,
        generator: EmbeddingGenerator
    ) async throws {
        self.tripleStore = try await TripleStore(database: database, rootPrefix: rootPrefix)
        self.embeddingStore = EmbeddingStore(database: database, rootPrefix: rootPrefix)
        self.generator = generator
    }

    /// Generate embedding using simple concatenation
    func generateSimpleEmbedding(for triple: Triple) async throws -> EmbeddingRecord {
        // Convert triple to text: "subject predicate object"
        let text = "\(triple.subject.stringValue) \(triple.predicate.stringValue) \(triple.object.stringValue)"

        // Generate embedding
        let vector = try await generator.generate(text: text)

        // Create embedding record
        let record = EmbeddingRecord(
            id: "triple:\(triple.id)",
            vector: vector,
            model: generator.modelMetadata.name,
            dimension: generator.modelMetadata.dimension,
            sourceType: .triple,
            metadata: [
                "subject": triple.subject.stringValue,
                "predicate": triple.predicate.stringValue,
                "object": triple.object.stringValue
            ],
            createdAt: Date()
        )

        return record
    }
}

extension Value {
    var stringValue: String {
        switch self {
        case .uri(let uri):
            return uri
        case .text(let text, _):
            return text
        case .integer(let int):
            return String(int)
        case .float(let float):
            return String(float)
        case .boolean(let bool):
            return String(bool)
        case .binary(let data):
            return data.base64EncodedString()
        }
    }
}
```

**Pros**: Simple, fast, no dependencies
**Cons**: Loses semantic structure, URIs may not be meaningful

---

#### Strategy B: Contextual with Labels

Fetch human-readable labels for URIs to generate more meaningful text.

```swift
actor ContextualTripleEmbedder {
    private let tripleStore: TripleStore
    private let embeddingStore: EmbeddingStore
    private let generator: EmbeddingGenerator

    /// Generate embedding with human-readable labels
    func generateContextualEmbedding(for triple: Triple) async throws -> EmbeddingRecord {
        // Fetch labels for URIs
        let subjectLabel = await fetchLabel(for: triple.subject)
        let predicateLabel = await fetchLabel(for: triple.predicate)
        let objectLabel = await fetchLabel(for: triple.object)

        // Create natural language sentence
        let text = "\(subjectLabel) \(predicateLabel) \(objectLabel)"

        // Example: "Alice knows Bob" instead of
        // "http://example.org/person/Alice http://xmlns.com/foaf/0.1/knows http://example.org/person/Bob"

        let vector = try await generator.generate(text: text)

        return EmbeddingRecord(
            id: "triple:\(triple.id)",
            vector: vector,
            model: generator.modelMetadata.name,
            dimension: generator.modelMetadata.dimension,
            sourceType: .triple,
            metadata: [
                "subject_uri": triple.subject.stringValue,
                "predicate_uri": triple.predicate.stringValue,
                "object_uri": triple.object.stringValue,
                "text": text
            ],
            createdAt: Date()
        )
    }

    private func fetchLabel(for value: Value) async -> String {
        guard case .uri(let uri) = value else {
            return value.stringValue
        }

        // Try to fetch rdfs:label or similar
        let labelTriples = try? await tripleStore.query(
            subject: value,
            predicate: .uri("http://www.w3.org/2000/01/rdf-schema#label"),
            object: nil
        )

        if let labelTriple = labelTriples?.first,
           case .text(let label, _) = labelTriple.object {
            return label
        }

        // Fall back to extracting from URI
        return uri.components(separatedBy: "/").last ?? uri
    }
}
```

**Pros**: More meaningful text, better semantic representation
**Cons**: Requires label lookup (slower), labels may not exist

---

#### Strategy C: Graph Context (Advanced)

Include 1-hop neighborhood to provide richer context.

```swift
actor GraphContextualEmbedder {
    private let tripleStore: TripleStore
    private let embeddingStore: EmbeddingStore
    private let generator: EmbeddingGenerator

    /// Generate embedding with graph context
    func generateGraphEmbedding(for triple: Triple) async throws -> EmbeddingRecord {
        // Fetch 1-hop neighbors of subject
        let subjectNeighbors = try await tripleStore.query(
            subject: triple.subject,
            predicate: nil,
            object: nil
        )

        // Fetch 1-hop neighbors of object (if URI)
        var objectNeighbors: [Triple] = []
        if case .uri = triple.object {
            objectNeighbors = try await tripleStore.query(
                subject: triple.object,
                predicate: nil,
                object: nil
            )
        }

        // Build context text
        var contextParts: [String] = []

        // Main triple
        let mainLabel = await fetchLabel(for: triple.subject)
        let predLabel = await fetchLabel(for: triple.predicate)
        let objLabel = await fetchLabel(for: triple.object)
        contextParts.append("\(mainLabel) \(predLabel) \(objLabel)")

        // Subject context
        for neighbor in subjectNeighbors.prefix(5) {
            let pLabel = await fetchLabel(for: neighbor.predicate)
            let oLabel = await fetchLabel(for: neighbor.object)
            contextParts.append("\(mainLabel) \(pLabel) \(oLabel)")
        }

        // Object context
        for neighbor in objectNeighbors.prefix(5) {
            let pLabel = await fetchLabel(for: neighbor.predicate)
            let oLabel = await fetchLabel(for: neighbor.object)
            contextParts.append("\(objLabel) \(pLabel) \(oLabel)")
        }

        let text = contextParts.joined(separator: ". ")
        let vector = try await generator.generate(text: text)

        return EmbeddingRecord(
            id: "triple:\(triple.id)",
            vector: vector,
            model: generator.modelMetadata.name,
            dimension: generator.modelMetadata.dimension,
            sourceType: .triple,
            metadata: [
                "subject_uri": triple.subject.stringValue,
                "predicate_uri": triple.predicate.stringValue,
                "object_uri": triple.object.stringValue,
                "text": text,
                "context_size": String(contextParts.count)
            ],
            createdAt: Date()
        )
    }

    private func fetchLabel(for value: Value) async -> String {
        // Same as Strategy B
        guard case .uri(let uri) = value else {
            return value.stringValue
        }
        return uri.components(separatedBy: "/").last ?? uri
    }
}
```

**Pros**: Rich semantic context, better for complex queries
**Cons**: Much slower, larger context may dilute main triple meaning

---

### 1.2 Batch Embedding Generation for Triples

For large-scale triple ingestion, batch processing is essential.

```swift
actor BatchTripleEmbedder {
    private let tripleStore: TripleStore
    private let embeddingStore: EmbeddingStore
    private let generator: EmbeddingGenerator

    /// Generate embeddings for all triples in batch
    func generateEmbeddingsForAllTriples(batchSize: Int = 100) async throws {
        var offset = 0
        var processedCount = 0

        while true {
            // Fetch batch of triples
            let triples = try await tripleStore.query(
                subject: nil,
                predicate: nil,
                object: nil,
                limit: batchSize,
                offset: offset
            )

            if triples.isEmpty {
                break
            }

            // Convert triples to text batch
            let texts = triples.map { triple in
                "\(triple.subject.stringValue) \(triple.predicate.stringValue) \(triple.object.stringValue)"
            }

            // Generate embeddings in batch (much faster)
            let vectors = try await generator.generateBatch(texts: texts)

            // Create embedding records
            let records = zip(triples, vectors).map { (triple, vector) in
                EmbeddingRecord(
                    id: "triple:\(triple.id)",
                    vector: vector,
                    model: generator.modelMetadata.name,
                    dimension: generator.modelMetadata.dimension,
                    sourceType: .triple,
                    metadata: [
                        "subject": triple.subject.stringValue,
                        "predicate": triple.predicate.stringValue,
                        "object": triple.object.stringValue
                    ],
                    createdAt: Date()
                )
            }

            // Save batch
            try await embeddingStore.saveBatch(records)

            processedCount += triples.count
            offset += batchSize

            print("Processed \(processedCount) triples...")
        }

        print("Completed: generated embeddings for \(processedCount) triples")
    }
}
```

---

### 1.3 Search Triples by Semantic Similarity

Once embeddings are generated, search for semantically similar triples.

```swift
actor SemanticTripleSearch {
    private let tripleStore: TripleStore
    private let embeddingStore: EmbeddingStore
    private let searchEngine: SearchEngine
    private let generator: EmbeddingGenerator

    /// Search for triples semantically similar to query text
    func searchTriples(
        query: String,
        topK: Int = 10,
        similarityThreshold: Float = 0.7
    ) async throws -> [(Triple, Float)] {
        // 1. Generate query embedding
        let queryVector = try await generator.generate(text: query)

        // 2. Search for similar embeddings
        let results = try await searchEngine.search(
            queryVector: queryVector,
            topK: topK,
            model: generator.modelMetadata.name,
            similarityMetric: .cosine
        )

        // 3. Filter by threshold and fetch triples
        var tripleResults: [(Triple, Float)] = []

        for result in results where result.score >= similarityThreshold {
            // Extract triple ID from embedding ID
            guard let tripleIdStr = result.id.split(separator: ":").last,
                  let tripleId = Int64(tripleIdStr) else {
                continue
            }

            // Fetch triple from store
            if let triple = try await tripleStore.getTriple(id: tripleId) {
                tripleResults.append((triple, result.score))
            }
        }

        return tripleResults
    }

    /// Find similar triples to a given triple
    func findSimilarTriples(to triple: Triple, topK: Int = 10) async throws -> [(Triple, Float)] {
        // 1. Get embedding for this triple
        guard let embedding = try await embeddingStore.get(
            id: "triple:\(triple.id)",
            model: generator.modelMetadata.name
        ) else {
            throw EmbeddingError.vectorNotFound("triple:\(triple.id)")
        }

        // 2. Search using this embedding
        let results = try await searchEngine.search(
            queryVector: embedding.vector,
            topK: topK + 1,  // +1 because it will include itself
            model: generator.modelMetadata.name,
            similarityMetric: .cosine
        )

        // 3. Exclude the original triple and fetch others
        var similarTriples: [(Triple, Float)] = []

        for result in results where result.id != "triple:\(triple.id)" {
            guard let tripleIdStr = result.id.split(separator: ":").last,
                  let tripleId = Int64(tripleIdStr) else {
                continue
            }

            if let similarTriple = try await tripleStore.getTriple(id: tripleId) {
                similarTriples.append((similarTriple, result.score))
            }
        }

        return Array(similarTriples.prefix(topK))
    }
}
```

---

### 1.4 Example Workflow: Store Triple → Generate Embedding → Semantic Search

```swift
import FoundationDB
import TripleLayer
import EmbeddingLayer

// Initialize FoundationDB
try await FDBClient.initialize()
let database = try FDBClient.openDatabase()

// Initialize stores
let tripleStore = try await TripleStore(database: database, rootPrefix: "myapp")
let embeddingStore = EmbeddingStore(database: database, rootPrefix: "myapp")
let modelManager = ModelManager(database: database, rootPrefix: "myapp")

// Register embedding model
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

// Create generator
let generator = MLXEmbeddingGenerator(model: model)

// 1. Store triple
let triple = Triple(
    id: 12345,
    subject: .uri("http://example.org/person/Alice"),
    predicate: .uri("http://xmlns.com/foaf/0.1/knows"),
    object: .uri("http://example.org/person/Bob")
)
try await tripleStore.insert(triple)

// 2. Generate and store embedding
let embedder = TripleEmbeddingGenerator(
    database: database,
    rootPrefix: "myapp",
    generator: generator
)
let embedding = try await embedder.generateSimpleEmbedding(for: triple)
try await embeddingStore.save(embedding)

// 3. Semantic search
let searchEngine = SearchEngine(store: embeddingStore)
let searcher = SemanticTripleSearch(
    tripleStore: tripleStore,
    embeddingStore: embeddingStore,
    searchEngine: searchEngine,
    generator: generator
)

let results = try await searcher.searchTriples(
    query: "people who know each other",
    topK: 5
)

for (triple, score) in results {
    print("Score: \(score)")
    print("Triple: \(triple.subject) \(triple.predicate) \(triple.object)")
}
```

---

## 2. Integration with fdb-ontology-layer

The ontology layer defines semantic structure (classes, properties, constraints). The embedding layer generates embeddings for ontology classes to enable semantic class similarity and concept discovery.

### 2.1 Generating Embeddings for Ontology Classes

Convert class definitions to rich text representations.

```swift
import OntologyLayer
import EmbeddingLayer

actor OntologyEmbeddingGenerator {
    private let ontologyStore: OntologyStore
    private let embeddingStore: EmbeddingStore
    private let generator: EmbeddingGenerator

    init(
        database: any DatabaseProtocol,
        rootPrefix: String,
        generator: EmbeddingGenerator
    ) {
        self.ontologyStore = OntologyStore(database: database, rootPrefix: rootPrefix)
        self.embeddingStore = EmbeddingStore(database: database, rootPrefix: rootPrefix)
        self.generator = generator
    }

    /// Generate embedding for ontology class
    func generateClassEmbedding(for cls: OntologyClass) async throws -> EmbeddingRecord {
        // Build rich text representation
        var parts: [String] = []

        // Class name
        parts.append("Class: \(cls.name)")

        // Parent class
        if let parent = cls.parent {
            parts.append("Inherits from: \(parent)")
        }

        // Description
        if let description = cls.description {
            parts.append("Description: \(description)")
        }

        // Properties
        if !cls.properties.isEmpty {
            parts.append("Properties: \(cls.properties.joined(separator: ", "))")
        }

        // Constraints
        if !cls.constraints.isEmpty {
            let constraintTexts = cls.constraints.map { constraint in
                switch constraint {
                case .propertyRequired(let prop):
                    return "requires \(prop)"
                case .propertyType(let prop, let type):
                    return "\(prop) must be \(type)"
                case .propertyRange(let prop, let range):
                    return "\(prop) must be in range \(range)"
                }
            }
            parts.append("Constraints: \(constraintTexts.joined(separator: ", "))")
        }

        let text = parts.joined(separator: ". ")

        // Generate embedding
        let vector = try await generator.generate(text: text)

        // Create embedding record
        let record = EmbeddingRecord(
            id: "class:\(cls.name)",
            vector: vector,
            model: generator.modelMetadata.name,
            dimension: generator.modelMetadata.dimension,
            sourceType: .entity,
            metadata: [
                "className": cls.name,
                "parent": cls.parent ?? "",
                "description": cls.description ?? "",
                "text": text
            ],
            createdAt: Date()
        )

        return record
    }

    /// Generate embeddings for all classes
    func generateAllClassEmbeddings() async throws {
        let classes = try await ontologyStore.allClasses()

        var records: [EmbeddingRecord] = []

        for cls in classes {
            let record = try await generateClassEmbedding(for: cls)
            records.append(record)
        }

        // Save batch
        try await embeddingStore.saveBatch(records)

        print("Generated embeddings for \(records.count) classes")
    }
}
```

---

### 2.2 Entity Description to Embedding

Generate embeddings for entity instances based on their class definition.

```swift
actor EntityEmbeddingGenerator {
    private let ontologyStore: OntologyStore
    private let tripleStore: TripleStore
    private let embeddingStore: EmbeddingStore
    private let generator: EmbeddingGenerator

    /// Generate embedding for entity instance
    func generateEntityEmbedding(
        entityURI: String,
        className: String
    ) async throws -> EmbeddingRecord {
        // 1. Get class definition
        guard let cls = try await ontologyStore.getClass(named: className) else {
            throw OntologyError.classNotFound(className)
        }

        // 2. Fetch entity properties from triples
        let entityValue = Value.uri(entityURI)
        let triples = try await tripleStore.query(
            subject: entityValue,
            predicate: nil,
            object: nil
        )

        // 3. Build entity description
        var parts: [String] = []

        // Entity type
        parts.append("\(extractLabel(from: entityURI)) is a \(cls.name)")

        // Add description if exists
        if let description = cls.description {
            parts.append(description)
        }

        // Add property values
        for triple in triples {
            let predicate = extractLabel(from: triple.predicate.stringValue)
            let object = triple.object.stringValue
            parts.append("has \(predicate): \(object)")
        }

        let text = parts.joined(separator: ". ")

        // 4. Generate embedding
        let vector = try await generator.generate(text: text)

        // 5. Create embedding record
        let record = EmbeddingRecord(
            id: "entity:\(entityURI.hashValue)",
            vector: vector,
            model: generator.modelMetadata.name,
            dimension: generator.modelMetadata.dimension,
            sourceType: .entity,
            metadata: [
                "entityURI": entityURI,
                "className": className,
                "text": text
            ],
            createdAt: Date()
        )

        return record
    }

    private func extractLabel(from uri: String) -> String {
        return uri.components(separatedBy: "/").last ?? uri
    }
}
```

---

### 2.3 Class Hierarchy Embeddings

Generate embeddings that encode class hierarchy information.

```swift
actor HierarchicalClassEmbedder {
    private let ontologyStore: OntologyStore
    private let embeddingStore: EmbeddingStore
    private let generator: EmbeddingGenerator

    /// Generate embedding with full hierarchy context
    func generateHierarchicalEmbedding(for cls: OntologyClass) async throws -> EmbeddingRecord {
        // 1. Get ancestors
        let ancestors = try await ontologyStore.getSuperclasses(of: cls.name)

        // 2. Get siblings (other subclasses of parent)
        var siblings: [String] = []
        if let parent = cls.parent {
            let subclasses = try await ontologyStore.getSubclasses(of: parent)
            siblings = subclasses.filter { $0 != cls.name }
        }

        // 3. Get children
        let children = try await ontologyStore.getSubclasses(of: cls.name)

        // 4. Build hierarchical description
        var parts: [String] = []

        parts.append("Class: \(cls.name)")

        if let description = cls.description {
            parts.append(description)
        }

        if !ancestors.isEmpty {
            parts.append("Hierarchy: \(ancestors.joined(separator: " > ")) > \(cls.name)")
        }

        if !siblings.isEmpty {
            parts.append("Similar to: \(siblings.joined(separator: ", "))")
        }

        if !children.isEmpty {
            parts.append("Specializations: \(children.joined(separator: ", "))")
        }

        if !cls.properties.isEmpty {
            parts.append("Properties: \(cls.properties.joined(separator: ", "))")
        }

        let text = parts.joined(separator: ". ")

        // 5. Generate embedding
        let vector = try await generator.generate(text: text)

        // 6. Create record
        let record = EmbeddingRecord(
            id: "class:\(cls.name):hierarchical",
            vector: vector,
            model: generator.modelMetadata.name,
            dimension: generator.modelMetadata.dimension,
            sourceType: .entity,
            metadata: [
                "className": cls.name,
                "hierarchy": ancestors.joined(separator: " > "),
                "text": text,
                "mode": "hierarchical"
            ],
            createdAt: Date()
        )

        return record
    }
}
```

---

### 2.4 Semantic Class Similarity

Find similar classes based on semantic embeddings.

```swift
actor SemanticClassSearch {
    private let ontologyStore: OntologyStore
    private let embeddingStore: EmbeddingStore
    private let searchEngine: SearchEngine
    private let generator: EmbeddingGenerator

    /// Find classes similar to a concept
    func findSimilarClasses(
        toConcept concept: String,
        topK: Int = 5
    ) async throws -> [(OntologyClass, Float)] {
        // 1. Generate embedding for concept
        let queryVector = try await generator.generate(text: concept)

        // 2. Search for similar class embeddings
        let results = try await searchEngine.search(
            queryVector: queryVector,
            topK: topK,
            model: generator.modelMetadata.name,
            similarityMetric: .cosine,
            filter: SearchFilter(sourceTypes: [.entity], metadata: nil, createdAfter: nil, createdBefore: nil)
        )

        // 3. Fetch class definitions
        var classResults: [(OntologyClass, Float)] = []

        for result in results {
            guard let className = result.id.split(separator: ":").last else {
                continue
            }

            if let cls = try await ontologyStore.getClass(named: String(className)) {
                classResults.append((cls, result.score))
            }
        }

        return classResults
    }

    /// Find classes similar to a given class
    func findSimilarClasses(
        to cls: OntologyClass,
        topK: Int = 5
    ) async throws -> [(OntologyClass, Float)] {
        // 1. Get embedding for this class
        guard let embedding = try await embeddingStore.get(
            id: "class:\(cls.name)",
            model: generator.modelMetadata.name
        ) else {
            throw EmbeddingError.vectorNotFound("class:\(cls.name)")
        }

        // 2. Search using this embedding
        let results = try await searchEngine.search(
            queryVector: embedding.vector,
            topK: topK + 1,
            model: generator.modelMetadata.name,
            similarityMetric: .cosine,
            filter: SearchFilter(sourceTypes: [.entity], metadata: nil, createdAfter: nil, createdBefore: nil)
        )

        // 3. Exclude the original class
        var similarClasses: [(OntologyClass, Float)] = []

        for result in results where result.id != "class:\(cls.name)" {
            guard let className = result.id.split(separator: ":").last else {
                continue
            }

            if let similarClass = try await ontologyStore.getClass(named: String(className)) {
                similarClasses.append((similarClass, result.score))
            }
        }

        return Array(similarClasses.prefix(topK))
    }
}
```

---

### 2.5 Example Workflow: Define Class → Generate Embedding → Find Similar Concepts

```swift
import FoundationDB
import OntologyLayer
import EmbeddingLayer

// Initialize
try await FDBClient.initialize()
let database = try FDBClient.openDatabase()

let ontologyStore = OntologyStore(database: database, rootPrefix: "myapp")
let embeddingStore = EmbeddingStore(database: database, rootPrefix: "myapp")
let modelManager = ModelManager(database: database, rootPrefix: "myapp")

// Register model
let model = EmbeddingModelMetadata(
    name: "mlx-embed-1024-v1",
    version: "1.0.0",
    dimension: 1024,
    provider: "Apple MLX",
    modelType: "Sentence-BERT",
    normalized: true
)
try await modelManager.registerModel(model)

let generator = MLXEmbeddingGenerator(model: model)

// 1. Define ontology class
let personClass = OntologyClass(
    name: "Person",
    parent: "Entity",
    description: "A human being with name, age, and social relationships",
    properties: ["name", "age", "knows", "worksFor"],
    constraints: [
        .propertyRequired("name"),
        .propertyType("age", "integer")
    ]
)
try await ontologyStore.defineClass(personClass)

// 2. Generate embedding for class
let classEmbedder = OntologyEmbeddingGenerator(
    database: database,
    rootPrefix: "myapp",
    generator: generator
)
let classEmbedding = try await classEmbedder.generateClassEmbedding(for: personClass)
try await embeddingStore.save(classEmbedding)

// 3. Find similar concepts
let searchEngine = SearchEngine(store: embeddingStore)
let classSearcher = SemanticClassSearch(
    ontologyStore: ontologyStore,
    embeddingStore: embeddingStore,
    searchEngine: searchEngine,
    generator: generator
)

let similarClasses = try await classSearcher.findSimilarClasses(
    toConcept: "software developer who works at a tech company",
    topK: 5
)

for (cls, score) in similarClasses {
    print("Class: \(cls.name), Similarity: \(score)")
    print("Description: \(cls.description ?? "N/A")")
}
```

---

## 3. Integration with fdb-knowledge-layer

The knowledge layer provides a unified API that combines structured queries (triples), semantic constraints (ontology), and semantic search (embeddings).

### 3.1 Unified API for Semantic Operations

```swift
import FoundationDB
import TripleLayer
import OntologyLayer
import EmbeddingLayer

actor UnifiedKnowledgeStore {
    private let tripleStore: TripleStore
    private let ontologyStore: OntologyStore
    private let embeddingStore: EmbeddingStore
    private let searchEngine: SearchEngine
    private let generator: EmbeddingGenerator

    init(
        database: any DatabaseProtocol,
        rootPrefix: String,
        generator: EmbeddingGenerator
    ) async throws {
        self.tripleStore = try await TripleStore(database: database, rootPrefix: rootPrefix)
        self.ontologyStore = OntologyStore(database: database, rootPrefix: rootPrefix)
        self.embeddingStore = EmbeddingStore(database: database, rootPrefix: rootPrefix)
        self.searchEngine = SearchEngine(store: embeddingStore)
        self.generator = generator
    }

    /// Insert triple with automatic embedding generation
    func insertTriple(
        _ triple: Triple,
        generateEmbedding: Bool = true
    ) async throws {
        // 1. Insert triple
        try await tripleStore.insert(triple)

        // 2. Generate embedding if requested
        if generateEmbedding {
            let text = "\(triple.subject.stringValue) \(triple.predicate.stringValue) \(triple.object.stringValue)"
            let vector = try await generator.generate(text: text)

            let record = EmbeddingRecord(
                id: "triple:\(triple.id)",
                vector: vector,
                model: generator.modelMetadata.name,
                dimension: generator.modelMetadata.dimension,
                sourceType: .triple,
                metadata: [
                    "subject": triple.subject.stringValue,
                    "predicate": triple.predicate.stringValue,
                    "object": triple.object.stringValue
                ],
                createdAt: Date()
            )

            try await embeddingStore.save(record)
        }
    }

    /// Define class with automatic embedding generation
    func defineClass(
        _ cls: OntologyClass,
        generateEmbedding: Bool = true
    ) async throws {
        // 1. Define class
        try await ontologyStore.defineClass(cls)

        // 2. Generate embedding if requested
        if generateEmbedding {
            var parts: [String] = []
            parts.append("Class: \(cls.name)")
            if let description = cls.description {
                parts.append(description)
            }
            if !cls.properties.isEmpty {
                parts.append("Properties: \(cls.properties.joined(separator: ", "))")
            }

            let text = parts.joined(separator: ". ")
            let vector = try await generator.generate(text: text)

            let record = EmbeddingRecord(
                id: "class:\(cls.name)",
                vector: vector,
                model: generator.modelMetadata.name,
                dimension: generator.modelMetadata.dimension,
                sourceType: .entity,
                metadata: [
                    "className": cls.name,
                    "description": cls.description ?? ""
                ],
                createdAt: Date()
            )

            try await embeddingStore.save(record)
        }
    }
}
```

---

### 3.2 Hybrid Queries (Structural + Semantic)

Combine structured triple queries with semantic similarity.

```swift
actor HybridQueryEngine {
    private let tripleStore: TripleStore
    private let embeddingStore: EmbeddingStore
    private let searchEngine: SearchEngine
    private let generator: EmbeddingGenerator

    /// Hybrid query: structural constraints + semantic ranking
    func query(
        subjectType: String? = nil,
        predicateType: String? = nil,
        semanticQuery: String? = nil,
        topK: Int = 10
    ) async throws -> [(Triple, Float)] {
        var candidates: [Triple] = []

        // 1. Structural filtering
        if let subjectType = subjectType {
            // Filter by subject type (requires type information in metadata)
            candidates = try await tripleStore.query(
                subject: nil,
                predicate: nil,
                object: nil
            ).filter { triple in
                // Check if subject matches type
                // This assumes type information is encoded in URI
                triple.subject.stringValue.contains("/\(subjectType.lowercased())/")
            }
        } else {
            // Get all triples
            candidates = try await tripleStore.query(
                subject: nil,
                predicate: nil,
                object: nil
            )
        }

        // 2. Semantic ranking
        if let semanticQuery = semanticQuery {
            // Generate query embedding
            let queryVector = try await generator.generate(text: semanticQuery)

            // Get embeddings for candidate triples
            var rankedTriples: [(Triple, Float)] = []

            for triple in candidates {
                if let embedding = try? await embeddingStore.get(
                    id: "triple:\(triple.id)",
                    model: generator.modelMetadata.name
                ) {
                    // Compute similarity
                    let similarity = computeCosineSimilarity(queryVector, embedding.vector)
                    rankedTriples.append((triple, similarity))
                }
            }

            // Sort by similarity
            rankedTriples.sort { $0.1 > $1.1 }

            return Array(rankedTriples.prefix(topK))
        } else {
            // No semantic ranking, return structural results
            return candidates.prefix(topK).map { ($0, 1.0) }
        }
    }

    private func computeCosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        assert(a.count == b.count)

        let dotProduct = zip(a, b).map(*).reduce(0, +)
        let normA = sqrt(a.map { $0 * $0 }.reduce(0, +))
        let normB = sqrt(b.map { $0 * $0 }.reduce(0, +))

        return dotProduct / (normA * normB)
    }
}
```

---

### 3.3 Knowledge Graph Embeddings

Generate embeddings for entire subgraphs.

```swift
actor KnowledgeGraphEmbedder {
    private let tripleStore: TripleStore
    private let embeddingStore: EmbeddingStore
    private let generator: EmbeddingGenerator

    /// Generate embedding for entity's subgraph
    func generateSubgraphEmbedding(
        entityURI: String,
        depth: Int = 2
    ) async throws -> EmbeddingRecord {
        // 1. Fetch subgraph (BFS up to depth)
        let subgraph = try await fetchSubgraph(entityURI: entityURI, depth: depth)

        // 2. Convert subgraph to text
        let text = subgraph.map { triple in
            "\(triple.subject.stringValue) \(triple.predicate.stringValue) \(triple.object.stringValue)"
        }.joined(separator: ". ")

        // 3. Generate embedding
        let vector = try await generator.generate(text: text)

        // 4. Create record
        let record = EmbeddingRecord(
            id: "subgraph:\(entityURI.hashValue)",
            vector: vector,
            model: generator.modelMetadata.name,
            dimension: generator.modelMetadata.dimension,
            sourceType: .entity,
            metadata: [
                "entityURI": entityURI,
                "depth": String(depth),
                "tripleCount": String(subgraph.count)
            ],
            createdAt: Date()
        )

        return record
    }

    private func fetchSubgraph(entityURI: String, depth: Int) async throws -> [Triple] {
        var visited = Set<String>()
        var subgraph: [Triple] = []
        var frontier = [entityURI]

        for _ in 0..<depth {
            var nextFrontier: [String] = []

            for uri in frontier where !visited.contains(uri) {
                visited.insert(uri)

                let triples = try await tripleStore.query(
                    subject: .uri(uri),
                    predicate: nil,
                    object: nil
                )

                subgraph.append(contentsOf: triples)

                for triple in triples {
                    if case .uri(let objectURI) = triple.object {
                        nextFrontier.append(objectURI)
                    }
                }
            }

            frontier = nextFrontier
        }

        return subgraph
    }
}
```

---

### 3.4 Multi-Hop Reasoning with Embeddings

Use embeddings to guide multi-hop reasoning.

```swift
actor SemanticReasoningEngine {
    private let tripleStore: TripleStore
    private let embeddingStore: EmbeddingStore
    private let searchEngine: SearchEngine
    private let generator: EmbeddingGenerator

    /// Multi-hop reasoning with semantic guidance
    func reason(
        from startEntity: String,
        to targetConcept: String,
        maxHops: Int = 3
    ) async throws -> [[Triple]] {
        // 1. Generate target concept embedding
        let targetVector = try await generator.generate(text: targetConcept)

        // 2. BFS with semantic guidance
        var paths: [[Triple]] = []
        var queue: [(entity: String, path: [Triple])] = [(startEntity, [])]
        var visited = Set<String>()

        while !queue.isEmpty && paths.count < 10 {
            let (currentEntity, currentPath) = queue.removeFirst()

            if currentPath.count >= maxHops {
                continue
            }

            visited.insert(currentEntity)

            // Get outgoing triples
            let outgoing = try await tripleStore.query(
                subject: .uri(currentEntity),
                predicate: nil,
                object: nil
            )

            // Rank triples by semantic similarity to target
            var rankedTriples: [(Triple, Float)] = []

            for triple in outgoing {
                if let embedding = try? await embeddingStore.get(
                    id: "triple:\(triple.id)",
                    model: generator.modelMetadata.name
                ) {
                    let similarity = computeCosineSimilarity(targetVector, embedding.vector)
                    rankedTriples.append((triple, similarity))
                }
            }

            rankedTriples.sort { $0.1 > $1.1 }

            // Explore top-K most similar triples
            for (triple, similarity) in rankedTriples.prefix(5) {
                guard case .uri(let nextEntity) = triple.object else {
                    continue
                }

                let newPath = currentPath + [triple]

                // Check if this path is relevant to target concept
                if similarity > 0.7 {
                    paths.append(newPath)
                }

                if !visited.contains(nextEntity) {
                    queue.append((nextEntity, newPath))
                }
            }
        }

        return paths
    }

    private func computeCosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        assert(a.count == b.count)
        let dotProduct = zip(a, b).map(*).reduce(0, +)
        let normA = sqrt(a.map { $0 * $0 }.reduce(0, +))
        let normB = sqrt(b.map { $0 * $0 }.reduce(0, +))
        return dotProduct / (normA * normB)
    }
}
```

---

### 3.5 Example Workflow: Query Knowledge → Semantic Expansion → Ranked Results

```swift
import FoundationDB
import EmbeddingLayer

// Initialize
try await FDBClient.initialize()
let database = try FDBClient.openDatabase()

let model = EmbeddingModelMetadata(
    name: "mlx-embed-1024-v1",
    version: "1.0.0",
    dimension: 1024,
    provider: "Apple MLX",
    modelType: "Sentence-BERT"
)
let generator = MLXEmbeddingGenerator(model: model)

let knowledgeStore = try await UnifiedKnowledgeStore(
    database: database,
    rootPrefix: "myapp",
    generator: generator
)

// 1. Query with structural and semantic constraints
let queryEngine = HybridQueryEngine(
    tripleStore: knowledgeStore.tripleStore,
    embeddingStore: knowledgeStore.embeddingStore,
    searchEngine: knowledgeStore.searchEngine,
    generator: generator
)

let results = try await queryEngine.query(
    subjectType: "Person",
    predicateType: nil,
    semanticQuery: "software engineers working at tech companies",
    topK: 10
)

// 2. Display ranked results
for (triple, score) in results {
    print("Score: \(score)")
    print("Triple: \(triple.subject) \(triple.predicate) \(triple.object)")
}

// 3. Multi-hop reasoning
let reasoningEngine = SemanticReasoningEngine(
    tripleStore: knowledgeStore.tripleStore,
    embeddingStore: knowledgeStore.embeddingStore,
    searchEngine: knowledgeStore.searchEngine,
    generator: generator
)

let paths = try await reasoningEngine.reason(
    from: "http://example.org/person/Alice",
    to: "artificial intelligence researcher",
    maxHops: 3
)

for (idx, path) in paths.enumerated() {
    print("Path \(idx + 1):")
    for triple in path {
        print("  \(triple.predicate)")
    }
}
```

---

## 4. External Vector Database Integration

For production-scale systems (>100K vectors), integrating with specialized vector databases provides better performance.

### 4.1 Milvus Integration Pattern

[Milvus](https://milvus.io/) is an open-source vector database optimized for billion-scale similarity search.

```swift
import Milvus

actor MilvusEmbeddingStore {
    private let fdbDatabase: any DatabaseProtocol
    private let fdbStore: EmbeddingStore
    private let milvusClient: MilvusClient
    private let collectionName: String

    init(
        fdbDatabase: any DatabaseProtocol,
        milvusHost: String,
        milvusPort: Int,
        collectionName: String,
        rootPrefix: String
    ) async throws {
        self.fdbDatabase = fdbDatabase
        self.fdbStore = EmbeddingStore(database: fdbDatabase, rootPrefix: rootPrefix)
        self.milvusClient = try await MilvusClient(host: milvusHost, port: milvusPort)
        self.collectionName = collectionName

        // Create collection if not exists
        try await createCollectionIfNeeded()
    }

    private func createCollectionIfNeeded() async throws {
        let exists = try await milvusClient.hasCollection(name: collectionName)

        if !exists {
            // Define schema
            let schema = CollectionSchema(
                name: collectionName,
                fields: [
                    FieldSchema(name: "id", dataType: .varChar, maxLength: 256, isPrimary: true),
                    FieldSchema(name: "vector", dataType: .floatVector, dimension: 1024),
                    FieldSchema(name: "model", dataType: .varChar, maxLength: 128),
                    FieldSchema(name: "sourceType", dataType: .varChar, maxLength: 64),
                    FieldSchema(name: "createdAt", dataType: .int64)
                ]
            )

            try await milvusClient.createCollection(schema: schema)

            // Create index for vector field
            let indexParams = IndexParams(
                indexType: .hnsw,
                metricType: .cosine,
                params: ["M": 16, "efConstruction": 200]
            )

            try await milvusClient.createIndex(
                collectionName: collectionName,
                fieldName: "vector",
                indexParams: indexParams
            )
        }
    }

    /// Save embedding to both FDB (metadata) and Milvus (vector)
    func save(_ record: EmbeddingRecord) async throws {
        // 1. Save metadata to FDB
        let metadata = EmbeddingMetadata(
            id: record.id,
            model: record.model,
            dimension: record.dimension,
            sourceType: record.sourceType,
            createdAt: record.createdAt
        )
        try await fdbStore.saveMetadata(metadata)

        // 2. Save vector to Milvus
        let entity = MilvusEntity(
            id: record.id,
            vector: record.vector,
            model: record.model,
            sourceType: record.sourceType.rawValue,
            createdAt: Int64(record.createdAt.timeIntervalSince1970)
        )

        try await milvusClient.insert(
            collectionName: collectionName,
            entities: [entity]
        )
    }

    /// Search using Milvus (fast ANN search)
    func search(
        queryVector: [Float],
        topK: Int,
        model: String,
        similarityMetric: SimilarityMetric = .cosine
    ) async throws -> [SearchResult] {
        // 1. Search in Milvus
        let searchParams = SearchParams(
            topK: topK,
            metricType: similarityMetric == .cosine ? .cosine : .innerProduct,
            params: ["ef": 200]
        )

        let milvusResults = try await milvusClient.search(
            collectionName: collectionName,
            vectors: [queryVector],
            fieldName: "vector",
            searchParams: searchParams,
            filter: "model == '\(model)'"
        )

        // 2. Enrich with FDB metadata
        var results: [SearchResult] = []

        for hit in milvusResults[0] {
            let metadata = try? await fdbStore.getMetadata(id: hit.id)

            results.append(SearchResult(
                id: hit.id,
                score: hit.score,
                vector: nil,
                metadata: metadata?.toDict(),
                distance: hit.distance,
                sourceType: metadata?.sourceType,
                model: model
            ))
        }

        return results
    }

    /// Delete from both systems
    func delete(id: String, model: String) async throws {
        try await fdbStore.delete(id: id, model: model)
        try await milvusClient.delete(
            collectionName: collectionName,
            expr: "id == '\(id)'"
        )
    }
}

struct EmbeddingMetadata {
    let id: String
    let model: String
    let dimension: Int
    let sourceType: SourceType
    let createdAt: Date

    func toDict() -> [String: String] {
        return [
            "id": id,
            "model": model,
            "dimension": String(dimension),
            "sourceType": sourceType.rawValue
        ]
    }
}
```

**Usage**:

```swift
let milvusStore = try await MilvusEmbeddingStore(
    fdbDatabase: database,
    milvusHost: "localhost",
    milvusPort: 19530,
    collectionName: "embeddings",
    rootPrefix: "myapp"
)

// Save embedding
let record = EmbeddingRecord(
    id: "triple:12345",
    vector: vector,
    model: "mlx-embed-1024-v1",
    dimension: 1024,
    sourceType: .triple,
    createdAt: Date()
)
try await milvusStore.save(record)

// Search
let results = try await milvusStore.search(
    queryVector: queryVector,
    topK: 10,
    model: "mlx-embed-1024-v1"
)
```

---

### 4.2 Weaviate Integration Pattern

[Weaviate](https://weaviate.io/) is a vector database with built-in ML model integration.

```swift
import Weaviate

actor WeaviateEmbeddingStore {
    private let fdbStore: EmbeddingStore
    private let weaviateClient: WeaviateClient
    private let className: String

    init(
        fdbDatabase: any DatabaseProtocol,
        weaviateURL: String,
        className: String,
        rootPrefix: String
    ) async throws {
        self.fdbStore = EmbeddingStore(database: fdbDatabase, rootPrefix: rootPrefix)
        self.weaviateClient = try WeaviateClient(url: weaviateURL)
        self.className = className

        try await createClassIfNeeded()
    }

    private func createClassIfNeeded() async throws {
        let exists = try await weaviateClient.schema.hasClass(name: className)

        if !exists {
            let schema = WeaviateClass(
                name: className,
                vectorizer: "none",  // We provide vectors manually
                properties: [
                    WeaviateProperty(name: "id", dataType: ["string"]),
                    WeaviateProperty(name: "model", dataType: ["string"]),
                    WeaviateProperty(name: "sourceType", dataType: ["string"]),
                    WeaviateProperty(name: "createdAt", dataType: ["date"])
                ]
            )

            try await weaviateClient.schema.createClass(schema: schema)
        }
    }

    /// Save to both FDB and Weaviate
    func save(_ record: EmbeddingRecord) async throws {
        // 1. Save metadata to FDB
        try await fdbStore.saveMetadata(EmbeddingMetadata(
            id: record.id,
            model: record.model,
            dimension: record.dimension,
            sourceType: record.sourceType,
            createdAt: record.createdAt
        ))

        // 2. Save to Weaviate
        let object = WeaviateObject(
            class: className,
            properties: [
                "id": record.id,
                "model": record.model,
                "sourceType": record.sourceType.rawValue,
                "createdAt": ISO8601DateFormatter().string(from: record.createdAt)
            ],
            vector: record.vector
        )

        try await weaviateClient.data.create(object: object)
    }

    /// Search using Weaviate
    func search(
        queryVector: [Float],
        topK: Int,
        model: String
    ) async throws -> [SearchResult] {
        // Build GraphQL query
        let query = """
        {
          Get {
            \(className)(
              nearVector: {
                vector: \(queryVector)
                certainty: 0.7
              }
              where: {
                path: ["model"]
                operator: Equal
                valueString: "\(model)"
              }
              limit: \(topK)
            ) {
              id
              model
              sourceType
              _additional {
                certainty
                distance
              }
            }
          }
        }
        """

        let response = try await weaviateClient.graphql.query(query: query)

        // Parse results
        var results: [SearchResult] = []

        for item in response.data.get[className] {
            let metadata = try? await fdbStore.getMetadata(id: item.id)

            results.append(SearchResult(
                id: item.id,
                score: item._additional.certainty,
                vector: nil,
                metadata: metadata?.toDict(),
                distance: item._additional.distance,
                sourceType: metadata?.sourceType,
                model: model
            ))
        }

        return results
    }
}
```

---

### 4.3 Qdrant Integration Pattern

[Qdrant](https://qdrant.tech/) is a high-performance vector database with filtering support.

```swift
import Qdrant

actor QdrantEmbeddingStore {
    private let fdbStore: EmbeddingStore
    private let qdrantClient: QdrantClient
    private let collectionName: String

    init(
        fdbDatabase: any DatabaseProtocol,
        qdrantHost: String,
        qdrantPort: Int,
        collectionName: String,
        rootPrefix: String
    ) async throws {
        self.fdbStore = EmbeddingStore(database: fdbDatabase, rootPrefix: rootPrefix)
        self.qdrantClient = try QdrantClient(host: qdrantHost, port: qdrantPort)
        self.collectionName = collectionName

        try await createCollectionIfNeeded()
    }

    private func createCollectionIfNeeded() async throws {
        let exists = try await qdrantClient.hasCollection(name: collectionName)

        if !exists {
            let config = CollectionConfig(
                vectorsConfig: VectorParams(
                    size: 1024,
                    distance: .cosine
                )
            )

            try await qdrantClient.createCollection(
                name: collectionName,
                config: config
            )

            // Create payload index for filtering
            try await qdrantClient.createPayloadIndex(
                collectionName: collectionName,
                fieldName: "model",
                fieldType: .keyword
            )
        }
    }

    /// Save to both FDB and Qdrant
    func save(_ record: EmbeddingRecord) async throws {
        // 1. Save metadata to FDB
        try await fdbStore.saveMetadata(EmbeddingMetadata(
            id: record.id,
            model: record.model,
            dimension: record.dimension,
            sourceType: record.sourceType,
            createdAt: record.createdAt
        ))

        // 2. Save to Qdrant
        let point = QdrantPoint(
            id: record.id.hashValue,
            vector: record.vector,
            payload: [
                "id": .string(record.id),
                "model": .string(record.model),
                "sourceType": .string(record.sourceType.rawValue),
                "createdAt": .integer(Int64(record.createdAt.timeIntervalSince1970))
            ]
        )

        try await qdrantClient.upsert(
            collectionName: collectionName,
            points: [point]
        )
    }

    /// Search with filtering
    func search(
        queryVector: [Float],
        topK: Int,
        model: String,
        filter: SearchFilter? = nil
    ) async throws -> [SearchResult] {
        // Build filter
        var qdrantFilter: QdrantFilter? = nil

        if let filter = filter {
            var conditions: [QdrantCondition] = []

            // Model filter
            conditions.append(.match(key: "model", value: .string(model)))

            // Source type filter
            if let sourceTypes = filter.sourceTypes {
                let sourceTypeValues = sourceTypes.map { QdrantValue.string($0.rawValue) }
                conditions.append(.anyOf(key: "sourceType", values: sourceTypeValues))
            }

            qdrantFilter = QdrantFilter(must: conditions)
        }

        // Search
        let searchParams = SearchParams(
            vector: queryVector,
            limit: topK,
            filter: qdrantFilter
        )

        let qdrantResults = try await qdrantClient.search(
            collectionName: collectionName,
            params: searchParams
        )

        // Enrich with FDB metadata
        var results: [SearchResult] = []

        for hit in qdrantResults {
            guard let idStr = hit.payload["id"]?.stringValue else {
                continue
            }

            let metadata = try? await fdbStore.getMetadata(id: idStr)

            results.append(SearchResult(
                id: idStr,
                score: hit.score,
                vector: nil,
                metadata: metadata?.toDict(),
                distance: 1.0 - hit.score,
                sourceType: metadata?.sourceType,
                model: model
            ))
        }

        return results
    }
}
```

---

### 4.4 Hybrid Storage Strategy

**Design Pattern**: Use FDB for metadata and consistency, external vector DB for fast similarity search.

```
┌──────────────────────────────────────────────────┐
│              Application Layer                   │
└──────────────────┬───────────────────────────────┘
                   │
       ┌───────────┼────────────┐
       │                        │
┌──────▼──────┐         ┌───────▼────────┐
│     FDB     │         │   Vector DB    │
│             │         │                │
│ - Metadata  │         │ - Vectors      │
│ - IDs       │         │ - ANN Index    │
│ - Timestamps│         │ - Fast Search  │
│ - Mapping   │         │                │
└─────────────┘         └────────────────┘
```

**Consistency Guarantees**:

1. **Atomic Writes**: Use FDB transaction for metadata, then write to vector DB
2. **Read-Your-Writes**: Always read from FDB first to get latest metadata
3. **Eventual Consistency**: Vector DB may lag slightly, but FDB is authoritative
4. **Reconciliation**: Periodic sync job to ensure consistency

```swift
actor ConsistentHybridStore {
    private let fdbStore: EmbeddingStore
    private let vectorDB: VectorDBClient  // Milvus, Weaviate, or Qdrant

    /// Save with transaction semantics
    func save(_ record: EmbeddingRecord) async throws {
        // 1. Save to FDB (authoritative)
        try await fdbStore.save(record)

        // 2. Save to vector DB (best-effort)
        do {
            try await vectorDB.insert(
                id: record.id,
                vector: record.vector,
                metadata: record.metadata ?? [:]
            )
        } catch {
            // Log error but don't fail
            // Reconciliation job will fix inconsistency
            print("Warning: Failed to save to vector DB: \(error)")
        }
    }

    /// Search with fallback
    func search(
        queryVector: [Float],
        topK: Int,
        model: String
    ) async throws -> [SearchResult] {
        do {
            // Try vector DB first (fast)
            return try await vectorDB.search(
                vector: queryVector,
                limit: topK,
                filter: ["model": model]
            )
        } catch {
            // Fall back to FDB brute-force search
            print("Vector DB unavailable, falling back to FDB")
            return try await fdbStore.searchBruteForce(
                queryVector: queryVector,
                topK: topK,
                model: model
            )
        }
    }

    /// Reconciliation job (run periodically)
    func reconcile() async throws {
        // 1. Get all metadata from FDB
        let allRecords = try await fdbStore.listAll()

        // 2. Check which vectors are missing in vector DB
        var missingCount = 0

        for record in allRecords {
            let exists = try await vectorDB.exists(id: record.id)

            if !exists {
                // Re-insert missing vector
                try await vectorDB.insert(
                    id: record.id,
                    vector: record.vector,
                    metadata: record.metadata ?? [:]
                )
                missingCount += 1
            }
        }

        print("Reconciliation complete: fixed \(missingCount) inconsistencies")
    }
}
```

---

## 5. Embedding Model Integration

The embedding layer supports multiple embedding models through a common `EmbeddingGenerator` protocol.

### 5.1 MLX Embed (Local Apple Silicon)

[MLX](https://github.com/ml-explore/mlx) is Apple's ML framework optimized for Apple Silicon.

```swift
import MLX

public actor MLXEmbeddingGenerator: EmbeddingGenerator {
    private let model: MLXModel
    private let tokenizer: MLXTokenizer
    public let modelMetadata: EmbeddingModelMetadata

    public init(modelPath: String) async throws {
        // Load model
        self.model = try await MLXModel.load(path: modelPath)
        self.tokenizer = try MLXTokenizer(modelPath: modelPath)

        // Set metadata
        self.modelMetadata = EmbeddingModelMetadata(
            name: "mlx-embed-1024-v1",
            version: "1.0.0",
            dimension: 1024,
            provider: "Apple MLX",
            modelType: "Sentence-BERT",
            normalized: true,
            maxInputLength: 512,
            supportedLanguages: ["en", "ja"]
        )
    }

    public func generate(text: String) async throws -> [Float] {
        // 1. Tokenize
        let tokens = tokenizer.encode(text: text, maxLength: modelMetadata.maxInputLength)

        // 2. Forward pass
        let output = try await model.forward(tokens: tokens)

        // 3. Pool (mean pooling)
        let pooled = meanPool(output)

        // 4. Normalize
        return normalize(pooled)
    }

    public func generateBatch(texts: [String]) async throws -> [[Float]] {
        // Batch tokenization
        let tokenBatches = texts.map { text in
            tokenizer.encode(text: text, maxLength: modelMetadata.maxInputLength)
        }

        // Batch forward pass (more efficient)
        let outputs = try await model.forwardBatch(tokens: tokenBatches)

        // Pool and normalize each
        return outputs.map { output in
            normalize(meanPool(output))
        }
    }

    private func meanPool(_ embeddings: [Float]) -> [Float] {
        // Mean pooling implementation
        let dim = modelMetadata.dimension
        var pooled = [Float](repeating: 0, count: dim)

        for i in 0..<dim {
            pooled[i] = embeddings[i]
        }

        return pooled
    }

    private func normalize(_ vector: [Float]) -> [Float] {
        let norm = sqrt(vector.map { $0 * $0 }.reduce(0, +))
        return vector.map { $0 / norm }
    }
}
```

**Usage**:

```swift
let generator = try await MLXEmbeddingGenerator(
    modelPath: "/path/to/mlx/model"
)

let embedding = try await generator.generate(text: "Hello, world!")
```

---

### 5.2 OpenAI Embeddings

[OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings) provides text-embedding-3-small and text-embedding-3-large models.

```swift
import OpenAI

public actor OpenAIEmbeddingGenerator: EmbeddingGenerator {
    private let apiKey: String
    private let client: OpenAIClient
    private let modelName: String
    public let modelMetadata: EmbeddingModelMetadata

    public init(apiKey: String, modelName: String = "text-embedding-3-large") {
        self.apiKey = apiKey
        self.client = OpenAIClient(apiKey: apiKey)
        self.modelName = modelName

        // Set metadata based on model
        let dimension: Int
        switch modelName {
        case "text-embedding-3-small":
            dimension = 1536
        case "text-embedding-3-large":
            dimension = 3072
        default:
            dimension = 1536
        }

        self.modelMetadata = EmbeddingModelMetadata(
            name: modelName,
            version: "3.0.0",
            dimension: dimension,
            provider: "OpenAI",
            modelType: "GPT-based",
            normalized: true,
            maxInputLength: 8191
        )
    }

    public func generate(text: String) async throws -> [Float] {
        let response = try await client.createEmbedding(
            model: modelName,
            input: text
        )

        return response.data[0].embedding
    }

    public func generateBatch(texts: [String]) async throws -> [[Float]] {
        let response = try await client.createEmbedding(
            model: modelName,
            input: texts
        )

        return response.data.map { $0.embedding }
    }
}
```

**Usage**:

```swift
let generator = OpenAIEmbeddingGenerator(
    apiKey: "sk-...",
    modelName: "text-embedding-3-large"
)

let embedding = try await generator.generate(text: "Hello, world!")
```

---

### 5.3 Cohere Embed

[Cohere Embed](https://docs.cohere.com/reference/embed) provides multilingual embedding models.

```swift
import Cohere

public actor CohereEmbeddingGenerator: EmbeddingGenerator {
    private let apiKey: String
    private let client: CohereClient
    private let modelName: String
    public let modelMetadata: EmbeddingModelMetadata

    public init(apiKey: String, modelName: String = "embed-multilingual-v3.0") {
        self.apiKey = apiKey
        self.client = CohereClient(apiKey: apiKey)
        self.modelName = modelName

        self.modelMetadata = EmbeddingModelMetadata(
            name: modelName,
            version: "3.0.0",
            dimension: 1024,
            provider: "Cohere",
            modelType: "Transformer",
            normalized: true,
            maxInputLength: 512,
            supportedLanguages: ["en", "es", "fr", "de", "zh", "ja", "ko", "ar"]
        )
    }

    public func generate(text: String) async throws -> [Float] {
        let response = try await client.embed(
            model: modelName,
            texts: [text],
            inputType: .searchDocument
        )

        return response.embeddings[0]
    }

    public func generateBatch(texts: [String]) async throws -> [[Float]] {
        let response = try await client.embed(
            model: modelName,
            texts: texts,
            inputType: .searchDocument
        )

        return response.embeddings
    }
}
```

---

### 5.4 Sentence Transformers

[Sentence Transformers](https://www.sbert.net/) provides pre-trained models for sentence embeddings.

```swift
import PythonKit

public actor SentenceTransformerGenerator: EmbeddingGenerator {
    private let model: PythonObject
    private let modelName: String
    public let modelMetadata: EmbeddingModelMetadata

    public init(modelName: String = "all-MiniLM-L6-v2") throws {
        let sentenceTransformers = Python.import("sentence_transformers")
        self.model = sentenceTransformers.SentenceTransformer(modelName)
        self.modelName = modelName

        // Get dimension from model
        let dimension = Int(model.get_sentence_embedding_dimension())!

        self.modelMetadata = EmbeddingModelMetadata(
            name: modelName,
            version: "1.0.0",
            dimension: dimension,
            provider: "Sentence Transformers",
            modelType: "BERT",
            normalized: true,
            maxInputLength: 256
        )
    }

    public func generate(text: String) async throws -> [Float] {
        let embedding = model.encode([text])
        let array = [Float](numpy: embedding[0])!
        return array
    }

    public func generateBatch(texts: [String]) async throws -> [[Float]] {
        let embeddings = model.encode(texts)

        var result: [[Float]] = []
        for i in 0..<texts.count {
            let array = [Float](numpy: embeddings[i])!
            result.append(array)
        }

        return result
    }
}
```

---

### 5.5 Custom Model Integration

Implement `EmbeddingGenerator` protocol for custom models.

```swift
public protocol EmbeddingGenerator: Sendable {
    /// Generate embedding for single text
    func generate(text: String) async throws -> [Float]

    /// Generate embeddings for batch of texts
    func generateBatch(texts: [String]) async throws -> [[Float]]

    /// Model metadata
    var modelMetadata: EmbeddingModelMetadata { get }
}

// Example: Custom TensorFlow model
public actor CustomTensorFlowGenerator: EmbeddingGenerator {
    private let session: TFSession
    private let inputTensor: String
    private let outputTensor: String
    public let modelMetadata: EmbeddingModelMetadata

    public init(modelPath: String) throws {
        // Load TensorFlow model
        self.session = try TFSession(modelPath: modelPath)
        self.inputTensor = "input_ids:0"
        self.outputTensor = "embeddings:0"

        self.modelMetadata = EmbeddingModelMetadata(
            name: "custom-tf-model",
            version: "1.0.0",
            dimension: 768,
            provider: "Custom",
            modelType: "TensorFlow",
            normalized: false
        )
    }

    public func generate(text: String) async throws -> [Float] {
        // Tokenize and run inference
        let tokens = tokenize(text)
        let output = try session.run(
            feed: [inputTensor: tokens],
            fetch: [outputTensor]
        )

        return output[outputTensor] as! [Float]
    }

    public func generateBatch(texts: [String]) async throws -> [[Float]] {
        // Batch inference
        var results: [[Float]] = []
        for text in texts {
            let embedding = try await generate(text: text)
            results.append(embedding)
        }
        return results
    }

    private func tokenize(_ text: String) -> [Int] {
        // Custom tokenization logic
        return []
    }
}
```

---

## 6. Real-World Use Cases

### 6.1 Semantic Search Over Knowledge Base

**Scenario**: Search for information using natural language queries.

```swift
actor KnowledgeBaseSearch {
    private let tripleStore: TripleStore
    private let embeddingStore: EmbeddingStore
    private let searchEngine: SearchEngine
    private let generator: EmbeddingGenerator

    func search(query: String, topK: Int = 10) async throws -> [SearchResult] {
        // Generate query embedding
        let queryVector = try await generator.generate(text: query)

        // Search for similar embeddings
        let results = try await searchEngine.search(
            queryVector: queryVector,
            topK: topK,
            model: generator.modelMetadata.name,
            similarityMetric: .cosine
        )

        return results
    }

    func searchWithContext(query: String, topK: Int = 10) async throws -> [(Triple, Float, String)] {
        let results = try await search(query: query, topK: topK)

        var contextualResults: [(Triple, Float, String)] = []

        for result in results {
            // Extract triple ID
            guard let tripleIdStr = result.id.split(separator: ":").last,
                  let tripleId = Int64(tripleIdStr) else {
                continue
            }

            // Fetch triple
            if let triple = try await tripleStore.getTriple(id: tripleId) {
                // Generate explanation
                let explanation = "Matched: \(triple.predicate.stringValue)"
                contextualResults.append((triple, result.score, explanation))
            }
        }

        return contextualResults
    }
}
```

**Usage**:

```swift
let searcher = KnowledgeBaseSearch(
    tripleStore: tripleStore,
    embeddingStore: embeddingStore,
    searchEngine: searchEngine,
    generator: generator
)

let results = try await searcher.searchWithContext(
    query: "software engineers working on artificial intelligence",
    topK: 10
)

for (triple, score, explanation) in results {
    print("Score: \(score)")
    print("Triple: \(triple.subject) \(triple.predicate) \(triple.object)")
    print("Explanation: \(explanation)")
}
```

---

### 6.2 Entity Linking and Deduplication

**Scenario**: Identify duplicate entities using semantic similarity.

```swift
actor EntityDeduplicator {
    private let embeddingStore: EmbeddingStore
    private let searchEngine: SearchEngine
    private let threshold: Float

    init(
        embeddingStore: EmbeddingStore,
        searchEngine: SearchEngine,
        threshold: Float = 0.95
    ) {
        self.embeddingStore = embeddingStore
        self.searchEngine = searchEngine
        self.threshold = threshold
    }

    /// Find potential duplicates for an entity
    func findDuplicates(
        entityEmbedding: EmbeddingRecord
    ) async throws -> [String] {
        // Search for similar entities
        let results = try await searchEngine.search(
            queryVector: entityEmbedding.vector,
            topK: 20,
            model: entityEmbedding.model,
            similarityMetric: .cosine
        )

        // Filter by threshold (exclude self)
        let duplicates = results
            .filter { $0.id != entityEmbedding.id && $0.score >= threshold }
            .map { $0.id }

        return duplicates
    }

    /// Deduplicate all entities
    func deduplicateAll(model: String) async throws -> [(String, [String])] {
        let allEmbeddings = try await embeddingStore.listBySourceType(
            .entity,
            model: model
        )

        var duplicateGroups: [(String, [String])] = []
        var processed = Set<String>()

        for embedding in allEmbeddings where !processed.contains(embedding.id) {
            let duplicates = try await findDuplicates(entityEmbedding: embedding)

            if !duplicates.isEmpty {
                duplicateGroups.append((embedding.id, duplicates))
                processed.insert(embedding.id)
                processed.formUnion(duplicates)
            }
        }

        return duplicateGroups
    }
}
```

---

### 6.3 Knowledge Graph Completion

**Scenario**: Predict missing triples using embedding similarity.

```swift
actor KnowledgeGraphCompletion {
    private let tripleStore: TripleStore
    private let embeddingStore: EmbeddingStore
    private let searchEngine: SearchEngine

    /// Predict missing object for (subject, predicate, ?)
    func predictObject(
        subject: Value,
        predicate: Value,
        topK: Int = 5
    ) async throws -> [(Value, Float)] {
        // 1. Find similar triples with same predicate
        let similarTriples = try await tripleStore.query(
            subject: nil,
            predicate: predicate,
            object: nil
        )

        // 2. Create candidate embedding for incomplete triple
        let candidateText = "\(subject.stringValue) \(predicate.stringValue)"
        let candidateVector = try await generator.generate(text: candidateText)

        // 3. Find most similar complete triples
        var candidates: [(Value, Float)] = []

        for triple in similarTriples {
            if let embedding = try? await embeddingStore.get(
                id: "triple:\(triple.id)",
                model: generator.modelMetadata.name
            ) {
                let similarity = computeCosineSimilarity(
                    candidateVector,
                    embedding.vector
                )
                candidates.append((triple.object, similarity))
            }
        }

        // 4. Sort and return top-K
        candidates.sort { $0.1 > $1.1 }
        return Array(candidates.prefix(topK))
    }
}
```

---

### 6.4 Question Answering

**Scenario**: Answer natural language questions using knowledge graph.

```swift
actor QuestionAnsweringSystem {
    private let tripleStore: TripleStore
    private let embeddingStore: EmbeddingStore
    private let searchEngine: SearchEngine
    private let generator: EmbeddingGenerator

    func answer(question: String) async throws -> String {
        // 1. Generate question embedding
        let questionVector = try await generator.generate(text: question)

        // 2. Search for relevant triples
        let results = try await searchEngine.search(
            queryVector: questionVector,
            topK: 5,
            model: generator.modelMetadata.name,
            similarityMetric: .cosine
        )

        // 3. Fetch triples and construct answer
        var answerParts: [String] = []

        for result in results {
            guard let tripleIdStr = result.id.split(separator: ":").last,
                  let tripleId = Int64(tripleIdStr),
                  let triple = try await tripleStore.getTriple(id: tripleId) else {
                continue
            }

            // Extract answer from triple
            let subject = triple.subject.stringValue
            let predicate = triple.predicate.stringValue
            let object = triple.object.stringValue

            answerParts.append("\(subject) \(predicate) \(object) (confidence: \(result.score))")
        }

        return answerParts.joined(separator: "\n")
    }
}
```

---

### 6.5 Recommendation Systems

**Scenario**: Recommend related entities based on user interests.

```swift
actor RecommendationEngine {
    private let embeddingStore: EmbeddingStore
    private let searchEngine: SearchEngine

    /// Recommend entities based on user profile
    func recommend(
        userProfile: [String: Float],  // Entity ID → Interest score
        topK: Int = 10
    ) async throws -> [(String, Float)] {
        // 1. Compute user embedding as weighted average
        var userVector = [Float](repeating: 0, count: 1024)
        var totalWeight: Float = 0

        for (entityId, weight) in userProfile {
            if let embedding = try? await embeddingStore.get(
                id: entityId,
                model: "mlx-embed-1024-v1"
            ) {
                for i in 0..<userVector.count {
                    userVector[i] += embedding.vector[i] * weight
                }
                totalWeight += weight
            }
        }

        // Normalize
        userVector = userVector.map { $0 / totalWeight }

        // 2. Search for similar entities
        let results = try await searchEngine.search(
            queryVector: userVector,
            topK: topK * 2,  // Get more candidates
            model: "mlx-embed-1024-v1",
            similarityMetric: .cosine
        )

        // 3. Filter out entities already in profile
        let recommendations = results
            .filter { !userProfile.keys.contains($0.id) }
            .map { ($0.id, $0.score) }

        return Array(recommendations.prefix(topK))
    }
}
```

---

## 7. Best Practices

### 7.1 When to Use Which Integration Pattern

| Pattern | Use When | Pros | Cons |
|---------|----------|------|------|
| **Simple Concatenation** | Quick prototyping, small datasets | Fast, simple | Less semantic meaning |
| **Contextual with Labels** | Production systems, human-readable | Better quality | Requires label lookup |
| **Graph Context** | Complex reasoning, research | Rich context | Slow, may dilute meaning |
| **FDB Only** | <10K vectors | Simple, no dependencies | Slow search at scale |
| **Hybrid (FDB + VectorDB)** | >100K vectors | Fast search, consistent | More complex |
| **VectorDB Only** | Large scale, search-only | Very fast | No transactional guarantees |

---

### 7.2 Performance Optimization Strategies

#### 1. Batch Processing

```swift
// ❌ Bad: Process one at a time
for triple in triples {
    let vector = try await generator.generate(text: triple.text)
    try await embeddingStore.save(EmbeddingRecord(...))
}

// ✅ Good: Process in batches
let texts = triples.map { $0.text }
let vectors = try await generator.generateBatch(texts: texts)

let records = zip(triples, vectors).map { EmbeddingRecord(...) }
try await embeddingStore.saveBatch(records)
```

#### 2. Caching

```swift
actor CachedEmbeddingStore {
    private let embeddingStore: EmbeddingStore
    private var cache: [String: EmbeddingRecord] = [:]

    func get(id: String, model: String) async throws -> EmbeddingRecord? {
        let key = "\(model):\(id)"

        if let cached = cache[key] {
            return cached
        }

        let record = try await embeddingStore.get(id: id, model: model)
        if let record = record {
            cache[key] = record
        }

        return record
    }
}
```

#### 3. Parallel Processing

```swift
// Process multiple embeddings concurrently
await withTaskGroup(of: EmbeddingRecord.self) { group in
    for triple in triples {
        group.addTask {
            let vector = try await self.generator.generate(text: triple.text)
            return EmbeddingRecord(...)
        }
    }

    var records: [EmbeddingRecord] = []
    for await record in group {
        records.append(record)
    }

    try await embeddingStore.saveBatch(records)
}
```

---

### 7.3 Error Handling Across Layers

```swift
actor RobustIntegration {
    func insertTripleWithEmbedding(_ triple: Triple) async throws {
        do {
            // 1. Insert triple (critical)
            try await tripleStore.insert(triple)

        } catch {
            // Critical error - propagate
            throw IntegrationError.tripleInsertFailed(error)
        }

        do {
            // 2. Generate embedding (non-critical)
            let vector = try await generator.generate(text: triple.text)
            let record = EmbeddingRecord(...)
            try await embeddingStore.save(record)

        } catch {
            // Non-critical error - log and continue
            print("Warning: Failed to generate embedding: \(error)")
            // Triple is still saved successfully
        }
    }
}
```

---

### 7.4 Caching Coordination

```swift
actor CoordinatedCache {
    private var tripleCache: [Int64: Triple] = [:]
    private var embeddingCache: [String: EmbeddingRecord] = [:]

    func invalidate(tripleId: Int64) {
        tripleCache.removeValue(forKey: tripleId)
        embeddingCache.removeValue(forKey: "triple:\(tripleId)")
    }

    func clearAll() {
        tripleCache.removeAll()
        embeddingCache.removeAll()
    }
}
```

---

## 8. Migration Patterns

### 8.1 Moving Between Embedding Models

```swift
actor ModelMigrationManager {
    private let embeddingStore: EmbeddingStore
    private let oldGenerator: EmbeddingGenerator
    private let newGenerator: EmbeddingGenerator

    func migrate(batchSize: Int = 100) async throws {
        // 1. Get all embeddings for old model
        let oldEmbeddings = try await embeddingStore.listByModel(
            oldGenerator.modelMetadata.name,
            limit: Int.max
        )

        print("Migrating \(oldEmbeddings.count) embeddings...")

        // 2. Process in batches
        for batch in oldEmbeddings.chunked(into: batchSize) {
            // Fetch original text from metadata
            let texts = batch.compactMap { $0.metadata?["text"] }

            // Generate new embeddings
            let newVectors = try await newGenerator.generateBatch(texts: texts)

            // Create new records
            let newRecords = zip(batch, newVectors).map { (old, newVector) in
                EmbeddingRecord(
                    id: old.id,
                    vector: newVector,
                    model: newGenerator.modelMetadata.name,
                    dimension: newGenerator.modelMetadata.dimension,
                    sourceType: old.sourceType,
                    metadata: old.metadata,
                    createdAt: Date()
                )
            }

            // Save new embeddings
            try await embeddingStore.saveBatch(newRecords)

            print("Processed \(newRecords.count) embeddings")
        }

        print("Migration complete!")
    }
}
```

---

### 8.2 Moving Between Vector Databases

```swift
actor VectorDBMigrationManager {
    private let fdbStore: EmbeddingStore
    private let oldVectorDB: VectorDBClient
    private let newVectorDB: VectorDBClient

    func migrate(batchSize: Int = 1000) async throws {
        // 1. Get all embeddings from FDB (source of truth)
        let allEmbeddings = try await fdbStore.listAll()

        print("Migrating \(allEmbeddings.count) vectors to new vector DB...")

        // 2. Batch insert to new vector DB
        for batch in allEmbeddings.chunked(into: batchSize) {
            for record in batch {
                try await newVectorDB.insert(
                    id: record.id,
                    vector: record.vector,
                    metadata: record.metadata ?? [:]
                )
            }

            print("Migrated \(batch.count) vectors")
        }

        // 3. Verify migration
        for record in allEmbeddings.prefix(100) {
            let exists = try await newVectorDB.exists(id: record.id)
            if !exists {
                throw MigrationError.verificationFailed(record.id)
            }
        }

        print("Migration complete and verified!")
    }
}
```

---

## Summary

This integration guide covered:

1. **Triple Layer Integration**: Generate embeddings for triples, enable semantic search over knowledge graphs
2. **Ontology Layer Integration**: Create class embeddings, find similar concepts, hierarchical representations
3. **Knowledge Layer Integration**: Unified API, hybrid queries, multi-hop reasoning
4. **External Vector DB Integration**: Milvus, Weaviate, Qdrant patterns, hybrid storage strategies
5. **Embedding Models**: MLX, OpenAI, Cohere, Sentence Transformers, custom models
6. **Real-World Use Cases**: Semantic search, entity linking, knowledge graph completion, QA, recommendations
7. **Best Practices**: When to use each pattern, performance optimization, error handling, caching
8. **Migration Patterns**: Model migration, vector DB migration

The embedding layer serves as the semantic bridge between structured knowledge (triples, ontology) and intelligent applications (search, reasoning, recommendations).

---

**Document Version**: 1.0
**Last Updated**: 2025-10-30
**Status**: Complete
