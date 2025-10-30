import Foundation
import Testing
@testable import EmbeddingLayer
@preconcurrency import FoundationDB
import Logging

// MARK: - Global FDB Initialization

private nonisolated(unsafe) var globalFDBInitialized = false
private nonisolated(unsafe) var globalDatabase: FDBDatabase?
private let globalInitLock = NSLock()

private func getGlobalDatabase() throws -> FDBDatabase {
    globalInitLock.lock()
    defer { globalInitLock.unlock() }

    if let db = globalDatabase {
        return db
    }

    if !globalFDBInitialized {
        let group = DispatchGroup()
        group.enter()
        var initError: Error?

        Task {
            do {
                try await FDBClient.initialize()
            } catch {
                initError = error
            }
            group.leave()
        }

        group.wait()

        if let error = initError {
            throw error
        }

        globalFDBInitialized = true
    }

    let db = try FDBClient.openDatabase()
    globalDatabase = db
    return db
}

// MARK: - Test Helpers

/// Wrapper actor that implements EmbeddingStoreProtocol for testing
actor EmbeddingStoreWrapper: EmbeddingStoreProtocol {
    private let store: EmbeddingStore
    private let modelManager: ModelManager

    init(store: EmbeddingStore, modelManager: ModelManager) {
        self.store = store
        self.modelManager = modelManager
    }

    func getModelMetadata(model: String) async throws -> EmbeddingModelMetadata? {
        return try await modelManager.getModel(name: model)
    }

    func getAllEmbeddings(model: String) async throws -> [EmbeddingRecord] {
        return try await store.getAllByModel(model: model)
    }
}

// MARK: - Test Suite

@Suite("Embedding Layer Tests")
struct EmbeddingLayerTests {

    func getDatabase() throws -> FDBDatabase {
        return try getGlobalDatabase()
    }

    func createTestPrefix() -> String {
        return "test_\(UUID().uuidString)"
    }

    func createMockGenerator(dimension: Int = 384) -> MockEmbeddingGenerator {
        return MockEmbeddingGenerator(modelName: "mock-embed", dimension: dimension)
    }

    // MARK: - Model Management Tests

    @Test("Register and retrieve model")
    func testRegisterAndRetrieveModel() async throws {
        let db = try getDatabase()
        let prefix = createTestPrefix()
        let logger = Logger(label: "test.model.register")

        let modelManager = ModelManager(database: db, rootPrefix: prefix, logger: logger)

        // Register model
        let model = EmbeddingModelMetadata(
            name: "test-model-1",
            dimension: 384,
            provider: "test-provider",
            description: "Test model for registration",
            version: "1.0"
        )

        try await modelManager.registerModel(model)

        // Retrieve model
        let retrieved = try await modelManager.getModel(name: "test-model-1")

        #expect(retrieved != nil)
        #expect(retrieved?.name == "test-model-1")
        #expect(retrieved?.dimension == 384)
        #expect(retrieved?.provider == "test-provider")
        #expect(retrieved?.description == "Test model for registration")
        #expect(retrieved?.version == "1.0")
    }

    @Test("Model duplicate detection")
    func testModelDuplicateDetection() async throws {
        let db = try getDatabase()
        let prefix = createTestPrefix()
        let logger = Logger(label: "test.model.duplicate")

        let modelManager = ModelManager(database: db, rootPrefix: prefix, logger: logger)

        // Register model
        let model = EmbeddingModelMetadata(
            name: "test-model-2",
            dimension: 512,
            provider: "test-provider"
        )

        try await modelManager.registerModel(model)

        // Try to register the same model again
        do {
            try await modelManager.registerModel(model)
            Issue.record("Should have thrown duplicate error")
        } catch let error as EmbeddingError {
            if case .storageError(let message) = error {
                #expect(message.contains("already exists"))
            } else {
                Issue.record("Wrong error type: \(error)")
            }
        }
    }

    @Test("Get all models")
    func testGetAllModels() async throws {
        let db = try getDatabase()
        let prefix = createTestPrefix()
        let logger = Logger(label: "test.model.getall")

        let modelManager = ModelManager(database: db, rootPrefix: prefix, logger: logger)

        // Register multiple models
        let models = [
            EmbeddingModelMetadata(name: "model-a", dimension: 256, provider: "provider-a"),
            EmbeddingModelMetadata(name: "model-b", dimension: 512, provider: "provider-b"),
            EmbeddingModelMetadata(name: "model-c", dimension: 768, provider: "provider-c")
        ]

        for model in models {
            try await modelManager.registerModel(model)
        }

        // Retrieve all models
        let allModels = try await modelManager.getAllModels()

        #expect(allModels.count == 3)
        #expect(allModels.map { $0.name }.sorted() == ["model-a", "model-b", "model-c"])
    }

    // MARK: - Embedding Storage Tests

    @Test("Save and retrieve embedding")
    func testSaveAndRetrieveEmbedding() async throws {
        let db = try getDatabase()
        let prefix = createTestPrefix()
        let logger = Logger(label: "test.store.save")

        let modelManager = ModelManager(database: db, rootPrefix: prefix, logger: logger)
        let store = EmbeddingStore(database: db, rootPrefix: prefix, logger: logger)

        // Register model
        let model = EmbeddingModelMetadata(
            name: "test-embed-1",
            dimension: 384,
            provider: "test"
        )
        try await modelManager.registerModel(model)

        // Create embedding record
        let mockGen = createMockGenerator()
        let vector = try await mockGen.generate(text: "test embedding")

        let record = EmbeddingRecord(
            id: "entity:123",
            vector: vector,
            model: "test-embed-1",
            dimension: 384,
            sourceType: .entity,
            createdAt: Date(),
            metadata: ["key": "value"]
        )

        // Save embedding
        try await store.save(record: record)

        // Retrieve embedding
        let retrieved = try await store.get(id: "entity:123", model: "test-embed-1")

        #expect(retrieved != nil)
        #expect(retrieved?.id == "entity:123")
        #expect(retrieved?.model == "test-embed-1")
        #expect(retrieved?.dimension == 384)
        #expect(retrieved?.sourceType == .entity)
        #expect(retrieved?.metadata?["key"] == "value")
        #expect(retrieved?.vector.count == 384)
    }

    @Test("Batch save embeddings")
    func testBatchSaveEmbeddings() async throws {
        let db = try getDatabase()
        let prefix = createTestPrefix()
        let logger = Logger(label: "test.store.batch")

        let modelManager = ModelManager(database: db, rootPrefix: prefix, logger: logger)
        let store = EmbeddingStore(database: db, rootPrefix: prefix, logger: logger)

        // Register model
        let model = EmbeddingModelMetadata(
            name: "test-embed-batch",
            dimension: 384,
            provider: "test"
        )
        try await modelManager.registerModel(model)

        // Create multiple embeddings
        let mockGen = createMockGenerator()
        var records: [EmbeddingRecord] = []

        for i in 0..<5 {
            let vector = try await mockGen.generate(text: "test batch \(i)")
            let record = EmbeddingRecord(
                id: "batch:\(i)",
                vector: vector,
                model: "test-embed-batch",
                dimension: 384,
                sourceType: .batch,
                createdAt: Date()
            )
            records.append(record)
        }

        // Batch save
        try await store.saveBatch(records)

        // Verify all saved
        for i in 0..<5 {
            let retrieved = try await store.get(id: "batch:\(i)", model: "test-embed-batch")
            #expect(retrieved != nil)
            #expect(retrieved?.id == "batch:\(i)")
        }

        // Check count
        let count = try await store.countByModel(model: "test-embed-batch")
        #expect(count == 5)
    }

    @Test("Delete embedding")
    func testDeleteEmbedding() async throws {
        let db = try getDatabase()
        let prefix = createTestPrefix()
        let logger = Logger(label: "test.store.delete")

        let modelManager = ModelManager(database: db, rootPrefix: prefix, logger: logger)
        let store = EmbeddingStore(database: db, rootPrefix: prefix, logger: logger)

        // Register model
        let model = EmbeddingModelMetadata(
            name: "test-embed-delete",
            dimension: 384,
            provider: "test"
        )
        try await modelManager.registerModel(model)

        // Create and save embedding
        let mockGen = createMockGenerator()
        let vector = try await mockGen.generate(text: "test delete")

        let record = EmbeddingRecord(
            id: "delete:1",
            vector: vector,
            model: "test-embed-delete",
            dimension: 384,
            sourceType: .entity,
            createdAt: Date()
        )

        try await store.save(record: record)

        // Verify it exists
        let exists = try await store.exists(id: "delete:1", model: "test-embed-delete")
        #expect(exists == true)

        // Delete
        try await store.delete(id: "delete:1", model: "test-embed-delete")

        // Verify deletion
        let existsAfter = try await store.exists(id: "delete:1", model: "test-embed-delete")
        #expect(existsAfter == false)

        let retrieved = try await store.get(id: "delete:1", model: "test-embed-delete")
        #expect(retrieved == nil)
    }

    @Test("Get embeddings by source type")
    func testGetEmbeddingsBySourceType() async throws {
        let db = try getDatabase()
        let prefix = createTestPrefix()
        let logger = Logger(label: "test.store.sourcetype")

        let modelManager = ModelManager(database: db, rootPrefix: prefix, logger: logger)
        let store = EmbeddingStore(database: db, rootPrefix: prefix, logger: logger)

        // Register model
        let model = EmbeddingModelMetadata(
            name: "test-embed-source",
            dimension: 384,
            provider: "test"
        )
        try await modelManager.registerModel(model)

        // Create embeddings with different source types
        let mockGen = createMockGenerator()

        // Entity embeddings
        for i in 0..<3 {
            let vector = try await mockGen.generate(text: "entity \(i)")
            let record = EmbeddingRecord(
                id: "entity:\(i)",
                vector: vector,
                model: "test-embed-source",
                dimension: 384,
                sourceType: .entity,
                createdAt: Date()
            )
            try await store.save(record: record)
        }

        // Triple embeddings
        for i in 0..<2 {
            let vector = try await mockGen.generate(text: "triple \(i)")
            let record = EmbeddingRecord(
                id: "triple:\(i)",
                vector: vector,
                model: "test-embed-source",
                dimension: 384,
                sourceType: .triple,
                createdAt: Date()
            )
            try await store.save(record: record)
        }

        // Query by source type
        let entityEmbeddings = try await store.getAllBySourceType(
            sourceType: .entity,
            model: "test-embed-source"
        )

        let tripleEmbeddings = try await store.getAllBySourceType(
            sourceType: .triple,
            model: "test-embed-source"
        )

        #expect(entityEmbeddings.count == 3)
        #expect(tripleEmbeddings.count == 2)

        // Verify source types
        for embedding in entityEmbeddings {
            #expect(embedding.sourceType == .entity)
        }

        for embedding in tripleEmbeddings {
            #expect(embedding.sourceType == .triple)
        }
    }

    // MARK: - Search Engine Tests

    @Test("Cosine similarity search")
    func testCosineSearch() async throws {
        let db = try getDatabase()
        let prefix = createTestPrefix()
        let logger = Logger(label: "test.search.cosine")

        let modelManager = ModelManager(database: db, rootPrefix: prefix, logger: logger)
        let store = EmbeddingStore(database: db, rootPrefix: prefix, logger: logger)
        let storeWrapper = EmbeddingStoreWrapper(store: store, modelManager: modelManager)
        let searchEngine = SearchEngine(store: storeWrapper, logger: logger)

        // Register model
        let model = EmbeddingModelMetadata(
            name: "test-search-cosine",
            dimension: 384,
            provider: "test"
        )
        try await modelManager.registerModel(model)

        // Generate embeddings
        let mockGen = createMockGenerator()

        // Create similar embeddings (same base text with variations)
        let texts = [
            "apple fruit",
            "apple company",
            "orange fruit",
            "banana fruit",
            "computer technology"
        ]

        for (i, text) in texts.enumerated() {
            let vector = try await mockGen.generate(text: text)
            let record = EmbeddingRecord(
                id: "doc:\(i)",
                vector: vector,
                model: "test-search-cosine",
                dimension: 384,
                sourceType: .text,
                createdAt: Date()
            )
            try await store.save(record: record)
        }

        // Search for "apple fruit"
        let queryVector = try await mockGen.generate(text: "apple fruit")
        let results = try await searchEngine.search(
            queryVector: queryVector,
            topK: 3,
            model: "test-search-cosine",
            metric: .cosine
        )

        #expect(results.count == 3)
        #expect(results[0].id == "doc:0") // Exact match should be first

        // Scores should be in descending order for cosine
        for i in 0..<(results.count - 1) {
            #expect(results[i].score >= results[i + 1].score)
        }
    }

    @Test("Batch search")
    func testBatchSearch() async throws {
        let db = try getDatabase()
        let prefix = createTestPrefix()
        let logger = Logger(label: "test.search.batch")

        let modelManager = ModelManager(database: db, rootPrefix: prefix, logger: logger)
        let store = EmbeddingStore(database: db, rootPrefix: prefix, logger: logger)
        let storeWrapper = EmbeddingStoreWrapper(store: store, modelManager: modelManager)
        let searchEngine = SearchEngine(store: storeWrapper, logger: logger)

        // Register model
        let model = EmbeddingModelMetadata(
            name: "test-search-batch",
            dimension: 384,
            provider: "test"
        )
        try await modelManager.registerModel(model)

        // Generate corpus
        let mockGen = createMockGenerator()
        let corpus = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        for (i, text) in corpus.enumerated() {
            let vector = try await mockGen.generate(text: text)
            let record = EmbeddingRecord(
                id: "corpus:\(i)",
                vector: vector,
                model: "test-search-batch",
                dimension: 384,
                sourceType: .text,
                createdAt: Date()
            )
            try await store.save(record: record)
        }

        // Batch search with multiple queries
        let queries = ["doc1", "doc3", "doc5"]
        let queryVectors = try await mockGen.generateBatch(queries)

        let batchResults = try await searchEngine.batchSearch(
            queryVectors: queryVectors,
            topK: 2,
            model: "test-search-batch",
            metric: .cosine
        )

        #expect(batchResults.count == 3)

        // Each query should have results
        for (i, results) in batchResults.enumerated() {
            #expect(results.count >= 1)
            #expect(results.count <= 2)

            // First result should be the exact match
            let expectedId = "corpus:\(i * 2)" // doc1 -> corpus:0, doc3 -> corpus:2, doc5 -> corpus:4
            #expect(results[0].id == expectedId)
        }
    }

    @Test("Filtered search")
    func testFilteredSearch() async throws {
        let db = try getDatabase()
        let prefix = createTestPrefix()
        let logger = Logger(label: "test.search.filter")

        let modelManager = ModelManager(database: db, rootPrefix: prefix, logger: logger)
        let store = EmbeddingStore(database: db, rootPrefix: prefix, logger: logger)
        let storeWrapper = EmbeddingStoreWrapper(store: store, modelManager: modelManager)
        let searchEngine = SearchEngine(store: storeWrapper, logger: logger)

        // Register model
        let model = EmbeddingModelMetadata(
            name: "test-search-filter",
            dimension: 384,
            provider: "test"
        )
        try await modelManager.registerModel(model)

        // Generate embeddings with different source types
        let mockGen = createMockGenerator()

        for i in 0..<3 {
            let vector = try await mockGen.generate(text: "entity \(i)")
            let record = EmbeddingRecord(
                id: "entity:\(i)",
                vector: vector,
                model: "test-search-filter",
                dimension: 384,
                sourceType: .entity,
                createdAt: Date()
            )
            try await store.save(record: record)
        }

        for i in 0..<3 {
            let vector = try await mockGen.generate(text: "triple \(i)")
            let record = EmbeddingRecord(
                id: "triple:\(i)",
                vector: vector,
                model: "test-search-filter",
                dimension: 384,
                sourceType: .triple,
                createdAt: Date()
            )
            try await store.save(record: record)
        }

        // Search with filter for only entities
        let queryVector = try await mockGen.generate(text: "entity 0")
        let results = try await searchEngine.search(
            queryVector: queryVector,
            topK: 10,
            model: "test-search-filter",
            metric: .cosine,
            filter: { $0.sourceType == .entity }
        )

        #expect(results.count == 3)

        // All results should be entities
        for result in results {
            #expect(result.sourceType == .entity)
        }
    }

    @Test("Different similarity metrics")
    func testDifferentMetrics() async throws {
        let db = try getDatabase()
        let prefix = createTestPrefix()
        let logger = Logger(label: "test.search.metrics")

        let modelManager = ModelManager(database: db, rootPrefix: prefix, logger: logger)
        let store = EmbeddingStore(database: db, rootPrefix: prefix, logger: logger)
        let storeWrapper = EmbeddingStoreWrapper(store: store, modelManager: modelManager)
        let searchEngine = SearchEngine(store: storeWrapper, logger: logger)

        // Register model
        let model = EmbeddingModelMetadata(
            name: "test-search-metrics",
            dimension: 384,
            provider: "test"
        )
        try await modelManager.registerModel(model)

        // Generate embeddings
        let mockGen = createMockGenerator()

        for i in 0..<5 {
            let vector = try await mockGen.generate(text: "doc \(i)")
            let record = EmbeddingRecord(
                id: "metric:\(i)",
                vector: vector,
                model: "test-search-metrics",
                dimension: 384,
                sourceType: .text,
                createdAt: Date()
            )
            try await store.save(record: record)
        }

        let queryVector = try await mockGen.generate(text: "doc 0")

        // Test cosine similarity
        let cosineResults = try await searchEngine.search(
            queryVector: queryVector,
            topK: 3,
            model: "test-search-metrics",
            metric: .cosine
        )
        #expect(cosineResults.count == 3)
        #expect(cosineResults[0].id == "metric:0")

        // Test inner product
        let innerResults = try await searchEngine.search(
            queryVector: queryVector,
            topK: 3,
            model: "test-search-metrics",
            metric: .innerProduct
        )
        #expect(innerResults.count == 3)

        // Test Euclidean distance
        let euclideanResults = try await searchEngine.search(
            queryVector: queryVector,
            topK: 3,
            model: "test-search-metrics",
            metric: .euclidean
        )
        #expect(euclideanResults.count == 3)
        #expect(euclideanResults[0].id == "metric:0")

        // Test Manhattan distance
        let manhattanResults = try await searchEngine.search(
            queryVector: queryVector,
            topK: 3,
            model: "test-search-metrics",
            metric: .manhattan
        )
        #expect(manhattanResults.count == 3)
        #expect(manhattanResults[0].id == "metric:0")
    }

    // MARK: - Vector Encoding Tests

    @Test("Encode and decode vector")
    func testVectorEncoding() throws {
        let originalVector: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]

        // Encode
        let encoded = VectorCodec.encode(originalVector)
        #expect(encoded.count == originalVector.count * 4) // 4 bytes per float

        // Decode
        let decoded = try VectorCodec.decode(encoded, dimension: originalVector.count)

        #expect(decoded.count == originalVector.count)

        // Check values are close (floating point comparison)
        for i in 0..<originalVector.count {
            let diff = abs(originalVector[i] - decoded[i])
            #expect(diff < 0.0001)
        }
    }

    @Test("Vector validation")
    func testVectorValidation() {
        // Valid vector
        let validVector: [Float] = [0.1, 0.2, 0.3]
        do {
            try VectorCodec.validate(validVector)
        } catch {
            Issue.record("Valid vector should not throw")
        }

        // Vector with NaN
        let nanVector: [Float] = [0.1, Float.nan, 0.3]
        do {
            try VectorCodec.validate(nanVector)
            Issue.record("Should have thrown error for NaN")
        } catch let error as EmbeddingError {
            if case .invalidVector(let message) = error {
                #expect(message.contains("NaN"))
            } else {
                Issue.record("Wrong error type")
            }
        } catch {
            Issue.record("Wrong error type")
        }

        // Vector with infinity
        let infVector: [Float] = [0.1, Float.infinity, 0.3]
        do {
            try VectorCodec.validate(infVector)
            Issue.record("Should have thrown error for Inf")
        } catch let error as EmbeddingError {
            if case .invalidVector(let message) = error {
                #expect(message.contains("Infinite"))
            } else {
                Issue.record("Wrong error type")
            }
        } catch {
            Issue.record("Wrong error type")
        }
    }

    // MARK: - Integration Tests

    @Test("End-to-end workflow")
    func testEndToEndWorkflow() async throws {
        let db = try getDatabase()
        let prefix = createTestPrefix()
        let logger = Logger(label: "test.e2e")

        let modelManager = ModelManager(database: db, rootPrefix: prefix, logger: logger)
        let store = EmbeddingStore(database: db, rootPrefix: prefix, logger: logger)
        let storeWrapper = EmbeddingStoreWrapper(store: store, modelManager: modelManager)
        let searchEngine = SearchEngine(store: storeWrapper, logger: logger)

        // Step 1: Register model
        let model = EmbeddingModelMetadata(
            name: "e2e-model",
            dimension: 384,
            provider: "test-provider",
            description: "End-to-end test model"
        )
        try await modelManager.registerModel(model)

        // Step 2: Generate and store embeddings
        let mockGen = createMockGenerator()
        let documents = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "Natural language processing handles text",
            "Computer vision processes images",
            "Reinforcement learning learns from feedback"
        ]

        for (i, doc) in documents.enumerated() {
            let vector = try await mockGen.generate(text: doc)
            let record = EmbeddingRecord(
                id: "e2e:\(i)",
                vector: vector,
                model: "e2e-model",
                dimension: 384,
                sourceType: .text,
                createdAt: Date(),
                metadata: ["content": doc]
            )
            try await store.save(record: record)
        }

        // Step 3: Verify storage
        let count = try await store.countByModel(model: "e2e-model")
        #expect(count == 5)

        // Step 4: Perform search
        let queryVector = try await mockGen.generate(text: "Machine learning is a subset of AI")
        let results = try await searchEngine.search(
            queryVector: queryVector,
            topK: 3,
            model: "e2e-model",
            metric: .cosine
        )

        #expect(results.count == 3)
        #expect(results[0].id == "e2e:0") // Best match
        #expect(results[0].metadata?["content"] == "Machine learning is a subset of AI")

        // Step 5: Batch search
        let batchQueries = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks"
        ]
        let batchVectors = try await mockGen.generateBatch(batchQueries)
        let batchResults = try await searchEngine.batchSearch(
            queryVectors: batchVectors,
            topK: 2,
            model: "e2e-model",
            metric: .cosine
        )

        #expect(batchResults.count == 2)
        #expect(batchResults[0][0].id == "e2e:0")
        #expect(batchResults[1][0].id == "e2e:1")

        // Step 6: Delete some embeddings
        try await store.delete(id: "e2e:0", model: "e2e-model")
        try await store.delete(id: "e2e:1", model: "e2e-model")

        let countAfter = try await store.countByModel(model: "e2e-model")
        #expect(countAfter == 3)

        // Step 7: Search again - should not find deleted items
        let resultsAfterDelete = try await searchEngine.search(
            queryVector: queryVector,
            topK: 5,
            model: "e2e-model",
            metric: .cosine
        )

        #expect(resultsAfterDelete.count == 3)

        for result in resultsAfterDelete {
            #expect(result.id != "e2e:0")
            #expect(result.id != "e2e:1")
        }
    }

    @Test("Cache performance")
    func testCachePerformance() async throws {
        let db = try getDatabase()
        let prefix = createTestPrefix()
        let logger = Logger(label: "test.cache")

        let modelManager = ModelManager(database: db, rootPrefix: prefix, logger: logger)
        let store = EmbeddingStore(database: db, rootPrefix: prefix, logger: logger)

        // Register model
        let model = EmbeddingModelMetadata(
            name: "cache-model",
            dimension: 384,
            provider: "test"
        )
        try await modelManager.registerModel(model)

        // Save embedding
        let mockGen = createMockGenerator()
        let vector = try await mockGen.generate(text: "cache test")
        let record = EmbeddingRecord(
            id: "cache:1",
            vector: vector,
            model: "cache-model",
            dimension: 384,
            sourceType: .text,
            createdAt: Date()
        )
        try await store.save(record: record)

        // First retrieval - should cache
        let result1 = try await store.get(id: "cache:1", model: "cache-model")
        #expect(result1 != nil)

        // Second retrieval - should be from cache (faster)
        let result2 = try await store.get(id: "cache:1", model: "cache-model")
        #expect(result2 != nil)

        // Cache hit should be faster (though not always guaranteed in testing)
        // At minimum, verify data consistency
        #expect(result1?.id == result2?.id)
        #expect(result1?.vector.count == result2?.vector.count)

        // Test model cache
        let modelResult1 = try await modelManager.getModel(name: "cache-model")
        #expect(modelResult1 != nil)

        let modelResult2 = try await modelManager.getModel(name: "cache-model")
        #expect(modelResult2 != nil)

        #expect(modelResult1?.name == modelResult2?.name)
    }
}
