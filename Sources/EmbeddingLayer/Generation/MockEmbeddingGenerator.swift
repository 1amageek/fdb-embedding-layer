import Foundation

/// Mock embedding generator for testing
public actor MockEmbeddingGenerator: EmbeddingGenerator {
    public let modelName: String
    public let dimension: Int

    public init(modelName: String = "mock-embed", dimension: Int = 384) {
        self.modelName = modelName
        self.dimension = dimension
    }

    public func generate(text: String) async throws -> [Float] {
        // Generate deterministic pseudo-random vector based on text hash
        // Use bitPattern to safely convert potentially negative hashValue to UInt64
        var generator = SeededRandomGenerator(seed: UInt64(bitPattern: Int64(text.hashValue)))
        return (0..<dimension).map { _ in Float.random(in: -1...1, using: &generator) }
    }
}

// Seeded random number generator for deterministic tests
private struct SeededRandomGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}
