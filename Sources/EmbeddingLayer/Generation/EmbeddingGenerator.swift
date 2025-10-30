import Foundation

/// Protocol for embedding model integration
public protocol EmbeddingGenerator: Sendable {
    /// Model name
    var modelName: String { get }

    /// Vector dimension
    var dimension: Int { get }

    /// Generate embedding for text
    func generate(text: String) async throws -> [Float]

    /// Generate embeddings for multiple texts
    func generateBatch(_ texts: [String]) async throws -> [[Float]]
}

/// Default batch implementation
extension EmbeddingGenerator {
    public func generateBatch(_ texts: [String]) async throws -> [[Float]] {
        var results: [[Float]] = []
        for text in texts {
            let embedding = try await generate(text: text)
            results.append(embedding)
        }
        return results
    }
}
