import Foundation

public struct EmbeddingModelMetadata: Codable, Hashable, Sendable {
    public let name: String
    public let dimension: Int
    public let provider: String
    public let description: String?
    public let version: String?
    public let createdAt: Date

    public init(
        name: String,
        dimension: Int,
        provider: String,
        description: String? = nil,
        version: String? = nil,
        createdAt: Date = Date()
    ) {
        self.name = name
        self.dimension = dimension
        self.provider = provider
        self.description = description
        self.version = version
        self.createdAt = createdAt
    }
}
