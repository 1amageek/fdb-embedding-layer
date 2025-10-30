import Foundation

public struct EmbeddingRecord: Codable, Hashable, Sendable {
    public let id: String
    public let vector: [Float]
    public let model: String
    public let dimension: Int
    public let sourceType: SourceType
    public let createdAt: Date
    public let updatedAt: Date?
    public let metadata: [String: String]?

    public init(
        id: String,
        vector: [Float],
        model: String,
        dimension: Int,
        sourceType: SourceType,
        createdAt: Date,
        updatedAt: Date? = nil,
        metadata: [String: String]? = nil
    ) {
        self.id = id
        self.vector = vector
        self.model = model
        self.dimension = dimension
        self.sourceType = sourceType
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.metadata = metadata
    }
}

public enum SourceType: String, Codable, Hashable, Sendable {
    case entity
    case triple
    case text
    case batch
}
