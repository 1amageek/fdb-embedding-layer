import Foundation

public struct SearchResult: Codable, Hashable, Sendable {
    public let id: String
    public let score: Float
    public let vector: [Float]?
    public let sourceType: SourceType?
    public let metadata: [String: String]?

    public init(
        id: String,
        score: Float,
        vector: [Float]? = nil,
        sourceType: SourceType? = nil,
        metadata: [String: String]? = nil
    ) {
        self.id = id
        self.score = score
        self.vector = vector
        self.sourceType = sourceType
        self.metadata = metadata
    }
}
