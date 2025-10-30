import Foundation

public enum EmbeddingError: Error, LocalizedError, Sendable {
    case modelNotFound(String)
    case dimensionMismatch(expected: Int, actual: Int)
    case vectorNotFound(String)
    case invalidVector(String)
    case storageError(String)
    case encodingError(String)
    case invalidModel(String)
    case searchError(String)
    case generationError(String)
    case cacheError(String)
    case validationError(String)
    case migrationError(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Model '\(name)' not found in registry"
        case .dimensionMismatch(let expected, let actual):
            return "Dimension mismatch: expected \(expected), got \(actual)"
        case .vectorNotFound(let id):
            return "Vector not found: \(id)"
        case .invalidVector(let message):
            return "Invalid vector: \(message)"
        case .storageError(let message):
            return "Storage error: \(message)"
        case .encodingError(let message):
            return "Encoding error: \(message)"
        case .invalidModel(let message):
            return "Invalid model: \(message)"
        case .searchError(let message):
            return "Search error: \(message)"
        case .generationError(let message):
            return "Generation error: \(message)"
        case .cacheError(let message):
            return "Cache error: \(message)"
        case .validationError(let message):
            return "Validation error: \(message)"
        case .migrationError(let message):
            return "Migration error: \(message)"
        }
    }
}
