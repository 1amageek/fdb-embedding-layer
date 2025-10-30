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

    public var recoverySuggestion: String? {
        switch self {
        case .modelNotFound:
            return "Verify the model name is correct and registered. Check available models using the model registry."
        case .dimensionMismatch:
            return "Ensure the vector dimension matches the model's expected dimension. Regenerate the embedding if necessary."
        case .vectorNotFound:
            return "Verify the embedding ID is correct, or generate the embedding using save() before retrieving it."
        case .invalidVector:
            return "Check that the vector contains valid Float values (no NaN or Inf) and has the correct dimension."
        case .storageError:
            return "Check FoundationDB connectivity and storage health. Retry the operation if it's transient."
        case .encodingError:
            return "This is an internal encoding error. Ensure vector data is valid and serializable."
        case .invalidModel:
            return "Verify model configuration and ensure the model is properly initialized."
        case .searchError:
            return "Check search parameters and ensure the embedding store contains sufficient data for similarity search."
        case .generationError:
            return "Verify the input text is valid and the embedding model is accessible. Check model API connectivity."
        case .cacheError:
            return "Clear the cache using clearCache() and retry. This may be a transient cache corruption issue."
        case .validationError:
            return "Review the embedding record fields and ensure all required data is provided with correct types."
        case .migrationError:
            return "Check migration script compatibility and database schema version. Restore from backup if needed."
        }
    }
}

// MARK: - CustomStringConvertible

extension EmbeddingError: CustomStringConvertible {
    public var description: String {
        return errorDescription ?? "Unknown embedding error"
    }
}
