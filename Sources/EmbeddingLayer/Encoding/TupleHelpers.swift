import Foundation
@preconcurrency import FoundationDB

/// Helper functions for encoding/decoding FoundationDB keys using Tuple encoding
public enum TupleHelpers {

    // MARK: - Embedding Keys

    /// Encode embedding key: (rootPrefix, "embedding", model, id)
    public static func encodeEmbeddingKey(
        rootPrefix: String,
        model: String,
        id: String
    ) -> FDB.Bytes {
        return Tuple(rootPrefix, "embedding", model, id).encode()
    }

    /// Encode embedding range prefix for a model
    public static func encodeEmbeddingRangePrefix(
        rootPrefix: String,
        model: String
    ) -> FDB.Bytes {
        return Tuple(rootPrefix, "embedding", model).encode()
    }

    // MARK: - Model Keys

    /// Encode model key: (rootPrefix, "model", modelName)
    public static func encodeModelKey(
        rootPrefix: String,
        modelName: String
    ) -> FDB.Bytes {
        return Tuple(rootPrefix, "model", modelName).encode()
    }

    /// Encode model range prefix
    public static func encodeModelRangePrefix(rootPrefix: String) -> FDB.Bytes {
        return Tuple(rootPrefix, "model").encode()
    }

    // MARK: - Index Keys

    /// Encode source type index key: (rootPrefix, "index", "source", sourceType, model, id)
    public static func encodeSourceTypeIndexKey(
        rootPrefix: String,
        sourceType: String,
        model: String,
        id: String
    ) -> FDB.Bytes {
        return Tuple(rootPrefix, "index", "source", sourceType, model, id).encode()
    }

    /// Encode source type index range prefix
    public static func encodeSourceTypeIndexRangePrefix(
        rootPrefix: String,
        sourceType: String,
        model: String
    ) -> FDB.Bytes {
        return Tuple(rootPrefix, "index", "source", sourceType, model).encode()
    }

    // MARK: - Statistics Keys

    /// Encode statistics key: (rootPrefix, "stats", model, statType)
    public static func encodeStatsKey(
        rootPrefix: String,
        model: String,
        statType: String
    ) -> FDB.Bytes {
        return Tuple(rootPrefix, "stats", model, statType).encode()
    }

    // MARK: - Range Query Helpers

    /// Create begin and end keys for range query
    public static func encodeRangeKeys(prefix: FDB.Bytes) -> (beginKey: FDB.Bytes, endKey: FDB.Bytes) {
        let beginKey = prefix
        let endKey = prefix + [0xFF]
        return (beginKey, endKey)
    }

    // MARK: - Value Encoding/Decoding

    /// Encode UInt64 as little-endian bytes
    public static func encodeUInt64(_ value: UInt64) -> FDB.Bytes {
        return withUnsafeBytes(of: value.littleEndian) { Array($0) }
    }

    /// Decode UInt64 from little-endian bytes
    public static func decodeUInt64(_ bytes: FDB.Bytes) -> UInt64 {
        return bytes.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
    }

    /// Encode Int64 as little-endian bytes
    public static func encodeInt64(_ value: Int64) -> FDB.Bytes {
        return withUnsafeBytes(of: value.littleEndian) { Array($0) }
    }

    /// Decode Int64 from little-endian bytes
    public static func decodeInt64(_ bytes: FDB.Bytes) -> Int64 {
        return bytes.withUnsafeBytes { $0.load(as: Int64.self).littleEndian }
    }
}

// MARK: - FDB.Bytes Extensions

extension FDB.Bytes {
    var utf8String: String {
        String(decoding: self, as: UTF8.self)
    }
}

extension String {
    var utf8Bytes: FDB.Bytes {
        [UInt8](self.utf8)
    }
}
