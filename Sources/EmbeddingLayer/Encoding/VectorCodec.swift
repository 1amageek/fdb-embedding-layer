import Foundation
@preconcurrency import FoundationDB

/// Efficient vector encoding/decoding for FoundationDB storage
public enum VectorCodec {

    /// Encode Float32 vector to bytes (4 bytes per dimension)
    public static func encode(_ vector: [Float]) -> FDB.Bytes {
        var bytes = FDB.Bytes()
        bytes.reserveCapacity(vector.count * 4)

        for value in vector {
            let bits = value.bitPattern
            bytes.append(contentsOf: withUnsafeBytes(of: bits.littleEndian) { Array($0) })
        }

        return bytes
    }

    /// Decode bytes to Float32 vector
    public static func decode(_ bytes: FDB.Bytes, dimension: Int) throws -> [Float] {
        guard bytes.count == dimension * 4 else {
            throw EmbeddingError.encodingError(
                "Invalid byte count: expected \(dimension * 4), got \(bytes.count)"
            )
        }

        var vector = [Float]()
        vector.reserveCapacity(dimension)

        for i in 0..<dimension {
            let offset = i * 4
            let bits = bytes[offset..<offset+4].withUnsafeBytes {
                $0.load(as: UInt32.self).littleEndian
            }
            vector.append(Float(bitPattern: bits))
        }

        return vector
    }

    /// Validate vector (check for NaN, Inf)
    public static func validate(_ vector: [Float]) throws {
        for (index, value) in vector.enumerated() {
            if value.isNaN {
                throw EmbeddingError.invalidVector("NaN at index \(index)")
            }
            if value.isInfinite {
                throw EmbeddingError.invalidVector("Infinite value at index \(index)")
            }
        }
    }

    /// Normalize vector to unit length
    public static func normalize(_ vector: [Float]) -> [Float] {
        let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        guard norm > 0 else { return vector }
        return vector.map { $0 / norm }
    }
}
