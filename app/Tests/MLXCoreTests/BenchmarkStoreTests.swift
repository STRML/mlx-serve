import XCTest
@testable import MLXCore

/// Local history + the community wire format.
///
/// The community database is append-only and openly writable in phase 1, so
/// the READER is what has to be robust: one malformed row must never blank the
/// whole board, and a row written by a future version must not break a client
/// that predates it.
final class BenchmarkStoreTests: XCTestCase {

    // MARK: - Wire format

    func testDatesRideAsISO8601SoTheWebsiteCanReadThemDirectly() throws {
        // The website parses this JSON with plain `fetch` and no SDK. Swift's
        // default Date encoding is a reference-epoch Double, which reads as a
        // meaningless number in JS and silently renders as 2001.
        let row = makeRow(arm: BenchmarkArm.defaults.id, decode: 50)
        let data = try BenchmarkStore.encoder.encode(row)
        let text = String(data: data, encoding: .utf8) ?? ""
        XCTAssertTrue(text.contains("\"date\":\""), "date is not a string — the website can't parse it")
        XCTAssertTrue(text.contains("T"), "date is not ISO8601")
    }

    func testWireRoundTripsThroughTheSharedCoders() throws {
        let row = makeRow(arm: BenchmarkArm.pld.id, decode: 61.5)
        let data = try BenchmarkStore.encoder.encode(row)
        let back = try BenchmarkStore.decoder.decode(BenchmarkResult.self, from: data)
        XCTAssertEqual(back.id, row.id)
        XCTAssertEqual(back.decodeTps, row.decodeTps, accuracy: 0.0001)
        XCTAssertEqual(back.hardware.gpuCores, row.hardware.gpuCores)
        XCTAssertEqual(back.date.timeIntervalSince1970, row.date.timeIntervalSince1970, accuracy: 1.0)
    }

    // MARK: - Reading the community database

    func testRTDBResponseIsADictionaryOfRowsNotAnArray() throws {
        // A POST to `/results.json` creates a child under a generated push key,
        // so the collection reads back as an object keyed by those pushes.
        let row = try String(data: BenchmarkStore.encoder.encode(makeRow(arm: "defaults", decode: 50)),
                             encoding: .utf8)!
        let payload = "{\"-NpushKeyA\":\(row),\"-NpushKeyB\":\(row)}"
        let rows = BenchmarkStore.decodeCommunity(Data(payload.utf8))
        XCTAssertEqual(rows.count, 2)
    }

    func testAnEmptyDatabaseReadsAsNoRowsRatherThanAnError() {
        // RTDB returns literal `null` for an empty path.
        XCTAssertTrue(BenchmarkStore.decodeCommunity(Data("null".utf8)).isEmpty)
        XCTAssertTrue(BenchmarkStore.decodeCommunity(Data("".utf8)).isEmpty)
    }

    func testOneMalformedRowDoesNotBlankTheWholeBoard() throws {
        // Anyone can write to this database in phase 1. Decoding the collection
        // as a single unit would let one junk row hide every real result.
        let good = try String(data: BenchmarkStore.encoder.encode(makeRow(arm: "defaults", decode: 50)),
                              encoding: .utf8)!
        let payload = "{\"-Ngood\":\(good),\"-Njunk\":{\"nonsense\":true}}"
        let rows = BenchmarkStore.decodeCommunity(Data(payload.utf8))
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows.first?.armId, "defaults")
    }

    func testUnknownFieldsFromAFutureVersionAreIgnored() throws {
        // Phase 2 adds attestation fields. A client shipped today must keep
        // reading rows written by a client shipped later.
        var object = try JSONSerialization.jsonObject(
            with: BenchmarkStore.encoder.encode(makeRow(arm: "defaults", decode: 50))) as! [String: Any]
        object["trust"] = "attested"
        object["attestationKeyId"] = "abc123"
        let wrapped = try JSONSerialization.data(withJSONObject: ["-Nrow": object])
        XCTAssertEqual(BenchmarkStore.decodeCommunity(wrapped).count, 1)
    }

    // MARK: - Local history

    func testMergeKeepsNewestFirstAndDedupesById() {
        // A resubmitted session must not appear twice in local history.
        let old = makeRow(arm: "defaults", decode: 50, date: Date(timeIntervalSince1970: 1_000))
        let new = makeRow(arm: "pld", decode: 60, date: Date(timeIntervalSince1970: 2_000))
        let merged = BenchmarkStore.merged([old], adding: [new, old])
        XCTAssertEqual(merged.count, 2)
        XCTAssertEqual(merged.first?.armId, "pld", "newest row is not first")
    }

    // MARK: - Aggregation

    func testCellKeyGroupsOnlyRowsThatAreActuallyComparable() {
        // Same everything except GPU cores: a 32-core and a 40-core M4 Max are
        // different machines and must not land in one median.
        let a = makeRow(arm: "defaults", decode: 50, gpuCores: 40)
        let b = makeRow(arm: "defaults", decode: 30, gpuCores: 32)
        XCTAssertNotEqual(BenchmarkStore.cellKey(a), BenchmarkStore.cellKey(b))

        let c = makeRow(arm: "defaults", decode: 52, gpuCores: 40)
        XCTAssertEqual(BenchmarkStore.cellKey(a), BenchmarkStore.cellKey(c))
    }

    func testAggregateReportsTheMedianAndTheSampleCount() {
        // n is displayed next to every median: a cell built from one submission
        // is a data point, not a benchmark.
        let rows = [
            makeRow(arm: "defaults", decode: 40),
            makeRow(arm: "defaults", decode: 50),
            makeRow(arm: "defaults", decode: 60),
        ]
        let cells = BenchmarkStore.aggregate(rows)
        XCTAssertEqual(cells.count, 1)
        XCTAssertEqual(cells.first?.decodeTps ?? 0, 50, accuracy: 0.001)
        XCTAssertEqual(cells.first?.sampleCount, 3)
    }

    func testAggregateSeparatesArmsSoLossyRowsNeverJoinALosslessMedian() {
        let rows = [
            makeRow(arm: "defaults", decode: 50),
            makeRow(arm: BenchmarkArm.kvQuant4.id, decode: 70),
        ]
        XCTAssertEqual(BenchmarkStore.aggregate(rows).count, 2)
    }

    // MARK: - Helpers

    private func makeRow(arm: String, decode: Double,
                         gpuCores: Int = 40, date: Date = Date()) -> BenchmarkResult {
        BenchmarkResult(
            sessionId: "s1",
            suiteId: BenchmarkSuite.standardV1.id,
            armId: arm,
            armLabel: arm,
            flags: [:],
            isLossy: false,
            modelId: "mlx-community/Qwen3.6-27B-4bit",
            engineVersion: "26.8.1",
            prefillTps: 900,
            decodeTps: decode,
            ttftMs: 240,
            promptTokens: 2048,
            completionTokens: 128,
            runs: 3,
            spreadPercent: 4,
            hardware: BenchmarkHardware(chip: "Apple M4 Max", gpuCores: gpuCores,
                                        ramGB: 128, osVersion: "27.0", onBattery: false),
            date: date
        )
    }
}
