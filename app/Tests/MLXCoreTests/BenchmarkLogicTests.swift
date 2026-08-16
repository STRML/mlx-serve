import XCTest
@testable import MLXCore

/// Pure logic behind the Benchmarks window.
///
/// The numbers this feature publishes are only worth anything if the
/// methodology holds, and every rule below is one that silently produces
/// plausible-but-wrong data when it's missing:
///
///  * A repeated prompt hits the server's KV prefix cache, so run 2 and 3
///    measure a cache lookup instead of prefill and the number triples.
///  * A prompt built by repeating one paragraph is exactly PLD's best case,
///    so a spec-on arm would look far better than it does on real text.
///  * Running all of arm A then all of arm B lets thermal drift land entirely
///    on B (root CLAUDE.md: interleaved A/B, same boot, drift cancels per pair).
///  * Without a `defaults` arm in every session there is no shared anchor, so
///    only absolute tok/s can be compared across machines — which is the
///    cross-version-absolute-diff trap the bench rules exist to prevent.
final class BenchmarkLogicTests: XCTestCase {

    // MARK: - Stats

    func testMedianUsesTheMiddleValueNotTheMean() {
        // A single slow run (thermal blip, background compile) must not drag
        // the reported number: that's why we publish a median, not a mean.
        XCTAssertEqual(BenchmarkStats.median([50, 51, 20]), 50, accuracy: 0.001)
        XCTAssertEqual(BenchmarkStats.median([10, 20]), 15, accuracy: 0.001)
        XCTAssertEqual(BenchmarkStats.median([42]), 42, accuracy: 0.001)
    }

    func testMedianOfNothingIsZeroRatherThanACrash() {
        // Every run discarded (all cache hits) is a real outcome the UI renders.
        XCTAssertEqual(BenchmarkStats.median([]), 0, accuracy: 0.001)
    }

    func testSpreadIsRelativeToTheMedianSoItComparesAcrossModels() {
        // 2 tok/s of spread means something different on a 3 tok/s 235B than on
        // a 200 tok/s 2B, so the published figure is a percentage.
        XCTAssertEqual(BenchmarkStats.spreadPercent([48, 50, 52]), 8, accuracy: 0.001)
        XCTAssertEqual(BenchmarkStats.spreadPercent([50, 50, 50]), 0, accuracy: 0.001)
        XCTAssertEqual(BenchmarkStats.spreadPercent([]), 0, accuracy: 0.001)
    }

    // MARK: - Prompt construction

    func testPromptIsIdenticalOnEveryMachine() {
        // Two submissions of the same suite must have run the same workload,
        // or the whole board is comparing different questions.
        XCTAssertEqual(BenchmarkPrompt.body(approxTokens: 2048),
                       BenchmarkPrompt.body(approxTokens: 2048))
    }

    func testPromptHasNoLongRepeatsSoItIsNotAFreeWinForPLD() {
        // Prompt-lookup decoding drafts from n-grams already in the context. A
        // prompt assembled by repeating one paragraph would let PLD replay it
        // verbatim and post an acceptance rate no real workload sees.
        let words = BenchmarkPrompt.body(approxTokens: 2048)
            .split(whereSeparator: { $0 == " " || $0 == "\n" })
            .map(String.init)
        XCTAssertGreaterThan(words.count, 200, "prompt too short to be a prefill measurement")

        var seen = Set<String>()
        let window = 8
        for start in 0...(words.count - window) {
            let gram = words[start..<(start + window)].joined(separator: " ")
            XCTAssertFalse(seen.contains(gram), "8-word run repeats verbatim: \(gram)")
            seen.insert(gram)
        }
    }

    func testPromptLengthTracksTheRequestedSize() {
        let small = BenchmarkPrompt.body(approxTokens: 512)
        let large = BenchmarkPrompt.body(approxTokens: 4096)
        XCTAssertLessThan(small.count, large.count)
        // Rough band only — the authoritative count is the server's `prompt_n`,
        // which every result records.
        XCTAssertGreaterThan(large.count, small.count * 4)
    }

    // MARK: - Cache defeat

    func testNonceLandsAtTheFrontBecauseTheCacheMatchesOnPrefix() {
        // The server reuses KV by prompt-PREFIX match. A nonce appended at the
        // end still shares the whole prefix, so it defeats nothing — run 2
        // would report a cache hit's prefill speed.
        let body = "shared body text that would otherwise match"
        let first = BenchmarkPrompt.nonced(body, run: 1)
        let second = BenchmarkPrompt.nonced(body, run: 2)

        XCTAssertNotEqual(first, second)
        XCTAssertFalse(first.hasPrefix(second.prefix(40)),
                       "runs share a 40-char prefix — the KV cache will match")
        XCTAssertTrue(first.hasSuffix(body), "the measured workload must be unchanged")
        XCTAssertTrue(second.hasSuffix(body))
    }

    // MARK: - Cache contamination

    func testTheTemplateHeaderAlwaysMatchesSoCachedTokensIsNeverZero() {
        // Live measurement on gemma-4-e2b, three nonced runs: prompt_n=1522,
        // cached_n=7 every time. The chat template's own header plus the
        // literal "Benchmark request " prefix are identical across runs by
        // construction, so a `cached > 0` rule discards EVERY run and the
        // whole session reports nothing. (Shipped exactly that way once.)
        XCTAssertFalse(BenchmarkPrompt.prefillWasReused(promptTokens: 1522, cachedTokens: 7))
        XCTAssertFalse(BenchmarkPrompt.prefillWasReused(promptTokens: 2048, cachedTokens: 40))
    }

    func testAWarmHitOnTheWholePromptIsStillDiscarded() {
        // The case the check exists for: the prompt was genuinely served from
        // cache, so the prefill figure describes a lookup and not compute.
        XCTAssertTrue(BenchmarkPrompt.prefillWasReused(promptTokens: 1522, cachedTokens: 1500))
        XCTAssertTrue(BenchmarkPrompt.prefillWasReused(promptTokens: 2048, cachedTokens: 2048))
    }

    func testTheThresholdIsAFractionOfThePromptNotAFixedCount() {
        // A fixed allowance can't serve both a 512-token and a 32K-token
        // suite. 10% of the prompt is far above any template header and far
        // below a real reuse.
        XCTAssertFalse(BenchmarkPrompt.prefillWasReused(promptTokens: 1000, cachedTokens: 100))
        XCTAssertTrue(BenchmarkPrompt.prefillWasReused(promptTokens: 1000, cachedTokens: 101))
    }

    func testADegenerateRunCountsAsUnusable() {
        // A zero-token prompt measured nothing; treating it as valid would
        // publish a divide-by-nothing rate.
        XCTAssertTrue(BenchmarkPrompt.prefillWasReused(promptTokens: 0, cachedTokens: 0))
    }

    // MARK: - Arms

    func testEverySessionCarriesTheDefaultsAnchor() {
        // Ratio-to-defaults is what makes results comparable across machines.
        // A session without the anchor can only contribute absolute tok/s.
        let normalized = BenchmarkPlan.normalize([BenchmarkArm.kvQuant4])
        XCTAssertEqual(normalized.first?.id, BenchmarkArm.defaults.id)
        XCTAssertEqual(normalized.count, 2)
    }

    func testDefaultsIsNotDuplicatedWhenTheUserAlreadyPickedIt() {
        let normalized = BenchmarkPlan.normalize([BenchmarkArm.defaults, BenchmarkArm.kvQuant4])
        XCTAssertEqual(normalized.filter { $0.id == BenchmarkArm.defaults.id }.count, 1)
        XCTAssertEqual(normalized.count, 2)
    }

    func testLossyArmsAreLabelledSoFastestNeverQuietlyMeansWorse() {
        // --kv-quant and --decode-attn-quant trade output quality for speed by
        // design. A "fastest config" row that doesn't say so is a
        // recommendation to degrade answers.
        XCTAssertTrue(BenchmarkArm.kvQuant4.isLossy)
        XCTAssertFalse(BenchmarkArm.defaults.isLossy)
        XCTAssertFalse(BenchmarkArm.pld.isLossy, "speculative decoding is output-preserving")
    }

    func testCatalogArmsAllCarryFlagsExceptDefaults() {
        for arm in BenchmarkArm.catalog where arm.id != BenchmarkArm.defaults.id {
            XCTAssertFalse(arm.flags.isEmpty, "\(arm.id) changes nothing — it is a duplicate of defaults")
        }
        XCTAssertTrue(BenchmarkArm.defaults.flags.isEmpty)
    }

    func testRunnableArmsNeedNoServerRestart() {
        // Interleaving alternates arms on EVERY run, so an arm that needs a
        // launch flag would cost a model reload per run — 30 s+ each on a large
        // checkpoint. Offering one would turn a 2-minute session into an hour.
        for arm in BenchmarkArm.runnable {
            XCTAssertFalse(arm.requiresRestart, "\(arm.id) would force a reload mid-session")
        }
        XCTAssertTrue(BenchmarkArm.kvQuant4.requiresRestart)
    }

    func testRunnableArmsAreAllInTheCatalog() {
        // The catalog is what labels a stored row. An arm we can run but can't
        // name would show up in history as a bare id.
        for arm in BenchmarkArm.runnable {
            XCTAssertNotNil(BenchmarkArm.byId(arm.id), "\(arm.id) is runnable but unlabelled")
        }
    }

    func testRecordedFlagsAreDerivedFromWhatTheArmActuallySends() {
        // One source of truth: an arm must not be able to record a
        // configuration different from the one it ran.
        XCTAssertEqual(BenchmarkArm.pld.flags, ["--pld": "on"])
        XCTAssertEqual(BenchmarkArm.noSpec.flags, ["--pld": "off", "--mtp": "off"])
        XCTAssertEqual(BenchmarkArm.kvQuant4.flags, ["--kv-quant": "4"])
    }

    // MARK: - Interleaving

    func testRunsAreInterleavedAcrossArmsSoThermalDriftCancels() {
        // Sequential blocks (A,A,A,B,B,B) put all of the machine's heat-up on
        // the arm that ran last, which reads as a regression that isn't there.
        let order = BenchmarkPlan.runOrder(armCount: 3, runs: 2)
        XCTAssertEqual(order.map(\.arm), [0, 1, 2, 0, 1, 2])
        XCTAssertEqual(order.map(\.run), [0, 0, 0, 1, 1, 1])
    }

    func testRunOrderCoversEveryArmAndRunExactlyOnce() {
        let order = BenchmarkPlan.runOrder(armCount: 4, runs: 3)
        XCTAssertEqual(order.count, 12)
        XCTAssertEqual(Set(order.map { "\($0.arm)-\($0.run)" }).count, 12)
    }

    func testDegenerateRunOrdersAreEmptyRatherThanCrashing() {
        XCTAssertTrue(BenchmarkPlan.runOrder(armCount: 0, runs: 3).isEmpty)
        XCTAssertTrue(BenchmarkPlan.runOrder(armCount: 3, runs: 0).isEmpty)
    }

    // MARK: - Ratios

    func testRatiosAreComputedAgainstTheSessionsOwnDefaultsArm() {
        // The point of the anchor: absolute tok/s varies with thermals and
        // background load, but the within-session ratio does not.
        let results = [
            makeResult(arm: BenchmarkArm.defaults.id, decode: 50),
            makeResult(arm: BenchmarkArm.kvQuant4.id, decode: 60),
        ]
        let ratios = BenchmarkRatios.toDefaults(results)
        XCTAssertEqual(ratios[BenchmarkArm.kvQuant4.id] ?? 0, 1.2, accuracy: 0.0001)
        XCTAssertEqual(ratios[BenchmarkArm.defaults.id] ?? 0, 1.0, accuracy: 0.0001)
    }

    func testRatiosAreEmptyWithoutAnAnchorRatherThanInventingOne() {
        // Falling back to "ratio against the fastest arm" would publish a
        // number that means something completely different under the same name.
        let ratios = BenchmarkRatios.toDefaults([makeResult(arm: BenchmarkArm.kvQuant4.id, decode: 60)])
        XCTAssertTrue(ratios.isEmpty)
    }

    func testAZeroAnchorProducesNoRatiosInsteadOfInfinity() {
        let results = [
            makeResult(arm: BenchmarkArm.defaults.id, decode: 0),
            makeResult(arm: BenchmarkArm.kvQuant4.id, decode: 60),
        ]
        XCTAssertTrue(BenchmarkRatios.toDefaults(results).isEmpty)
    }

    // MARK: - Publishability

    func testARowWithNoCompletedRunsMeasuredNothing() {
        // Every run discarded still produced an arm. Publishing it puts a
        // 0 tok/s row in the community median and shows the user a table of
        // dashes above a Share button.
        var empty = makeResult(arm: BenchmarkArm.defaults.id, decode: 0)
        empty.runs = 0
        XCTAssertFalse(empty.isPublishable)

        var oneRun = makeResult(arm: BenchmarkArm.defaults.id, decode: 42)
        oneRun.runs = 1
        XCTAssertTrue(oneRun.isPublishable)
    }

    func testAZeroRateIsNeverPublishableEvenWithRunsRecorded() {
        var broken = makeResult(arm: BenchmarkArm.defaults.id, decode: 0)
        broken.runs = 3
        XCTAssertFalse(broken.isPublishable)
    }

    // MARK: - Wire format

    func testResultSurvivesACodableRoundTrip() throws {
        // The same struct is written to disk and POSTed to RTDB; a field that
        // silently fails to decode is a row that vanishes from local history.
        let original = makeResult(arm: BenchmarkArm.kvQuant4.id, decode: 61.5)
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(BenchmarkResult.self, from: data)

        XCTAssertEqual(decoded.sessionId, original.sessionId)
        XCTAssertEqual(decoded.armId, original.armId)
        XCTAssertEqual(decoded.suiteId, original.suiteId)
        XCTAssertEqual(decoded.decodeTps, original.decodeTps, accuracy: 0.0001)
        XCTAssertEqual(decoded.flags, original.flags)
        XCTAssertEqual(decoded.chip, original.chip)
    }

    func testResultCarriesTheSchemaVersionSoPhase2CanMigrate() {
        // App Attest adds a trust field later; rows written today must be
        // identifiable as pre-attestation rather than assumed verified.
        XCTAssertEqual(makeResult(arm: BenchmarkArm.defaults.id, decode: 1).schemaVersion,
                       BenchmarkResult.currentSchemaVersion)
        XCTAssertGreaterThan(BenchmarkResult.currentSchemaVersion, 0)
    }

    // MARK: - Helpers

    private func makeResult(arm: String, decode: Double) -> BenchmarkResult {
        BenchmarkResult(
            sessionId: "session-1",
            suiteId: BenchmarkSuite.standardV1.id,
            armId: arm,
            armLabel: arm,
            flags: arm == BenchmarkArm.defaults.id ? [:] : ["--kv-quant": "4"],
            isLossy: arm != BenchmarkArm.defaults.id,
            modelId: "mlx-community/Qwen3.6-27B-4bit",
            engineVersion: "26.8.1",
            prefillTps: 900,
            decodeTps: decode,
            ttftMs: 240,
            promptTokens: 2048,
            completionTokens: 128,
            runs: 3,
            spreadPercent: 4,
            hardware: BenchmarkHardware(
                chip: "Apple M4 Max",
                gpuCores: 40,
                ramGB: 128,
                osVersion: "27.0",
                onBattery: false
            )
        )
    }
}
