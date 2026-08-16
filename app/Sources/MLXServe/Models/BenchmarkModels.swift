import Foundation

// Benchmark data model + the pure logic around it. Everything here is
// deterministic and testable; the runner (BenchmarkRunner) and the storage
// (BenchmarkStore) hold the IO.
//
// Design notes that are load-bearing rather than stylistic:
//
//  * A suite id pins a WORKLOAD forever. New workload = new id, so rows
//    submitted a year apart stay comparable and old rows never need migrating.
//  * A session is a group of ARMS measured back to back on one machine. The
//    within-session ratio between arms is the only comparison that survives
//    different thermals, background load and OS versions, which is why every
//    session is required to carry the `defaults` arm as its anchor.

// MARK: - Hardware

/// Coarse machine identity. Deliberately nothing that identifies a PERSON or a
/// specific unit — no serial, no host name, no account.
struct BenchmarkHardware: Codable, Hashable {
    var chip: String        // "Apple M4 Max"
    var gpuCores: Int       // 40 — the big within-tier differentiator
    var ramGB: Int
    var osVersion: String
    var onBattery: Bool     // a laptop on battery is a different machine

    static let unknown = BenchmarkHardware(
        chip: "Unknown", gpuCores: 0, ramGB: 0, osVersion: "", onBattery: false)

    /// "Apple M4 Max" → "M4". Empty for anything that isn't an Apple silicon
    /// chip: an honest blank filters correctly, a guessed family does not.
    static func chipFamily(_ brand: String) -> String {
        for token in brand.split(separator: " ") where token.count >= 2 && token.hasPrefix("M") {
            if token.dropFirst().allSatisfy(\.isNumber) { return String(token) }
        }
        return ""
    }

    /// "Apple M4 Max" → "Max", "Apple M4" → "". Filtered separately from the
    /// family because an M4 Max is a different machine from an M4.
    static func chipTier(_ brand: String) -> String {
        let tokens = brand.split(separator: " ").map(String.init)
        let family = chipFamily(brand)
        guard !family.isEmpty, let index = tokens.firstIndex(of: family),
              index + 1 < tokens.count else { return "" }
        let next = tokens[index + 1]
        return ["Pro", "Max", "Ultra"].contains(next) ? next : ""
    }

    var chipFamily: String { BenchmarkHardware.chipFamily(chip) }
    var chipTier: String { BenchmarkHardware.chipTier(chip) }

    /// Grid label. GPU cores are shown because they're the big within-tier
    /// differentiator — a 32-core and a 40-core M4 Max share a chip string and
    /// do not share a decode speed.
    var displayName: String {
        var parts = [chip]
        if gpuCores > 0 { parts.append("\(gpuCores) GPU") }
        parts.append("\(ramGB) GB")
        return parts.joined(separator: " · ")
    }
}

// MARK: - Result

/// One arm's measured result. Written to local history and POSTed verbatim to
/// the community database.
struct BenchmarkResult: Codable, Identifiable, Hashable {
    /// Bumped when a field changes meaning. Phase 2 adds attestation, and rows
    /// written now must stay identifiable as pre-attestation rather than being
    /// silently read as verified.
    static let currentSchemaVersion = 1

    var id: String = UUID().uuidString
    var schemaVersion: Int = BenchmarkResult.currentSchemaVersion

    var sessionId: String
    var suiteId: String

    var armId: String
    var armLabel: String
    var flags: [String: String]
    var isLossy: Bool

    var modelId: String
    var quant: String?
    var engineVersion: String

    var prefillTps: Double
    var decodeTps: Double
    var ttftMs: Double
    var promptTokens: Int
    var completionTokens: Int

    var runs: Int
    var spreadPercent: Double

    var hardware: BenchmarkHardware
    var date: Date = Date()

    // Flattened accessors — the grids and the website read these names.
    var chip: String { hardware.chip }
    var gpuCores: Int { hardware.gpuCores }
    var ramGB: Int { hardware.ramGB }

    init(
        id: String = UUID().uuidString,
        schemaVersion: Int = BenchmarkResult.currentSchemaVersion,
        sessionId: String,
        suiteId: String,
        armId: String,
        armLabel: String,
        flags: [String: String],
        isLossy: Bool,
        modelId: String,
        quant: String? = nil,
        engineVersion: String,
        prefillTps: Double,
        decodeTps: Double,
        ttftMs: Double,
        promptTokens: Int,
        completionTokens: Int,
        runs: Int,
        spreadPercent: Double,
        hardware: BenchmarkHardware,
        date: Date = Date()
    ) {
        self.id = id
        self.schemaVersion = schemaVersion
        self.sessionId = sessionId
        self.suiteId = suiteId
        self.armId = armId
        self.armLabel = armLabel
        self.flags = flags
        self.isLossy = isLossy
        self.modelId = modelId
        self.quant = quant
        self.engineVersion = engineVersion
        self.prefillTps = prefillTps
        self.decodeTps = decodeTps
        self.ttftMs = ttftMs
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.runs = runs
        self.spreadPercent = spreadPercent
        self.hardware = hardware
        self.date = date
    }
}

extension BenchmarkResult {
    /// A row that actually measured something.
    ///
    /// An arm whose every run was discarded still exists as a row full of
    /// zeroes. Letting one through renders a table of dashes above a Share
    /// button, and if shared it drags a 0 tok/s sample into the community
    /// median for that cell.
    var isPublishable: Bool { runs > 0 && decodeTps > 0 }
}

// MARK: - Suite

/// A named, versioned workload. One request shape yields BOTH numbers: the
/// prompt measures prefill, the capped completion measures decode.
struct BenchmarkSuite: Identifiable, Hashable {
    let id: String
    let title: String
    let detail: String
    let promptTokens: Int
    let maxTokens: Int
    let ctxSize: Int
    let runs: Int
    let warmups: Int

    static let standardV1 = BenchmarkSuite(
        id: "standard-v1",
        title: "Standard",
        detail: "2K prompt, 128 tokens out. Measures prefill and decode in one request.",
        promptTokens: 2048,
        maxTokens: 128,
        ctxSize: 4096,
        runs: 3,
        warmups: 1
    )

    static let all: [BenchmarkSuite] = [.standardV1]

    static func byId(_ id: String) -> BenchmarkSuite? { all.first { $0.id == id } }
}

// MARK: - Prompt

/// Builds the fixed workload text.
///
/// Generated rather than pasted so a new suite is a new token count instead of
/// a new 8 KB literal, and so the no-long-repeats property is guaranteed by
/// construction rather than by inspection.
enum BenchmarkPrompt {

    /// Instruction prefix — asks for a long continuation so the completion
    /// reliably reaches the suite's token cap instead of stopping early and
    /// leaving the decode measurement short.
    static let instruction = """
        Read the notes below and write a detailed technical summary. \
        Cover the trade-offs in full and keep writing until you have \
        explained every point.


        """

    /// Deterministic filler prose of roughly `approxTokens` tokens.
    ///
    /// A fixed 64-bit LCG walks a small word bank, so the text is identical on
    /// every machine (no `Date`, no `random`, no locale) while never repeating
    /// a long n-gram — a prompt built by repeating one paragraph is exactly
    /// PLD's best case and would post a speculative-decoding win no real
    /// workload sees.
    static func body(approxTokens: Int) -> String {
        let targetWords = max(16, Int(Double(approxTokens) * 0.75))
        var rng: UInt64 = 0x9E3779B97F4A7C15
        func next(_ bound: Int) -> Int {
            // xorshift64* — fixed seed, integer only, identical everywhere.
            rng ^= rng >> 12
            rng ^= rng << 25
            rng ^= rng >> 27
            let value = rng &* 0x2545F4914F6CDD1D
            return Int((value >> 33) % UInt64(bound))
        }

        var words: [String] = []
        words.reserveCapacity(targetWords)
        var sentenceLength = 0
        while words.count < targetWords {
            var word = wordBank[next(wordBank.count)]
            if sentenceLength == 0 { word = word.prefix(1).uppercased() + word.dropFirst() }
            sentenceLength += 1
            // Vary sentence length so the shape of the text isn't periodic either.
            if sentenceLength >= 9 + next(8) {
                word += "."
                sentenceLength = 0
            }
            words.append(word)
        }
        if !words[words.count - 1].hasSuffix(".") { words[words.count - 1] += "." }

        var text = instruction
        for (index, word) in words.enumerated() {
            text += word
            // Paragraph breaks keep it looking like prose rather than one blob.
            text += (index % 60 == 59) ? "\n\n" : " "
        }
        return text
    }

    /// Prefixes a per-run marker.
    ///
    /// The server reuses KV by prompt-PREFIX match, so this has to land at the
    /// FRONT. A nonce appended at the end shares the entire prefix and defeats
    /// nothing: run 2 would report the prefill speed of a cache lookup, which
    /// is several times the real number.
    static func nonced(_ body: String, run: Int) -> String {
        "Benchmark request \(run), sequence \(run &* 7919). \(body)"
    }

    /// The largest share of a prompt that may come from the KV cache before the
    /// run stops being a prefill measurement.
    ///
    /// It cannot be zero. The chat template's header and this file's own
    /// "Benchmark request " prefix are byte-identical across runs, so a few
    /// tokens ALWAYS match — measured at `cached_n = 7` of `prompt_n = 1522` on
    /// gemma-4-e2b. The server already divides `prompt_per_second` by the
    /// tokens it actually computed, so a header-sized overlap costs nothing;
    /// what has to be caught is a genuine warm hit covering the whole prompt.
    static let maxCachedFraction = 0.10

    /// True when the prefill was served from cache to a degree that invalidates
    /// the measurement.
    static func prefillWasReused(promptTokens: Int, cachedTokens: Int) -> Bool {
        guard promptTokens > 0 else { return true }   // measured nothing
        return Double(cachedTokens) > Double(promptTokens) * maxCachedFraction
    }

    private static let wordBank: [String] = [
        "memory", "bandwidth", "throughput", "latency", "kernel", "quantization",
        "attention", "context", "inference", "decode", "prefill", "cache",
        "scheduler", "tensor", "weights", "activation", "residual", "embedding",
        "router", "expert", "speculative", "acceptance", "checkpoint", "pipeline",
        "allocator", "buffer", "dispatch", "occupancy", "register", "threadgroup",
        "precision", "rounding", "accumulator", "reduction", "gradient", "sampling",
        "temperature", "token", "vocabulary", "sequence", "batch", "window",
        "sliding", "recurrent", "convolution", "normalization", "projection", "matrix",
        "product", "vector", "lookup", "table", "compression", "ratio",
        "measured", "observed", "baseline", "regression", "variance", "median",
        "thermal", "sustained", "peak", "budget", "ceiling", "pressure",
        "engine", "runtime", "backend", "device", "unified", "resident",
        "streaming", "chunked", "parallel", "serial", "concurrent", "queue",
    ]
}

// MARK: - Arms

/// One configuration measured inside a session.
///
/// Split by HOW the setting is applied, because it decides what a session can
/// afford to measure. Speculation is a per-REQUEST body field, so those arms
/// interleave for free. `--kv-quant` and friends are launch flags, and since
/// interleaving alternates arms on every run, offering them would mean a model
/// reload per run (30 s+ each on a large checkpoint). Phase 1 runs the
/// per-request arms only; the launch-flag arms are defined so the grids and
/// the website can label rows a later version submits.
struct BenchmarkArm: Identifiable, Hashable {
    let id: String
    let label: String
    let detail: String

    /// Per-request body overrides. `nil` = leave the field out and let the
    /// server's own default win.
    let enablePLD: Bool?
    let enableMTP: Bool?

    /// Settings that can only be applied at server launch. Empty = runnable
    /// without restarting anything.
    let launchFlags: [String: String]

    init(id: String, label: String, detail: String,
         enablePLD: Bool? = nil, enableMTP: Bool? = nil,
         launchFlags: [String: String] = [:]) {
        self.id = id
        self.label = label
        self.detail = detail
        self.enablePLD = enablePLD
        self.enableMTP = enableMTP
        self.launchFlags = launchFlags
    }

    /// Canonical recorded representation — what gets stored, filtered and
    /// displayed. Derived from the overrides so there is ONE source of truth:
    /// an arm can't record a configuration different from the one it ran.
    var flags: [String: String] {
        var out = launchFlags
        if let pld = enablePLD { out["--pld"] = pld ? "on" : "off" }
        if let mtp = enableMTP { out["--mtp"] = mtp ? "on" : "off" }
        return out
    }

    var requiresRestart: Bool { !launchFlags.isEmpty }

    /// True when the arm trades output quality for speed.
    ///
    /// `--kv-quant` and `--decode-attn-quant` are lossy by design, so a board
    /// that ranks by speed alone would quietly recommend degrading answers.
    /// Speculative decoding is NOT lossy — it reproduces the same tokens.
    var isLossy: Bool { BenchmarkArm.lossy(flags) }

    static func lossy(_ flags: [String: String]) -> Bool {
        if let kv = flags["--kv-quant"], kv != "off" { return true }
        if flags["--decode-attn-quant"] != nil { return true }
        return false
    }

    static let defaults = BenchmarkArm(
        id: "defaults",
        label: "Defaults",
        detail: "The server's shipping configuration. Anchors every comparison."
    )

    static let pld = BenchmarkArm(
        id: "pld",
        label: "PLD on",
        detail: "Prompt-lookup speculative decoding. Output-preserving.",
        enablePLD: true
    )

    static let noSpec = BenchmarkArm(
        id: "no-spec",
        label: "No speculation",
        detail: "PLD and MTP off — the plain autoregressive floor.",
        enablePLD: false, enableMTP: false
    )

    static let kvQuant4 = BenchmarkArm(
        id: "kv-quant-4",
        label: "KV quant 4-bit",
        detail: "Smaller KV cache, more context per GB. Lossy.",
        launchFlags: ["--kv-quant": "4"]
    )

    static let kvQuant8 = BenchmarkArm(
        id: "kv-quant-8",
        label: "KV quant 8-bit",
        detail: "Smaller KV cache at higher precision than 4-bit. Lossy.",
        launchFlags: ["--kv-quant": "8"]
    )

    /// What a session can actually run today: no restarts, so interleaving is
    /// free and a full session is a handful of requests.
    static let runnable: [BenchmarkArm] = [defaults, noSpec, pld]

    /// Every arm the schema knows about, including ones only a later version
    /// will run. Used to label community rows, never to offer a control.
    static let catalog: [BenchmarkArm] = [defaults, noSpec, pld, kvQuant8, kvQuant4]

    static func byId(_ id: String) -> BenchmarkArm? { catalog.first { $0.id == id } }
}

// MARK: - Plan

enum BenchmarkPlan {

    /// Guarantees the `defaults` anchor is present exactly once, first.
    ///
    /// Without it a session contributes only absolute tok/s, which is the
    /// cross-machine comparison the bench rules forbid.
    static func normalize(_ arms: [BenchmarkArm]) -> [BenchmarkArm] {
        var seen = Set<String>([BenchmarkArm.defaults.id])
        var out = [BenchmarkArm.defaults]
        for arm in arms where !seen.contains(arm.id) {
            seen.insert(arm.id)
            out.append(arm)
        }
        return out
    }

    /// Interleaved execution order: every arm runs once, then every arm again.
    ///
    /// Running arm A's three runs before arm B's puts the machine's whole
    /// warm-up on A and its heat on B, which reads as a regression that isn't
    /// there. Interleaving spreads drift evenly so it cancels in the ratio.
    static func runOrder(armCount: Int, runs: Int) -> [(arm: Int, run: Int)] {
        guard armCount > 0, runs > 0 else { return [] }
        var order: [(arm: Int, run: Int)] = []
        order.reserveCapacity(armCount * runs)
        for run in 0..<runs {
            for arm in 0..<armCount { order.append((arm: arm, run: run)) }
        }
        return order
    }
}

// MARK: - Ratios

enum BenchmarkRatios {

    /// Decode speed of each arm relative to the session's own `defaults` arm.
    ///
    /// This is the figure that aggregates honestly across machines: absolute
    /// tok/s moves with thermals, background load and OS version, but the
    /// same-session ratio does not. Returns empty when there is no usable
    /// anchor — falling back to "ratio against the fastest arm" would publish
    /// a different quantity under the same name.
    static func toDefaults(_ results: [BenchmarkResult]) -> [String: Double] {
        guard let anchor = results.first(where: { $0.armId == BenchmarkArm.defaults.id }),
              anchor.decodeTps > 0 else { return [:] }
        var out: [String: Double] = [:]
        for result in results where result.decodeTps > 0 || result.armId == anchor.armId {
            out[result.armId] = result.decodeTps / anchor.decodeTps
        }
        return out
    }
}

// MARK: - Stats

enum BenchmarkStats {

    /// Middle value. A single slow run (thermal blip, a background build) must
    /// not drag the published number, which a mean would let it do.
    static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 1 { return sorted[mid] }
        return (sorted[mid - 1] + sorted[mid]) / 2
    }

    /// (max − min) as a percentage of the median.
    ///
    /// Relative because 2 tok/s of spread means something very different on a
    /// 3 tok/s 235B than on a 200 tok/s 2B. A wide spread is what tells the
    /// user (and later, the validator) that the run wasn't clean.
    static func spreadPercent(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0 }
        let mid = median(values)
        guard mid > 0, let low = values.min(), let high = values.max() else { return 0 }
        return (high - low) / mid * 100
    }
}
