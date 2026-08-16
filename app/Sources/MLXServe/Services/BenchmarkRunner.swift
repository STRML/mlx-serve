import Foundation

/// Runs a benchmark session: one suite, several arms, interleaved.
///
/// The methodology rules live in `BenchmarkModels` (interleaving, the defaults
/// anchor, the prompt nonce). This type owns the IO and the two live checks
/// that can only be made while a run is happening:
///
///  * a run whose prompt hit the KV prefix cache is DISCARDED — its prefill
///    figure measured a cache lookup and is several times the real number;
///  * a run that stopped before the token cap is kept but recorded, because a
///    short completion makes the decode figure noisier.
@MainActor
final class BenchmarkRunner: ObservableObject {

    enum Phase: Equatable {
        case idle
        case warmup(arm: String)
        case running(arm: String, run: Int, of: Int)
        case done
        case failed(String)
    }

    struct Progress: Equatable {
        var completed: Int
        var total: Int
        var fraction: Double { total > 0 ? Double(completed) / Double(total) : 0 }
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var progress = Progress(completed: 0, total: 0)
    /// Runs thrown away because the prompt hit the prefix cache. Surfaced so a
    /// session that silently measured nothing can't look like a clean result.
    @Published private(set) var discardedRuns = 0

    private let api: APIClient
    init(api: APIClient = APIClient()) { self.api = api }

    /// Measure `arms` on `suite`. Returns one result per arm.
    ///
    /// `normalize` guarantees the defaults anchor is present, so the caller
    /// cannot accidentally submit a session whose ratios mean nothing.
    func run(
        suite: BenchmarkSuite,
        arms requested: [BenchmarkArm],
        modelId: String,
        port: UInt16,
        engineVersion: String,
        hardware: BenchmarkHardware = SystemMetrics.benchmarkHardware()
    ) async throws -> [BenchmarkResult] {
        let arms = BenchmarkPlan.normalize(requested)
        let order = BenchmarkPlan.runOrder(armCount: arms.count, runs: suite.runs)
        let body = BenchmarkPrompt.body(approxTokens: suite.promptTokens)
        let sessionId = UUID().uuidString

        discardedRuns = 0
        progress = Progress(completed: 0, total: order.count + arms.count * suite.warmups)

        // Per-arm samples, keyed by arm id.
        var prefill: [String: [Double]] = [:]
        var decode: [String: [Double]] = [:]
        var ttft: [String: [Double]] = [:]
        var lastPromptTokens: [String: Int] = [:]
        var lastCompletionTokens: [String: Int] = [:]

        // Warmups: paged-in weights and a settled clock. Never recorded.
        // Each still gets a distinct nonce, or the warmup would prime the very
        // cache the measured runs are trying to miss.
        var nonce = 0
        for arm in arms {
            for _ in 0..<suite.warmups {
                nonce += 1
                phase = .warmup(arm: arm.label)
                _ = try? await request(suite: suite, body: body, nonce: nonce,
                                       arm: arm, modelId: modelId, port: port)
                progress.completed += 1
            }
        }

        for step in order {
            let arm = arms[step.arm]
            nonce += 1
            phase = .running(arm: arm.label, run: step.run + 1, of: suite.runs)

            let timings: APIClient.CompletionTimings
            do {
                timings = try await request(suite: suite, body: body, nonce: nonce,
                                            arm: arm, modelId: modelId, port: port)
            } catch {
                phase = .failed(error.localizedDescription)
                throw error
            }
            progress.completed += 1

            // The one check that can only be made live. A warm hit reports a
            // prefill speed that never happened. Note this is a FRACTION of the
            // prompt, not `cached > 0` — the template header always matches.
            guard !BenchmarkPrompt.prefillWasReused(promptTokens: timings.promptTokens,
                                                    cachedTokens: timings.cachedTokens) else {
                discardedRuns += 1
                continue
            }

            prefill[arm.id, default: []].append(timings.prefillTps)
            decode[arm.id, default: []].append(timings.decodeTps)
            ttft[arm.id, default: []].append(timings.ttftMs)
            lastPromptTokens[arm.id] = timings.promptTokens
            lastCompletionTokens[arm.id] = timings.completionTokens
        }

        let results = arms.compactMap { arm -> BenchmarkResult? in
            let row = BenchmarkResult(
                sessionId: sessionId,
                suiteId: suite.id,
                armId: arm.id,
                armLabel: arm.label,
                flags: arm.flags,
                isLossy: arm.isLossy,
                modelId: modelId,
                engineVersion: engineVersion,
                prefillTps: BenchmarkStats.median(prefill[arm.id] ?? []),
                decodeTps: BenchmarkStats.median(decode[arm.id] ?? []),
                ttftMs: BenchmarkStats.median(ttft[arm.id] ?? []),
                promptTokens: lastPromptTokens[arm.id] ?? 0,
                completionTokens: lastCompletionTokens[arm.id] ?? 0,
                runs: (decode[arm.id] ?? []).count,
                spreadPercent: BenchmarkStats.spreadPercent(decode[arm.id] ?? []),
                hardware: hardware
            )
            // An arm whose every run was discarded measured nothing. Returning
            // it anyway is how a table of dashes ends up above a Share button.
            return row.isPublishable ? row : nil
        }
        phase = .done
        return results
    }

    private func request(
        suite: BenchmarkSuite,
        body: String,
        nonce: Int,
        arm: BenchmarkArm,
        modelId: String,
        port: UInt16
    ) async throws -> APIClient.CompletionTimings {
        try await api.benchmarkCompletion(
            port: port,
            model: modelId,
            prompt: BenchmarkPrompt.nonced(body, run: nonce),
            maxTokens: suite.maxTokens,
            enablePLD: arm.enablePLD,
            enableMTP: arm.enableMTP
        )
    }
}
