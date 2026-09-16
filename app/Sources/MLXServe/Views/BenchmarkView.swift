import SwiftUI

/// The Benchmarks window: run a suite, keep your own history, compare against
/// what everyone else measured.
///
/// Three panes, because they answer three different questions — "how fast is
/// this?", "did my change help?", and "how does my Mac compare?".
///
/// The benchmark measures whatever model the server currently has loaded. That
/// is deliberate: picking a model here would mean re-implementing load/unload
/// orchestration for a window whose job is measurement, and the tray already
/// owns that.
///
/// Layout notes: the pane switcher rides the TOOLBAR (the native macOS place
/// for a view switcher — Finder, Mail), each pane owns its own scrolling
/// because `Table` scrolls itself and nesting it in a ScrollView breaks its
/// sizing, and the result is a row of stat tiles rather than a text table so
/// the number the user came for is the biggest thing on screen.
struct BenchmarkView: View {

    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var server: ServerManager

    @StateObject private var runner = BenchmarkRunner()

    @State private var pane: Pane = .run
    @State private var selectedArmIds: Set<String> = [BenchmarkArm.defaults.id, BenchmarkArm.pld.id]
    @State private var lastResults: [BenchmarkResult] = []
    @State private var history: [BenchmarkResult] = []
    @State private var community: [BenchmarkResult] = []
    @State private var communityLoading = false
    @State private var communityError: String?
    @State private var submitState: SubmitState = .idle
    @State private var runError: String?
    @State private var isRunning = false
    @State private var ranAtLeastOnce = false

    /// Community filter: only rows from a machine like this one. The whole
    /// point of the board is "what will I get", so it defaults to ON.
    @State private var onlyMyChip = true

    private let client = BenchmarkCommunityClient()
    private let suite = BenchmarkSuite.standardV1

    /// Recorded on every row so the board can show which build produced a
    /// number. Deliberately not part of the aggregation key — see
    /// `BenchmarkStore.cellKey`.
    static var appVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0.0.0"
    }

    /// Named `Pane`, not `Section` — a nested `Section` inside a View shadows
    /// SwiftUI's own and turns any later `Section { }` here into a baffling
    /// type error.
    enum Pane: String, CaseIterable, Identifiable {
        case run = "Run"
        case history = "History"
        case community = "Community"
        var id: String { rawValue }
        var icon: String {
            switch self {
            case .run: return "play.circle"
            case .history: return "clock"
            case .community: return "person.2"
            }
        }
    }

    enum SubmitState: Equatable {
        case idle, sending, sent, failed(String)
    }

    /// Read once. `benchmarkHardware()` iterates the IORegistry, and a view
    /// body re-evaluates constantly — this must never be called from one.
    private let hardware = SystemMetrics.benchmarkHardware()

    var body: some View {
        Group {
            switch pane {
            case .run: runPane
            case .history: historyPane
            case .community: communityPane
            }
        }
        // Wide enough for the 390pt setup column plus a result column that can
        // hold three stat tiles without crushing them.
        .frame(minWidth: 880, minHeight: 560)
        .background(.background)
        .toolbar {
            ToolbarItem(placement: .principal) {
                // Text, not Label: a segmented picker collapses Labels to
                // icon-only, and "Run" / "History" / "Community" have no
                // glyphs anyone would read correctly without the word.
                Picker("View", selection: $pane) {
                    ForEach(Pane.allCases) { pane in
                        Text(pane.rawValue).tag(pane)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
            }
        }
        .onAppear { history = BenchmarkStore.loadLocal() }
        .task(id: pane) {
            if pane == .community && community.isEmpty { await loadCommunity() }
        }
    }

    // MARK: - Run

    /// Two columns: what you're about to run on the left, what came out on the
    /// right. A single tall column meant scrolling past the setup every time to
    /// reach the number you ran it for.
    private var runPane: some View {
        ScrollView {
            HStack(alignment: .top, spacing: 18) {
                VStack(spacing: 18) {
                    setupCard
                    compareCard
                    runControl
                }
                .frame(width: 390)

                VStack(spacing: 18) {
                    if !lastResults.isEmpty {
                        resultCard
                    } else if ranAtLeastOnce && !isRunning {
                        noUsableRunsCard
                    } else {
                        resultPlaceholder
                    }
                }
                .frame(maxWidth: .infinity, alignment: .top)
            }
            .padding(20)
        }
    }

    /// An empty right column reads as a rendering bug. A dashed well says the
    /// results are going to land here.
    private var resultPlaceholder: some View {
        VStack(spacing: 10) {
            Image(systemName: "chart.bar")
                .font(.system(size: 32, weight: .light))
                .foregroundStyle(.tertiary)
            Text("No result yet")
                .font(.callout.weight(.medium))
                .foregroundStyle(.secondary)
            Text("Run the benchmark to see prefill and decode speed for each setting.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 230)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 64)
        .background(.quaternary.opacity(0.14), in: RoundedRectangle(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(.quaternary, style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
        }
    }

    private var setupCard: some View {
        BenchCard("Setup", icon: "gearshape") {
            VStack(spacing: 0) {
                BenchRow("Suite", detail: suite.detail) {
                    Text(suite.title).fontWeight(.medium)
                }
                Divider().padding(.vertical, 9)
                BenchRow("Model") {
                    Text(server.residentChatModel?.name ?? "No chat model loaded")
                        .foregroundStyle(server.residentChatModel == nil ? .secondary : .primary)
                        .lineLimit(1).truncationMode(.middle)
                }
                Divider().padding(.vertical, 9)
                BenchRow("This Mac") {
                    Text(hardware.displayName)
                }
            }
        }
    }

    private var compareCard: some View {
        BenchCard("Compare", icon: "square.split.2x1",
                  footnote: "Every session measures Defaults so results stay comparable across machines. Arms run interleaved, so warm-up drift lands evenly.") {
            // Explicit HStack + Spacer rather than Toggle's own label slot:
            // a `.switch` Toggle sizes to its content and centres the pair, so
            // three rows of different label lengths come out ragged at BOTH
            // edges. This flushes labels left and switches right.
            VStack(spacing: 0) {
                ForEach(Array(BenchmarkArm.runnable.enumerated()), id: \.element.id) { index, arm in
                    if index > 0 { Divider().padding(.vertical, 10) }
                    HStack(spacing: 12) {
                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 6) {
                                Text(arm.label).fontWeight(.medium)
                                if arm.isLossy { LossyBadge() }
                            }
                            Text(arm.id == BenchmarkArm.defaults.id
                                 ? "\(arm.detail) Always included."
                                 : arm.detail)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        Spacer(minLength: 12)
                        Toggle("", isOn: armBinding(arm))
                            .labelsHidden()
                            .toggleStyle(.switch)
                            .controlSize(.small)
                            .disabled(arm.id == BenchmarkArm.defaults.id)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }

    private var runControl: some View {
        VStack(spacing: 12) {
            if isRunning {
                VStack(spacing: 8) {
                    ProgressView(value: runner.progress.fraction)
                        .progressViewStyle(.linear)
                    HStack {
                        Text(phaseDescription)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                        Spacer()
                        if runner.discardedRuns > 0 {
                            Label("\(runner.discardedRuns) discarded", systemImage: "exclamationmark.triangle.fill")
                                .font(.caption)
                                .foregroundStyle(.orange)
                        }
                    }
                }
            }

            if let runError {
                Label(runError, systemImage: "exclamationmark.octagon.fill")
                    .font(.callout)
                    .foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            // Natural width, centred. A full-width primary button reads iOS;
            // macOS sizes an action button to its title.
            Button {
                Task { await runBenchmark() }
            } label: {
                Label(isRunning ? "Running…" : "Run Benchmark", systemImage: "play.fill")
                    .padding(.horizontal, 8)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(isRunning || server.status != .running || server.residentChatModel == nil)

            if server.residentChatModel == nil {
                Text("Load a chat model from the menu bar to run a benchmark.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var resultCard: some View {
        let ratios = BenchmarkRatios.toDefaults(lastResults)
        return BenchCard("Result", icon: "chart.bar.fill") {
            VStack(spacing: 16) {
                ForEach(Array(lastResults.enumerated()), id: \.element.id) { index, row in
                    if index > 0 { Divider() }
                    VStack(spacing: 10) {
                        HStack(spacing: 6) {
                            Text(row.armLabel).font(.headline)
                            if row.isLossy { LossyBadge() }
                            Spacer()
                            if let ratio = ratios[row.armId], row.armId != BenchmarkArm.defaults.id {
                                RatioBadge(ratio: ratio)
                            }
                            if row.runs > 1 {
                                Text("±\(Int(row.spreadPercent))%")
                                    .font(.caption)
                                    .foregroundStyle(row.spreadPercent > 15 ? .orange : .secondary)
                                    .help("Spread across \(row.runs) runs")
                            }
                        }
                        HStack(spacing: 10) {
                            StatTile(value: formatted(row.decodeTps, decimals: 1),
                                     unit: "tok/s", label: "Decode", emphasis: true)
                            StatTile(value: formatted(row.prefillTps, decimals: 0),
                                     unit: "tok/s", label: "Prefill")
                            StatTile(value: formatted(row.ttftMs, decimals: 0),
                                     unit: "ms", label: "TTFT")
                        }
                    }
                }

                Divider()

                HStack(alignment: .top, spacing: 12) {
                    switch submitState {
                    case .idle:
                        Button {
                            Task { await submit() }
                        } label: {
                            Label("Share to Community", systemImage: "square.and.arrow.up")
                        }
                        .controlSize(.regular)
                    case .sending:
                        ProgressView().controlSize(.small)
                    case .sent:
                        Label("Shared", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    case .failed(let message):
                        Label(message, systemImage: "exclamationmark.octagon.fill")
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                    Spacer(minLength: 0)
                    Text("Sends these numbers plus your chip, GPU cores, memory and macOS version. No account, nothing identifying.")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.trailing)
                        .frame(maxWidth: 260)
                }
            }
        }
    }

    private var noUsableRunsCard: some View {
        BenchCard("Result", icon: "chart.bar") {
            VStack(alignment: .leading, spacing: 6) {
                Label("No usable runs", systemImage: "exclamationmark.triangle.fill")
                    .font(.callout.weight(.medium))
                    .foregroundStyle(.orange)
                Text("Every run was served from the KV cache, so nothing was actually measured. Restarting the server clears it.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: - History

    private var historyPane: some View {
        Group {
            if history.isEmpty {
                ContentUnavailableView {
                    Label("No Runs Yet", systemImage: "clock")
                } description: {
                    Text("Benchmarks you run are kept here, on this Mac.")
                } actions: {
                    Button("Run a Benchmark") { pane = .run }
                        .buttonStyle(.borderedProminent)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 0) {
                    Table(history) {
                        TableColumn("Date") { row in
                            Text(row.date, format: .dateTime.month(.abbreviated).day().hour().minute())
                                .foregroundStyle(.secondary)
                        }
                        .width(min: 100, ideal: 120)

                        TableColumn("Model") { row in
                            Text(row.modelId).lineLimit(1).truncationMode(.middle)
                        }
                        .width(min: 140, ideal: 220)

                        TableColumn("Setting") { row in
                            HStack(spacing: 4) {
                                Text(row.armLabel)
                                if row.isLossy { LossyBadge() }
                            }
                        }
                        .width(min: 90, ideal: 110)

                        TableColumn("Prefill") { row in
                            Text(formatted(row.prefillTps, decimals: 0)).monospacedDigit()
                        }
                        .width(min: 60, ideal: 74)

                        TableColumn("Decode") { row in
                            Text(formatted(row.decodeTps, decimals: 1))
                                .monospacedDigit().fontWeight(.medium)
                        }
                        .width(min: 60, ideal: 74)
                    }
                    .tableStyle(.inset(alternatesRowBackgrounds: true))

                    footerBar {
                        Text("\(history.count) result\(history.count == 1 ? "" : "s") · tok/s")
                        Spacer()
                        Button(role: .destructive) {
                            BenchmarkStore.saveLocal([])
                            history = []
                        } label: {
                            Label("Clear", systemImage: "trash")
                        }
                        .controlSize(.small)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            }
        }
    }

    // MARK: - Community

    private var communityPane: some View {
        // maxHeight + .top: without it the VStack sizes to its content and
        // macOS centres the whole thing vertically, which floats the filter
        // bar into the middle of an otherwise empty window.
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Toggle("Machines like mine", isOn: $onlyMyChip)
                    .toggleStyle(.switch)
                    .controlSize(.small)
                Text(hardware.chip)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    Task { await loadCommunity() }
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .controlSize(.small)
                .disabled(communityLoading)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(.bar)

            Divider()

            if communityLoading {
                ProgressView("Loading results…")
                    .controlSize(.small)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let communityError {
                ContentUnavailableView {
                    Label("Couldn't Load Results", systemImage: "wifi.exclamationmark")
                } description: {
                    Text(communityError)
                } actions: {
                    Button("Try Again") { Task { await loadCommunity() } }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if visibleCommunityCells.isEmpty {
                ContentUnavailableView {
                    Label("Nothing Here Yet", systemImage: "person.2")
                } description: {
                    Text(onlyMyChip
                         ? "No results from a machine like yours yet. Turn off the filter to see everything."
                         : "Be the first to share a result.")
                } actions: {
                    if onlyMyChip {
                        Button("Show All Machines") { onlyMyChip = false }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                Table(visibleCommunityCells) {
                    TableColumn("Machine") { cell in
                        Text(cell.hardware.displayName).lineLimit(1)
                    }
                    .width(min: 150, ideal: 200)

                    TableColumn("Model") { cell in
                        Text(cell.modelId).lineLimit(1).truncationMode(.middle)
                    }
                    .width(min: 140, ideal: 200)

                    TableColumn("Setting") { cell in
                        HStack(spacing: 4) {
                            Text(cell.armLabel)
                            if cell.isLossy { LossyBadge() }
                        }
                    }
                    .width(min: 90, ideal: 110)

                    TableColumn("Decode") { cell in
                        Text(formatted(cell.decodeTps, decimals: 1))
                            .monospacedDigit().fontWeight(.medium)
                    }
                    .width(min: 60, ideal: 74)

                    // n is never hidden: a cell built from one submission is a
                    // data point, not a benchmark, and the reader has to be
                    // able to tell the difference at a glance.
                    TableColumn("n") { cell in
                        Text("\(cell.sampleCount)")
                            .monospacedDigit()
                            .foregroundStyle(cell.sampleCount == 1 ? .orange : .secondary)
                    }
                    .width(min: 28, ideal: 36)
                }
                .tableStyle(.inset(alternatesRowBackgrounds: true))

                footerBar {
                    Text("Median per machine + model + setting. n = results behind each median.")
                    Spacer()
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private var visibleCommunityCells: [BenchmarkStore.Cell] {
        let rows = onlyMyChip
            ? community.filter { $0.hardware.chip == hardware.chip }
            : community
        return BenchmarkStore.aggregate(rows)
    }

    // MARK: - Chrome

    private func footerBar<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        HStack(spacing: 8) {
            content()
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
    }

    private func formatted(_ value: Double, decimals: Int) -> String {
        guard value > 0 else { return "—" }
        return String(format: "%.\(decimals)f", value)
    }

    // MARK: - Actions

    private func armBinding(_ arm: BenchmarkArm) -> Binding<Bool> {
        Binding(
            get: { selectedArmIds.contains(arm.id) },
            set: { on in
                if on { selectedArmIds.insert(arm.id) } else { selectedArmIds.remove(arm.id) }
            }
        )
    }

    private var phaseDescription: String {
        switch runner.phase {
        case .idle: return "Ready"
        case .warmup(let arm): return "Warming up — \(arm)"
        case .running(let arm, let run, let total): return "\(arm) — run \(run) of \(total)"
        case .done: return "Done"
        case .failed(let message): return message
        }
    }

    private func runBenchmark() async {
        guard let model = server.residentChatModel?.name else { return }
        isRunning = true
        runError = nil
        submitState = .idle
        ranAtLeastOnce = true
        lastResults = []
        defer { isRunning = false }

        let arms = BenchmarkArm.runnable.filter { selectedArmIds.contains($0.id) }
        do {
            let results = try await runner.run(
                suite: suite,
                arms: arms,
                modelId: model,
                port: server.port,
                engineVersion: Self.appVersion,
                hardware: hardware
            )
            lastResults = results
            history = BenchmarkStore.appendLocal(results)
        } catch {
            runError = error.localizedDescription
        }
    }

    private func submit() async {
        submitState = .sending
        do {
            try await client.submit(lastResults)
            submitState = .sent
        } catch {
            submitState = .failed(error.localizedDescription)
        }
    }

    private func loadCommunity() async {
        communityLoading = true
        communityError = nil
        defer { communityLoading = false }
        do {
            community = try await client.fetch()
        } catch {
            communityError = error.localizedDescription
        }
    }
}

// MARK: - Components

/// One titled card. Cards rather than `GroupBox` so the title can carry an
/// icon and the fill can stay subtle — stacked GroupBoxes read as a debug
/// panel, which is what this window looked like before.
private struct BenchCard<Content: View>: View {
    let title: String
    let icon: String
    var footnote: String? = nil
    @ViewBuilder var content: Content

    init(_ title: String, icon: String, footnote: String? = nil,
         @ViewBuilder content: () -> Content) {
        self.title = title
        self.icon = icon
        self.footnote = footnote
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: icon)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)

            content

            if let footnote {
                Text(footnote)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.28), in: RoundedRectangle(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 0.5)
        }
    }
}

/// A label/value row with an optional explainer under the label.
private struct BenchRow<Value: View>: View {
    let label: String
    var detail: String? = nil
    @ViewBuilder var value: Value

    init(_ label: String, detail: String? = nil, @ViewBuilder value: () -> Value) {
        self.label = label
        self.detail = detail
        self.value = value()
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                if let detail {
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 12)
            value
                .multilineTextAlignment(.trailing)
        }
    }
}

/// The number the user came for, sized like it.
private struct StatTile: View {
    let value: String
    let unit: String
    let label: String
    var emphasis: Bool = false

    var body: some View {
        VStack(spacing: 3) {
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value)
                    .font(.system(size: emphasis ? 30 : 23, weight: .semibold, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(emphasis ? AnyShapeStyle(.tint) : AnyShapeStyle(.primary))
                Text(unit)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Text(label.uppercased())
                .font(.system(size: 9, weight: .semibold))
                .tracking(0.6)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 9))
    }
}

/// `--kv-quant` and `--decode-attn-quant` trade output quality for speed, so a
/// speed-ranked board that didn't say so would be recommending worse answers.
private struct LossyBadge: View {
    var body: some View {
        Text("LOSSY")
            .font(.system(size: 9, weight: .bold))
            .tracking(0.4)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(.orange.opacity(0.18), in: Capsule())
            .foregroundStyle(.orange)
    }
}

private struct RatioBadge: View {
    let ratio: Double
    private var faster: Bool { ratio >= 1.0 }

    var body: some View {
        Text(String(format: "%.2f×", ratio))
            .font(.caption.weight(.semibold))
            .monospacedDigit()
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background((faster ? Color.green : Color.secondary).opacity(0.16), in: Capsule())
            .foregroundStyle(faster ? Color.green : Color.secondary)
            .help("Decode speed against this session's Defaults run")
    }
}
