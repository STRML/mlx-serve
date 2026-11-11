//! DFlash round policy for a sparse target, where a verify row is NOT free (GLM-5.3-Flash: every
//! extra row routes to more distinct experts). Pure data, no MLX: the generator feeds it each
//! round and reads the next decision back.
//!
//! A round drafts once, then verifies the first R rows (t1 and R - 1 drafts). R maximizes
//! emitted tokens per ms under (a) a per-position acceptance probability calibrated online
//! from the drafter's confidence and (b) the cost of an R-row verify. R = 1 is a plain round.
//! When drafts keep failing to pay for their own forward, drafting is skipped with a probe backoff.
const std = @import("std");

/// t1 plus up to 15 drafts.
pub const MAX_ROWS: u32 = 16;
pub const MAX_DRAFTS: u32 = MAX_ROWS - 1;

/// Round ms by verify rows; index = rows, 0 unused.
pub const Costs = [MAX_ROWS + 1]f32;

/// GLM-5.3-Flash verify ladder on an M5 Ultra at 1k context (ms by rows, linear between).
const LADDER = [_][2]f32{ .{ 1, 17.7 }, .{ 2, 25.5 }, .{ 3, 29.6 }, .{ 4, 33.6 }, .{ 5, 36.8 }, .{ 6, 42.4 }, .{ 7, 44.8 }, .{ 8, 47.9 }, .{ 9, 51.1 }, .{ 12, 60.5 }, .{ 16, 69.1 } };

/// The ladder scaled so one row costs `plain_ms`: a faster or slower machine moves it whole.
pub fn priorCosts(plain_ms: f32) Costs {
    var out: Costs = undefined;
    out[0] = 0;
    const scale = plain_ms / LADDER[0][1];
    for (1..MAX_ROWS + 1) |r| {
        const rows: f32 = @floatFromInt(r);
        var ms: f32 = LADDER[LADDER.len - 1][1];
        for (LADDER[0 .. LADDER.len - 1], LADDER[1..]) |lo, hi| {
            if (rows <= hi[0]) {
                ms = lo[1] + (hi[1] - lo[1]) * (rows - lo[0]) / (hi[0] - lo[0]);
                break;
            }
        }
        out[r] = ms * scale;
    }
    return out;
}

/// A plain round's ms when nothing has been measured yet.
pub const PRIOR_PLAIN_MS: f32 = LADDER[0][1];

pub const Choice = struct {
    /// Rows to verify, t1 included (1 = a plain round).
    rows: u32,
    /// Expected tokens the round emits.
    tokens: f32,
    /// Verify cost of `rows`.
    ms: f32,
};

/// A row count within this fraction of the best rate loses to the cheaper one: the model is not
/// that exact, and a verify's cost is certain where its tokens are not.
const ROW_TIE: f32 = 0.02;

/// The row count with the best expected tokens per ms. `p[k]` is the probability that draft k is
/// accepted given drafts 0..k-1 were; `costs` are the verify ms by rows.
pub fn chooseRows(p: []const f32, costs: *const Costs) Choice {
    const n: usize = @min(p.len, MAX_DRAFTS); // typed: a bare @min narrows to u4 and n + 2 overflows at 15
    var tokens: [MAX_ROWS + 1]f32 = undefined;
    tokens[1] = 1;
    var alive: f32 = 1;
    for (p[0..n], 0..) |pk, k| {
        alive *= if (pk >= 0) @min(pk, 1) else 0; // a NaN is no acceptance
        tokens[k + 2] = tokens[k + 1] + alive;
    }
    var best: f32 = 0;
    for (1..n + 2) |r| best = @max(best, tokens[r] / costs[r]);
    for (1..n + 2) |r| {
        if (tokens[r] / costs[r] >= best * (1 - ROW_TIE)) return .{ .rows = @intCast(r), .tokens = tokens[r], .ms = costs[r] };
    }
    return .{ .rows = 1, .tokens = 1, .ms = costs[1] };
}

/// Probability that a draft is accepted given its predecessors were, by the drafter's confidence in
/// it: online bins over `-log10(1 - conf)` (the selector's shares crowd 0.9..1, where a linear axis has one
/// bin), each pulled toward a logistic prior fitted on recorded rounds by a few pseudo-counts and forgetting
/// its old rounds, so a change of content class re-calibrates in a few rounds.
pub const Calibration = struct {
    const BINS = 12;
    const PER_UNIT: f32 = 4;
    /// Pseudo-observations the prior is worth.
    const PRIOR_WEIGHT: f32 = 3;
    /// A bin forgets this share of its history per observation it takes.
    const KEEP: f32 = 0.97;

    hit: [BINS]f32 = @splat(0),
    seen: [BINS]f32 = @splat(0),
    /// The drafter's confidence is its own acceptance estimate (DSpark's confidence head), the prior the bins start
    /// from; else the logistic fit.
    head: bool = false,

    /// 0 at a coin flip's confidence, 1 at 0.9, 2 at 0.99, capped at 3.
    fn axis(conf: f32) f32 {
        return -std.math.log10(@max(1 - std.math.clamp(conf, 0, 1), 1e-3));
    }

    fn bin(conf: f32) usize {
        return @min(BINS - 1, @as(usize, @intFromFloat(axis(conf) * PER_UNIT)));
    }

    fn prior(self: *const Calibration, conf: f32) f32 {
        if (self.head) return std.math.clamp(conf, 0, 1);
        return 1 / (1 + @exp(-(-1.15 + 1.58 * axis(conf))));
    }

    pub fn probability(self: *const Calibration, conf: f32) f32 {
        const b = bin(conf);
        return (self.hit[b] + PRIOR_WEIGHT * self.prior(conf)) / (self.seen[b] + PRIOR_WEIGHT);
    }

    pub fn observe(self: *Calibration, conf: f32, accepted: bool) void {
        const b = bin(conf);
        self.hit[b] = self.hit[b] * KEEP + @as(f32, if (accepted) 1 else 0);
        self.seen[b] = self.seen[b] * KEEP + 1;
    }
};

/// What a plain round costs, learned from the plain rounds themselves and nothing else: slowly, and by at
/// most a quarter per sample, so a stall or a foreign GPU job cannot pose as the machine's speed.
pub const PlainCost = struct {
    ms: f32 = PRIOR_PLAIN_MS,

    pub fn sample(self: *PlainCost, ms: f32) void {
        if (!std.math.isFinite(ms) or ms <= 0) return;
        const bounded = std.math.clamp(ms, self.ms * 0.8, self.ms * 1.25);
        self.ms += 0.1 * (bounded - self.ms);
    }
};

/// How often a copied continuation survives verify, by how far back it agreed with the context (under or
/// over `STRONG`): the share of positions reached that were kept, forgetting old rounds.
pub const CopyStats = struct {
    pub const STRONG: u32 = 16;
    const PRIOR = [2]f32{ 0.7, 0.9 };
    const PRIOR_WEIGHT: f32 = 3;
    const KEEP: f32 = 0.95;

    hit: [2]f32 = .{ PRIOR[0] * PRIOR_WEIGHT, PRIOR[1] * PRIOR_WEIGHT },
    tries: [2]f32 = .{ PRIOR_WEIGHT, PRIOR_WEIGHT },

    fn class(suffix: u32) usize {
        return @intFromBool(suffix >= STRONG);
    }

    pub fn probability(self: *const CopyStats, suffix: u32) f32 {
        const c = class(suffix);
        return self.hit[c] / self.tries[c];
    }

    /// A round that verified `drafts` copied tokens and kept `accepted` of them.
    pub fn observe(self: *CopyStats, suffix: u32, drafts: u32, accepted: u32) void {
        const c = class(suffix);
        self.hit[c] = self.hit[c] * KEEP + @as(f32, @floatFromInt(accepted));
        self.tries[c] = self.tries[c] * KEEP + @as(f32, @floatFromInt(@min(accepted + 1, drafts)));
    }
};

/// Whether this round drafts at all. Drafting costs a forward whether or not its rows are kept, so
/// a stretch where it does not pay is skipped with a probe backoff: each probe that loses doubles
/// the plain rounds before the next, a probe that wins clears the debt.
pub const Drafting = struct {
    /// Longest stretch of plain rounds between two probes.
    const MAX_PERIOD: u32 = 32;
    const ALPHA: f32 = 0.3;

    /// Averages over drafted rounds of the plain-round ms their tokens replaced and of the ms they took:
    /// the ratio weighs a round by its time, where a mean of per-round ratios swings on every short miss.
    gain: f32 = 0,
    wall: f32 = 0,
    /// Plain rounds still owed before the next draft.
    owed: u32 = 0,
    /// Plain rounds the next losing round will owe.
    period: u32 = 1,
    /// Rounds skipped, for stats.
    skipped: u64 = 0,
    /// Drafted rounds that followed a skip stretch.
    probes: u64 = 0,
    after_skip: bool = false,

    pub fn shouldDraft(self: *Drafting) bool {
        if (self.owed > 0) {
            self.owed -= 1;
            self.skipped += 1;
            self.after_skip = true;
            return false;
        }
        if (self.after_skip) self.probes += 1;
        self.after_skip = false;
        return true;
    }

    /// A drafted round: `gain` = the ms of plain rounds its tokens would have cost, `wall` = what it took.
    pub fn record(self: *Drafting, gain: f32, wall: f32) void {
        if (self.wall <= 0) {
            self.gain = gain;
            self.wall = wall;
        } else {
            self.gain += ALPHA * (gain - self.gain);
            self.wall += ALPHA * (wall - self.wall);
        }
        if (self.gain >= self.wall) {
            self.owed = 0;
            self.period = @max(1, self.period / 2);
        } else {
            self.owed = self.period;
            self.period = @min(self.period * 2, MAX_PERIOD);
        }
    }
};

/// Rows one batched verify carries at most: the module batches its compiled stretches up to this many.
pub const MAX_GROUP_ROWS: u32 = 16;

pub const GroupCosts = [MAX_GROUP_ROWS + 1]f32;

/// Verify rows for slots sharing one batched verify of at most `max_rows` rows. `p[k]` is slot k's per-draft acceptance
/// as `chooseRows` reads it, `costs[r]` the verify's ms at `r` rows in all, the slots' `t1` rows included. The most tokens
/// any `r` rows buy are the `r` likeliest drafts (a slot's in order), so the choice is the `r` with the best expected
/// tokens per ms, the fewer rows on a near tie. Writes slot k's drafts to verify to `drafts[k]`; returns the expected tokens.
pub fn chooseGroupRows(p: []const []const f32, costs: *const GroupCosts, max_rows: usize, drafts: []u32) f32 {
    const n = p.len;
    std.debug.assert(n >= 1 and n <= max_rows and max_rows <= MAX_GROUP_ROWS);
    var alive: [MAX_GROUP_ROWS]f32 = @splat(1);
    var next: [MAX_GROUP_ROWS]u32 = @splat(0);
    var picks: [MAX_GROUP_ROWS]u8 = undefined;
    var tokens: [MAX_GROUP_ROWS + 1]f32 = undefined;
    tokens[n] = @floatFromInt(n);
    var r = n;
    while (r < max_rows) {
        var pick: ?usize = null;
        var gain: f32 = 0;
        for (p, 0..) |pk, k| {
            if (next[k] >= pk.len) continue;
            const g = alive[k] * (if (pk[next[k]] >= 0) @min(pk[next[k]], 1) else 0); // a NaN is no acceptance
            if (g > gain) {
                pick = k;
                gain = g;
            }
        }
        const k = pick orelse break;
        alive[k] = gain;
        next[k] += 1;
        picks[r - n] = @intCast(k);
        r += 1;
        tokens[r] = tokens[r - 1] + gain;
    }
    var best: f32 = 0;
    for (n..r + 1) |rr| best = @max(best, tokens[rr] / costs[rr]);
    var chosen = n;
    for (n..r + 1) |rr| if (tokens[rr] / costs[rr] >= best * (1 - ROW_TIE)) {
        chosen = rr;
        break;
    };
    @memset(drafts[0..n], 0);
    for (picks[0 .. chosen - n]) |k| drafts[k] += 1;
    return tokens[chosen];
}

/// How a round's slots split into verifies of at most `max_rows` rows: consecutive runs of the returned width (the last
/// takes the rest), each priced by `chooseGroupRows`, the width whose verifies buy the most expected tokens per ms. A
/// narrower split must beat a wider one by `ROW_TIE`: its extra verify's base is certain where its tokens are not.
/// Writes slot k's drafts to verify to `drafts[k]`.
pub fn planGroup(p: []const []const f32, cost: *const GroupCost, max_rows: usize, drafts: []u32) usize {
    const n = p.len;
    if (n == 0) return 0;
    // A verify carries at least its slots' own rows: a budget below one still verifies one slot at a time.
    const budget = std.math.clamp(max_rows, 1, MAX_GROUP_ROWS);
    var best_w: usize = 0;
    var best_rate: f32 = 0;
    var w = @min(n, budget);
    while (w >= 1) : (w -= 1) {
        const rate = splitRate(p, cost, budget, w, drafts);
        if (best_w == 0 or rate > best_rate * (1 + ROW_TIE)) {
            best_w = w;
            best_rate = rate;
        }
    }
    _ = splitRate(p, cost, budget, best_w, drafts);
    return best_w;
}

fn splitRate(p: []const []const f32, cost: *const GroupCost, max_rows: usize, width: usize, drafts: []u32) f32 {
    var tokens: f32 = 0;
    var ms: f32 = 0;
    var lo: usize = 0;
    while (lo < p.len) : (lo += width) {
        const hi = @min(p.len, lo + width);
        const c = cost.costs(hi - lo);
        tokens += chooseGroupRows(p[lo..hi], &c, max_rows, drafts[lo..hi]);
        var rows = hi - lo;
        for (drafts[lo..hi]) |d| rows += d;
        ms += c[rows];
    }
    return tokens / ms;
}

/// What a DSpark group round's verify costs: a base by its state count (one row a slot, the group's plain step) and one
/// slope for every row past them, fitted online over this machine's rounds by least squares that forget old rounds.
/// Both are pulled toward the M5 Ultra fit (`DSV41_GROUP`: `BASE + PER_STATE * states`, `PER_ROW`) as scaled by this
/// machine's rounds against it, so rounds that never vary their rows fit the prior scaled, and a base never drops below
/// `BASE_FLOOR` of its scaled prior: a fit at or below zero would buy no draft and then drop every plain round as a stall.
/// A count's first rounds only drop a stall against the prior; after them a round counts as at most a quarter off the fit.
pub const GroupCost = struct {
    const BASE: f32 = 12;
    const PER_STATE: f32 = 4.6;
    const PER_ROW: f32 = 4.4;
    /// Pseudo-rounds the prior base is worth, and the prior slope's weight in squared rows.
    const PRIOR_ROUNDS: f32 = 2;
    const PRIOR_SPREAD: f32 = 16;
    const KEEP: f32 = 0.95;
    const WARM_ROUNDS: f32 = 4;
    const MIN_ROW_MS: f32 = 0.05;
    const BASE_FLOOR: f32 = 0.25;
    const MIN_SCALE: f32 = 0.05;
    const MAX_SCALE: f32 = 20;
    const N = MAX_GROUP_ROWS + 1;

    /// By state count: the rounds' weight and their sums of extra rows `x`, ms `y`, `x * x` and `x * y`.
    w: [N]f32 = @splat(0),
    sx: [N]f32 = @splat(0),
    sy: [N]f32 = @splat(0),
    sxx: [N]f32 = @splat(0),
    sxy: [N]f32 = @splat(0),

    fn priorBase(states: usize) f32 {
        return BASE + PER_STATE * @as(f32, @floatFromInt(states));
    }

    fn prior(states: usize, rows: usize) f32 {
        return priorBase(states) + PER_ROW * @as(f32, @floatFromInt(rows -| states));
    }

    /// This machine's rounds' ms over what the prior prices them at (1 before any).
    fn scale(self: *const GroupCost) f32 {
        var took: f32 = 0;
        var priced: f32 = 0;
        for (self.w, self.sx, self.sy, 0..) |w, sx, sy, k| {
            took += sy;
            priced += w * priorBase(k) + PER_ROW * sx;
        }
        if (!(priced > 0) or !(took > 0)) return 1;
        return std.math.clamp(took / priced, MIN_SCALE, MAX_SCALE);
    }

    fn slopeAt(self: *const GroupCost, sc: f32) f32 {
        var cxy: f32 = PRIOR_SPREAD * PER_ROW * sc;
        var cxx: f32 = PRIOR_SPREAD;
        for (self.w, self.sx, self.sy, self.sxx, self.sxy) |w, sx, sy, sxx, sxy| {
            if (w <= 0) continue;
            cxy += sxy - sx * sy / w;
            cxx += @max(0, sxx - sx * sx / w);
        }
        const v = cxy / cxx;
        return if (v >= MIN_ROW_MS) v else MIN_ROW_MS; // a NaN is the floor too
    }

    /// The verify of `states` slots' rows alone, at `slope` ms a further row.
    fn baseAt(self: *const GroupCost, states: usize, sc: f32, slope: f32) f32 {
        const s = @min(states, MAX_GROUP_ROWS);
        const pb = sc * priorBase(states);
        const b = (self.sy[s] - slope * self.sx[s] + PRIOR_ROUNDS * pb) / (self.w[s] + PRIOR_ROUNDS);
        return if (b >= BASE_FLOOR * pb) b else BASE_FLOOR * pb;
    }

    /// What one more verify row costs.
    pub fn rowMs(self: *const GroupCost) f32 {
        return self.slopeAt(self.scale());
    }

    pub fn ms(self: *const GroupCost, states: usize, rows: usize) f32 {
        const sc = self.scale();
        const slope = self.slopeAt(sc);
        return self.baseAt(states, sc, slope) + slope * @as(f32, @floatFromInt(rows -| states));
    }

    /// The verify's ms by rows for a group of `states` slots (index = rows, below `states` unused).
    pub fn costs(self: *const GroupCost, states: usize) GroupCosts {
        const sc = self.scale();
        const slope = self.slopeAt(sc);
        const b = self.baseAt(states, sc, slope);
        var out: GroupCosts = @splat(0);
        for (states..MAX_GROUP_ROWS + 1) |r| out[r] = b + slope * @as(f32, @floatFromInt(r - states));
        return out;
    }

    /// A verify of `rows` rows over `states` slots took `took` ms.
    pub fn sample(self: *GroupCost, states: usize, rows: usize, took: f32) void {
        if (!std.math.isFinite(took) or took <= 0 or states == 0 or states > MAX_GROUP_ROWS) return;
        const y = if (self.w[states] >= WARM_ROUNDS) blk: {
            const fit = self.ms(states, rows);
            break :blk std.math.clamp(took, fit * 0.8, fit * 1.25);
        } else blk: {
            // A stall against the prior: up to 8x slower or 4x faster than the M5 Ultra counts.
            const p = prior(states, rows);
            if (took > p * 8 or took < p / 4) return;
            break :blk took;
        };
        const x: f32 = @floatFromInt(rows -| states);
        self.w[states] = self.w[states] * KEEP + 1;
        self.sx[states] = self.sx[states] * KEEP + x;
        self.sy[states] = self.sy[states] * KEEP + y;
        self.sxx[states] = self.sxx[states] * KEEP + x * x;
        self.sxy[states] = self.sxy[states] * KEEP + x * y;
    }
};

/// Per-draft-position acceptance and the verify widths a request ran, for `[spec-stats]`.
pub const PositionStats = struct {
    reached: [MAX_DRAFTS]u64 = @splat(0),
    accepted: [MAX_DRAFTS]u64 = @splat(0),
    /// Rounds by verify rows; index = rows.
    rows: [MAX_ROWS + 1]u64 = @splat(0),

    /// A round that verified `drafts` drafts and kept `accepted` of them: it reached every position
    /// up to the first rejection and no further.
    pub fn record(self: *PositionStats, drafts: u32, accepted: u32) void {
        const m: usize = @min(drafts, MAX_DRAFTS);
        self.rows[m + 1] += 1;
        for (0..m) |k| {
            if (k > accepted) break;
            self.reached[k] += 1;
            if (k < accepted) self.accepted[k] += 1;
        }
    }

    /// `1.00/0.95/...`: conditional acceptance by draft position, down to the deepest one any round reached.
    pub fn formatAcc(self: *const PositionStats, buf: []u8) []const u8 {
        var w = std.Io.Writer.fixed(buf);
        for (self.reached, self.accepted, 0..) |reached, kept, k| {
            if (reached == 0) break;
            if (k > 0) w.writeAll("/") catch break;
            w.print("{d:.2}", .{@as(f32, @floatFromInt(kept)) / @as(f32, @floatFromInt(reached))}) catch break;
        }
        return w.buffered();
    }

    /// `1:12,2:3,8:100`: rounds by verify rows.
    pub fn formatRows(self: *const PositionStats, buf: []u8) []const u8 {
        var w = std.Io.Writer.fixed(buf);
        var first = true;
        for (self.rows, 0..) |n, rows| {
            if (n == 0) continue;
            if (!first) w.writeAll(",") catch break;
            first = false;
            w.print("{d}:{d}", .{ rows, n }) catch break;
        }
        return w.buffered();
    }
};

const testing = std.testing;

fn approx(a: f32, b: f32) !void {
    try testing.expectApproxEqAbs(b, a, 0.02);
}

test "priorCosts: scaled to the measured plain step, never cheaper for more rows" {
    const c = priorCosts(17.7);
    try testing.expectApproxEqAbs(@as(f32, 17.7), c[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 25.5), c[2], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 47.9), c[8], 1e-3);
    var r: u32 = 2;
    while (r <= MAX_ROWS) : (r += 1) try testing.expect(c[r] >= c[r - 1]);
    // A faster machine's plain step moves the whole ladder with it.
    const half = priorCosts(8.85);
    try testing.expectApproxEqAbs(c[8] / 2, half[8], 1e-3);
}

test "chooseRows: a plain round when nothing pays, the whole chain when everything does" {
    const costs = priorCosts(17.7);
    const weak = [_]f32{ 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1 };
    try testing.expectEqual(@as(u32, 1), chooseRows(&weak, &costs).rows);
    const sure: [7]f32 = @splat(0.96);
    const c = chooseRows(&sure, &costs);
    try testing.expectEqual(@as(u32, 8), c.rows);
    try approx(c.tokens, 6.96);
    try approx(c.ms, 47.9);
}

test "chooseRows: a fifteen-draft chain is priced through its last row, and a poisoned input still answers" {
    const costs = priorCosts(17.7);
    const sure: [15]f32 = @splat(0.98);
    const c = chooseRows(&sure, &costs);
    try testing.expect(c.rows >= 8 and c.rows <= 16);
    // A NaN probability must not leave the function without a choice.
    const bad: [3]f32 = .{ std.math.nan(f32), 0.9, 0.9 };
    try testing.expect(chooseRows(&bad, &costs).rows >= 1);
}

test "chooseRows: the second row pays for its own +8 ms jump only past 48% acceptance" {
    const costs = priorCosts(17.7);
    try testing.expectEqual(@as(u32, 1), chooseRows(&[_]f32{0.40}, &costs).rows);
    try testing.expectEqual(@as(u32, 2), chooseRows(&[_]f32{0.60}, &costs).rows);
}

test "chooseRows: rows past a poor draft are not bought for the confident draft behind it" {
    const costs = priorCosts(17.7);
    // Draft 1 almost never lands, so draft 2 is only reached 5% of the time.
    const c = chooseRows(&[_]f32{ 0.95, 0.05, 0.99, 0.99 }, &costs);
    try testing.expectEqual(@as(u32, 2), c.rows);
}

test "chooseRows: within 2% of the best rate the cheaper verify wins" {
    var costs: Costs = @splat(20.0);
    costs[1] = 10.0;
    costs[2] = 12.0;
    costs[3] = 12.2;
    // The third row buys 1.5% over two rows: not worth verifying.
    try testing.expectEqual(@as(u32, 2), chooseRows(&[_]f32{ 0.5, 0.0958 }, &costs).rows);
    // At 15% it is.
    try testing.expectEqual(@as(u32, 3), chooseRows(&[_]f32{ 0.5, 0.5 }, &costs).rows);
}

test "Calibration: no data answers the prior, then the data takes over, bin by bin" {
    var c = Calibration{};
    try approx(c.probability(0.9), 0.606);
    try approx(c.probability(0.2), 0.269);
    for (0..40) |_| c.observe(0.9, false);
    try testing.expect(c.probability(0.9) < 0.3);
    try approx(c.probability(0.2), 0.269);
    // The bin is a confidence range: 0.88 shares 0.9's evidence.
    try testing.expect(c.probability(0.88) < 0.3);
}

test "Calibration: an old regime fades" {
    var c = Calibration{};
    for (0..80) |_| c.observe(0.7, true);
    try testing.expect(c.probability(0.7) > 0.9);
    for (0..80) |_| c.observe(0.7, false);
    try testing.expect(c.probability(0.7) < 0.2);
}

// A drafted round is judged by `record(gain, wall)`: the plain-round ms its tokens would have cost, against
// the ms it took. Ten against ten is break-even.

test "PlainCost: a stall moves it little, a steady new speed is reached" {
    var c = PlainCost{};
    c.sample(400); // a compile stall, a foreign GPU job
    try testing.expect(c.ms > PRIOR_PLAIN_MS and c.ms < 18.5);
    var faster = PlainCost{};
    for (0..80) |_| faster.sample(14.0);
    try testing.expect(faster.ms < 14.5 and faster.ms >= 14.0);
    var slower = PlainCost{};
    for (0..80) |_| slower.sample(22.0);
    try testing.expect(slower.ms > 21.0 and slower.ms <= 22.0);
}

test "CopyStats: a strong match starts trusted, and misses teach it otherwise" {
    var c = CopyStats{};
    try testing.expect(c.probability(20) > c.probability(8));
    for (0..12) |_| c.observe(20, 14, 0);
    try testing.expect(c.probability(20) < 0.3);
    try approx(c.probability(8), 0.7); // the other class is untouched
    var d = CopyStats{};
    for (0..12) |_| d.observe(20, 14, 14);
    try testing.expect(d.probability(20) > 0.9);
}

test "Drafting: a losing draft owes plain rounds, doubling to a cap; a winning probe resets" {
    var d = Drafting{};
    try testing.expect(d.shouldDraft());
    d.record(7, 10); // loses: one plain round owed
    try testing.expect(!d.shouldDraft());
    try testing.expect(d.shouldDraft());
    d.record(7, 10); // loses again: two owed
    try testing.expect(!d.shouldDraft());
    try testing.expect(!d.shouldDraft());
    try testing.expect(d.shouldDraft());
    var guard: u32 = 0;
    // However long it loses, the wait stops growing at 32 plain rounds.
    while (guard < 12) : (guard += 1) {
        d.record(5, 10);
        var owed: u32 = 0;
        while (!d.shouldDraft()) owed += 1;
        try testing.expect(owed <= 32);
    }
    d.record(5, 10);
    var owed: u32 = 0;
    while (!d.shouldDraft()) owed += 1;
    try testing.expectEqual(@as(u32, 32), owed);
    // A probe that pays clears the debt.
    d.record(60, 10);
    try testing.expect(d.shouldDraft());
    try testing.expect(d.probes > 0 and d.skipped > 0);
}

test "Drafting: a thin win with scattered misses keeps drafting, because the rounds are weighed by their time" {
    var d = Drafting{};
    // 1.6x on average, a quarter of the rounds below break-even: the mean of the ratios would dip under 1.
    const rounds = [_][2]f32{ .{ 70, 40 }, .{ 18, 40 }, .{ 70, 40 }, .{ 70, 40 }, .{ 18, 40 }, .{ 120, 45 }, .{ 18, 40 }, .{ 120, 45 }, .{ 70, 40 }, .{ 18, 40 }, .{ 70, 40 }, .{ 120, 45 } };
    for (rounds) |r| {
        try testing.expect(d.shouldDraft());
        d.record(r[0], r[1]);
    }
}

test "Drafting: one failed round among winners does not stop drafting" {
    var d = Drafting{};
    for (0..6) |_| {
        try testing.expect(d.shouldDraft());
        d.record(25, 10);
    }
    try testing.expect(d.shouldDraft());
    d.record(6, 10);
    try testing.expect(d.shouldDraft());
}

test "PositionStats: conditional acceptance counts only the positions a round reached" {
    var st = PositionStats{};
    st.record(7, 7); // every position reached and kept
    st.record(7, 2); // 0 and 1 kept, 2 rejected, 3.. never reached
    st.record(0, 0); // a plain round
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("1.00/1.00/0.50/1.00/1.00/1.00/1.00", st.formatAcc(&buf));
    try testing.expectEqualStrings("1:1,8:2", st.formatRows(&buf));
}

test "PositionStats: the widest copied chain records without overflow" {
    var st = PositionStats{};
    st.record(15, 15);
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("16:1", st.formatRows(&buf));
}

test "PositionStats: a position no round reached is not listed" {
    var st = PositionStats{};
    st.record(3, 0);
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("0.00", st.formatAcc(&buf));
    st.record(5, 5);
    try testing.expectEqualStrings("0.50/1.00/1.00/1.00/1.00", st.formatAcc(&buf));
}

test "chooseGroupRows: rows go to the likeliest drafts across slots, none when the group's rows do not pay" {
    var cost: GroupCost = .{};
    const c2 = cost.costs(2);
    var drafts: [2]u32 = undefined;
    // A sure copy and an unsure stage draft: the sure slot takes the rows.
    const sure: [8]f32 = @splat(0.97);
    const weak = [_]f32{ 0.2, 0.2, 0.2, 0.2, 0.2 };
    _ = chooseGroupRows(&.{ &sure, &weak }, &c2, MAX_GROUP_ROWS, &drafts);
    try testing.expect(drafts[0] >= 6);
    try testing.expectEqual(@as(u32, 0), drafts[1]);
    // Neither pays: the group's plain step.
    const tokens = chooseGroupRows(&.{ &weak, &weak }, &c2, MAX_GROUP_ROWS, &drafts);
    try testing.expectEqual([2]u32{ 0, 0 }, drafts);
    try approx(tokens, 2);
    // Sure slots share the row budget, the total never past it.
    const c4 = cost.costs(4);
    var d4: [4]u32 = undefined;
    for ([_]usize{ MAX_GROUP_ROWS, 8 }) |budget| {
        _ = chooseGroupRows(&.{ &sure, &sure, &sure, &sure }, &c4, budget, &d4);
        var total: u32 = 4;
        for (d4) |d| total += d;
        try testing.expect(total <= budget);
        try testing.expect(@max(d4[0], d4[3]) - @min(d4[0], d4[3]) <= 1);
    }
}

test "chooseGroupRows: a poisoned probability is no acceptance, and the cut stays a prefix of each slot's drafts" {
    var cost: GroupCost = .{};
    const c = cost.costs(1);
    var drafts: [1]u32 = undefined;
    _ = chooseGroupRows(&.{&[_]f32{ 0.99, std.math.nan(f32), 0.99, 0.99 }}, &c, MAX_GROUP_ROWS, &drafts);
    try testing.expect(drafts[0] <= 1);
}

/// A verify's rounds on a machine of base `b0 + b1 * states` and `slope` ms a further row, at every width up to 16 rows.
fn trained(b0: f32, b1: f32, slope: f32) GroupCost {
    var cost: GroupCost = .{};
    for (0..200) |i| for ([_]usize{ 1, 2, 3, 4 }) |st| {
        const rows = st + (i * 5) % (MAX_GROUP_ROWS + 1 - st);
        cost.sample(st, rows, b0 + b1 * @as(f32, @floatFromInt(st)) + slope * @as(f32, @floatFromInt(rows - st)));
    };
    return cost;
}

test "GroupCost: a machine's own base and row slope are learned apart, and its row choice follows them" {
    var cost = trained(20, 3, 2.45);
    try testing.expectApproxEqAbs(@as(f32, 4 * 2.45), cost.ms(2, 6) - cost.ms(2, 2), 0.4);
    try testing.expectApproxEqAbs(@as(f32, 26), cost.ms(2, 2), 1.5);
    try testing.expectApproxEqAbs(@as(f32, 32), cost.ms(4, 4), 1.5);
    // Medium drafts: the prior's 4.4 ms rows buy none of them, this machine's 2.45 ms rows do.
    const medium: [4]f32 = @splat(0.35);
    var d: [2]u32 = undefined;
    const fresh: GroupCost = .{};
    _ = chooseGroupRows(&.{ &medium, &medium }, &fresh.costs(2), MAX_GROUP_ROWS, &d);
    try testing.expectEqual([2]u32{ 0, 0 }, d);
    _ = chooseGroupRows(&.{ &medium, &medium }, &cost.costs(2), MAX_GROUP_ROWS, &d);
    try testing.expect(d[0] + d[1] > 0);
    // A stall and a NaN move it little.
    const before = cost.ms(2, 6);
    cost.sample(2, 6, 1e9);
    cost.sample(2, 6, std.math.nan(f32));
    try testing.expect(cost.ms(2, 6) < before * 1.05);
}

test "GroupCost: rounds that never vary their rows fit the prior scaled, every cost above zero" {
    // One state count, every verify at the budget: no row slope to learn, so the prior's shape scales to the rounds.
    var cost: GroupCost = .{};
    for (0..200) |_| cost.sample(2, 16, 20 + 3 * 2 + 2.45 * 14);
    try testing.expectApproxEqAbs(@as(f32, 60.3), cost.ms(2, 16), 1);
    const c = cost.costs(2);
    for (c[2..]) |x| try testing.expect(x > 0);
    try testing.expect(cost.ms(2, 2) >= 0.25 * 60.3 / (12 + 4.6 * 2 + 4.4 * 14) * (12 + 4.6 * 2) * 0.99);
}

test "GroupCost: one fast wide round never locks the plain step's price near zero" {
    // A 36 ms plain step and 1 ms rows: one wide round first, then plain rounds only.
    var cost: GroupCost = .{};
    cost.sample(2, 16, 50);
    try testing.expect(cost.ms(2, 2) > 0);
    for (0..200) |_| cost.sample(2, 2, 36);
    try testing.expectApproxEqAbs(@as(f32, 36), cost.ms(2, 2), 1.5);
}

test "GroupCost: a steeper slope learned at one count never prices another's plain step at or below zero" {
    var cost: GroupCost = .{};
    for (0..6) |_| cost.sample(2, 14, 74);
    for (0..20) |_| {
        cost.sample(3, 3, 33);
        cost.sample(3, 15, 153);
    }
    try testing.expect(cost.ms(2, 2) > 0);
    cost.sample(2, 2, 21.2); // its clamp's bounds stay ordered
    try testing.expect(cost.ms(2, 2) > 0);
}

test "GroupCost: a cold start at full width on a faster machine still learns its plain step" {
    // 2-rank tensor parallel: 7 + 2.6 ms a state + 2.4 ms a row; the first round verifies 14 rows.
    var cost: GroupCost = .{};
    cost.sample(2, 14, 41);
    const before = cost.w[2];
    cost.sample(2, 2, 12.2);
    try testing.expect(cost.w[2] > before);
    for (0..100) |_| {
        cost.sample(2, 2, 12.2);
        cost.sample(2, 14, 41);
    }
    try testing.expectApproxEqAbs(@as(f32, 12.2), cost.ms(2, 2), 0.5);
    try testing.expectApproxEqAbs(@as(f32, 41), cost.ms(2, 14), 1);
}

test "planGroup: a row budget of 0 or past the group's still plans" {
    const cost: GroupCost = .{};
    const sure: [7]f32 = @splat(0.99);
    var d: [3]u32 = undefined;
    try testing.expectEqual(@as(usize, 1), planGroup(&.{ &sure, &sure, &sure }, &cost, 0, &d));
    try testing.expectEqual([3]u32{ 0, 0, 0 }, d);
    try testing.expect(planGroup(&.{ &sure, &sure, &sure }, &cost, 1000, &d) >= 1);
    try testing.expectEqual(@as(usize, 0), planGroup(&.{}, &cost, 8, d[0..0]));
}

test "planGroup: slots past one verify's rows split into narrower verifies only where the extra base pays" {
    const cost = trained(2, 6, 1);
    const sure: [7]f32 = @splat(0.99);
    const weak: [7]f32 = @splat(0.3);
    var d: [4]u32 = undefined;
    // Sure drafts: two verifies of two slots, each slot's whole block.
    try testing.expectEqual(@as(usize, 2), planGroup(&.{ &sure, &sure, &sure, &sure }, &cost, MAX_GROUP_ROWS, &d));
    try testing.expectEqual([4]u32{ 7, 7, 7, 7 }, d);
    // Weak drafts: one verify, a second base buys nothing.
    try testing.expectEqual(@as(usize, 4), planGroup(&.{ &weak, &weak, &weak, &weak }, &cost, MAX_GROUP_ROWS, &d));
    // A bank whose decode call takes 8 rows: no verify past them.
    const w = planGroup(&.{ &sure, &sure, &sure, &sure }, &cost, 8, &d);
    var lo: usize = 0;
    while (lo < d.len) : (lo += w) {
        var rows = @min(d.len, lo + w) - lo;
        for (d[lo..@min(d.len, lo + w)]) |x| rows += x;
        try testing.expect(rows <= 8);
    }
}

test "Calibration: a head prior reads the drafter's confidence as its acceptance until verifies say otherwise" {
    var head: Calibration = .{ .head = true };
    const fitted: Calibration = .{};
    try testing.expectApproxEqAbs(@as(f32, 0.93), head.probability(0.93), 1e-4);
    try testing.expect(fitted.probability(0.93) < 0.7);
    for (0..60) |_| head.observe(0.93, false);
    try testing.expect(head.probability(0.93) < 0.2);
}
