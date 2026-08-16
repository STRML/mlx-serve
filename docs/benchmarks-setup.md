# Community benchmarks — backend setup

Phase 1 of the community benchmark board. One Firebase Realtime Database, read
and written over plain HTTPS. No SDK on either client, no Cloud Functions, no
auth.

## Why RTDB and not Firestore

The tier list uses Firestore through the JS SDK. The benchmark board
deliberately does not:

- RTDB's REST shape is ordinary JSON. Firestore's REST wraps every value in a
  type tag (`{"stringValue": "..."}`), which would need a translation layer on
  both the Swift and the JS side.
- The website is a static GitHub Pages file. With RTDB it needs one `fetch()`
  and zero client libraries.
- Submissions are append-only rows, not documents anyone edits. RTDB's push
  keys are ordered by time, so "newest N" is `orderBy="$key"&limitToLast=N`.

## Create the database

1. Firebase console → the existing **mlxserve** project → Build → Realtime
   Database → Create Database.
2. Pick the **United States (us-central1)** location, and start in **locked
   mode** (the rules below replace whatever it starts with).
3. Confirm the URL is `https://mlxserve-default-rtdb.firebaseio.com`. If the
   console hands back a different one, update `BenchmarkStore.communityBaseURL`
   (Swift) and `DB_URL` (`website/benchmarks/index.html`) to match.

## Security rules

Paste into Realtime Database → Rules.

Phase 1 has no identity, so these rules are the only server-side gate. They do
three things: keep the collection append-only, bound what a single row may
contain, and reject anything that isn't shaped like a result. They do **not**
attempt to decide whether a number is true — that's phase 2's job (App Attest
plus the roofline check).

```json
{
  "rules": {
    ".read": false,
    ".write": false,

    "results": {
      // Public board.
      ".read": true,

      // Push keys only — a client may add a row, never replace the collection.
      "$row": {
        // Append-only: a row that already exists can never be edited or
        // deleted by a client. Without this, one request could blank the
        // board or rewrite someone else's number.
        ".write": "!data.exists() && newData.exists()",

        ".validate": "newData.hasChildren(['schemaVersion','sessionId','suiteId','armId','modelId','prefillTps','decodeTps','runs','hardware','date'])",

        "schemaVersion": { ".validate": "newData.isNumber() && newData.val() >= 1" },
        "id":            { ".validate": "newData.isString() && newData.val().length <= 64" },
        "sessionId":     { ".validate": "newData.isString() && newData.val().length <= 64" },
        "suiteId":       { ".validate": "newData.isString() && newData.val().length <= 64" },
        "armId":         { ".validate": "newData.isString() && newData.val().length <= 64" },
        "armLabel":      { ".validate": "newData.isString() && newData.val().length <= 64" },
        "modelId":       { ".validate": "newData.isString() && newData.val().length <= 200" },
        "quant":         { ".validate": "newData.isString() && newData.val().length <= 40" },
        "engineVersion": { ".validate": "newData.isString() && newData.val().length <= 40" },
        "isLossy":       { ".validate": "newData.isBoolean()" },
        "date":          { ".validate": "newData.isString() && newData.val().length <= 40" },

        // Loose physical bounds. Not a correctness check — just enough that a
        // row can't carry a number no Mac could ever produce, or a negative
        // one that would break the medians.
        "prefillTps":       { ".validate": "newData.isNumber() && newData.val() >= 0 && newData.val() <= 200000" },
        "decodeTps":        { ".validate": "newData.isNumber() && newData.val() >= 0 && newData.val() <= 10000" },
        "ttftMs":           { ".validate": "newData.isNumber() && newData.val() >= 0 && newData.val() <= 3600000" },
        "promptTokens":     { ".validate": "newData.isNumber() && newData.val() >= 0 && newData.val() <= 10000000" },
        "completionTokens": { ".validate": "newData.isNumber() && newData.val() >= 0 && newData.val() <= 10000000" },
        "runs":             { ".validate": "newData.isNumber() && newData.val() >= 0 && newData.val() <= 1000" },
        "spreadPercent":    { ".validate": "newData.isNumber() && newData.val() >= 0 && newData.val() <= 100000" },

        "flags": {
          "$flag": { ".validate": "newData.isString() && newData.val().length <= 40" }
        },

        "hardware": {
          ".validate": "newData.hasChildren(['chip','ramGB'])",
          "chip":      { ".validate": "newData.isString() && newData.val().length <= 60" },
          "gpuCores":  { ".validate": "newData.isNumber() && newData.val() >= 0 && newData.val() <= 1024" },
          "ramGB":     { ".validate": "newData.isNumber() && newData.val() > 0 && newData.val() <= 8192" },
          "osVersion": { ".validate": "newData.isString() && newData.val().length <= 40" },
          "onBattery": { ".validate": "newData.isBoolean()" },
          "$other":    { ".validate": false }
        },

        // Anything not named above is rejected, so a client can't pad rows
        // with arbitrary payload and use the board as free storage.
        "$other": { ".validate": false }
      }
    }
  }
}
```

Note the `$other: false` terminators. Without them RTDB accepts any extra child,
and an open database with no field allowlist is a free object store.

That does mean phase 2's attestation fields have to be added to this rule set at
the same time they're added to the payload — otherwise every attested row is
rejected. The readers already tolerate unknown fields (both are unit-tested for
it), so the client side of that upgrade is backwards compatible.

## Indexing

**Do not add `".indexOn": "$key"`.** It looks reasonable and is a hard syntax
error — the deploy fails with:

```
Invalid key: $key. Index must be either .value or declared on a valid path
```

`orderBy="$key"` needs no index; RTDB orders by key natively. `.indexOn` only
becomes necessary if a query moves to a child field (e.g. `orderBy="date"`), and
then it names that field, never `$key`.

## Deploying

The repo carries `firebase.json`, `.firebaserc` and `database.rules.json`, so:

```bash
firebase deploy --only database
```

## Verified against the live instance

| Request | Result |
|---|---|
| `GET /results.json?orderBy="$key"&limitToLast=5` | 200 (`null` when empty) |
| `POST` a well-formed row | 200 |
| `POST {"nonsense":true}` | 401 |
| `POST` a valid row plus one extra field | 401 |
| `POST` a row with `decodeTps: 999999` | 401 |
| `PUT` over an existing row | 401 |
| `DELETE` an existing row | 401 |
| `DELETE /results.json` (wipe the collection) | 401 |

Admin removal bypasses the rules and is the moderation tool for a bad row:

```bash
firebase database:remove /results/<pushKey> --force
```

## What phase 1 does NOT do

Stated plainly because the board is public and its numbers should be read with
this in mind:

- **No identity.** Anyone can submit. There is no per-person or per-device
  deduplication, so one enthusiastic person can move a median.
- **No verification.** Numbers are taken on trust. The client measures honestly,
  but nothing stops a hand-crafted POST.
- **No rate limit.** Rules cannot express one without an authenticated
  principal.

The mitigations are deliberately client-side and methodological rather than
cryptographic: the workload is pinned by suite id, runs that hit the KV prefix
cache are discarded, arms are interleaved so thermal drift cancels, and the
headline comparison is a within-session ratio that a fabricated absolute can't
usefully distort.

## Phase 2 (when macOS 27 is broadly out)

1. **App Attest** for a hardware-rooted "genuine app on real Apple hardware"
   signal. Requires macOS 27+, so rows keep a `trust` field and the board
   defaults to attested-only once adoption is there. Needs verifying against the
   **Developer ID / notarized DMG** build specifically — the App Store path is
   the one that's obviously supported.
2. **Payload signing** via the App Attest assertion over a SHA256 of the row.
   The key lives in the Secure Enclave and can't be extracted, so a proxy can
   read a submission but not alter it.
3. **Roofline validation** server-side: `decodeTps × bytes-per-token ≤ chip
   bandwidth`. This is the check that actually catches impossible numbers, and
   it needs a Cloud Function since rules can't do table lookups. Note that
   speculative decoding legitimately exceeds the roofline, so spec arms need a
   looser ceiling.
4. Writes move behind that function; the read path and the schema don't change.
