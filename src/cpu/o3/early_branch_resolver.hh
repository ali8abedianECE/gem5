#ifndef __CPU_O3_EARLY_BRANCH_RESOLVER_HH__
#define __CPU_O3_EARLY_BRANCH_RESOLVER_HH__

#include <deque>
#include <vector>

#include "base/statistics.hh"
#include "base/types.hh"
#include "cpu/inst_seq.hh"
#include "cpu/o3/dyn_inst_ptr.hh"
#include "cpu/o3/limits.hh"
#include "cpu/reg_class.hh"

namespace gem5
{

struct BaseO3CPUParams;

namespace o3
{

class CPU;

/**
 * EarlyBranchResolver (EBR)
 *
 * Sits inside the BAC stage and attempts to resolve conditional direct
 * branches before they reach the execute stage.
 *
 * Phase 1 (fetch time): if both source registers have no in-flight writes,
 * read their committed values and compute taken/not-taken immediately.
 * Falls back to the BPU (TAGE-SC-L) when registers are busy.
 *
 * Tracks in-flight register writes via a per-thread deque of (seqNum,
 * destRegs) entries.  On commit, entries are popped from the front.
 * On squash, entries after the squash seqNum are removed and counts
 * are rebuilt — no ROB access required.
 */
class EarlyBranchResolver
{
  public:
    EarlyBranchResolver(CPU *cpu, const BaseO3CPUParams &params);

    /**
     * Called from BAC::updatePC for EVERY fetched instruction.
     * Records any integer destination registers as having a pending write.
     */
    void notifyFetched(const DynInstPtr &inst, ThreadID tid);

    /**
     * Called from BAC when fromCommit->doneSeqNum advances (no squash).
     * Pops all deque entries at or before doneSeqNum and decrements counts.
     */
    void commitUpTo(ThreadID tid, InstSeqNum doneSeqNum);

    /**
     * Called from BAC when fromCommit signals a squash at squashSeqNum.
     * Removes all deque entries with seqNum > squashSeqNum and rebuilds counts.
     */
    void squashAfter(ThreadID tid, InstSeqNum squashSeqNum);

    /**
     * Phase 1 — fetch time.
     * Returns true and sets `taken` if both source regs have no pending writes.
     * Returns false if registers are busy (caller should use BPU).
     */
    bool tryResolve(const DynInstPtr &inst, ThreadID tid, bool &taken);

    /**
     * Phase 1.5 — loop predictor, fetch time.
     * For conditional direct branches whose source regs are busy, checks
     * whether the per-PC speculative iteration counter equals the learned
     * trip count.  If so, overrides TAGE with "not-taken" (loop exit).
     * Returns true and sets `taken` if a high-confidence prediction is made.
     */
    bool tryResolveLoop(const DynInstPtr &inst, ThreadID tid, bool &taken);

    /**
     * Phase 1.6 — change-count predictor, fetch time.
     * For branches that Phase 1 and 1.5 could not resolve, uses a 2D
     * saturating-counter table indexed by (pc hash, register change delta)
     * to predict direction based on how many times the source registers were
     * committed since the last visit to this branch PC.
     * Returns true and sets `taken` if confident.
     */
    bool tryResolveChangeCount(const DynInstPtr &inst, ThreadID tid,
                               bool &taken);

    /**
     * Phase 1.7 — GHR perceptron predictor, fetch time.
     * Falls back here when Phase 1.6 is not confident. Dots a per-PC weight
     * vector against recent branch history bits (±1 per bit) to predict.
     * Returns true and sets `taken` if |y| > PERC_THETA.
     */
    bool tryResolvePerceptron(const DynInstPtr &inst, ThreadID tid,
                              bool &taken);

    /**
     * Called from BAC after all prediction phases have run for a conditional
     * branch.  Updates the speculative GHR with the final predicted direction
     * so subsequent branches see an up-to-date history.
     */
    void notifyPredicted(const DynInstPtr &inst, ThreadID tid,
                         bool predicted_taken);

    /**
     * Phase 2 — wakeup time (called from IQ::wakeDependents).
     * All source regs are now in the physical register file.
     * Reads values directly and evaluates the branch condition.
     * Returns true and sets `taken` if branch type is known.
     * Also sets `mispredicted` if the result disagrees with predTaken.
     * Also updates the loop predictor with the actual outcome.
     */
    bool tryResolveWakeup(const DynInstPtr &inst, ThreadID tid,
                          bool &taken, bool &mispredicted);

    // Stats are public so bac.cc can increment override counters directly.
    struct EBRStats : public statistics::Group {
        EBRStats(CPU *cpu);
        statistics::Scalar resolvedAtFetch;
        statistics::Scalar fallbackBusy;
        statistics::Scalar fallbackNotCond;
        statistics::Scalar overrideTaken;
        statistics::Scalar overrideNotTaken;
        // Loop predictor (Phase 1.5)
        statistics::Scalar loopPredOverride;
        statistics::Scalar loopPredCorrect;
        statistics::Scalar loopPredWrong;
        // Phase 2
        statistics::Scalar resolvedAtWakeup;
        statistics::Scalar wakeupMispredCorrections;
        statistics::Scalar earlySquashesInitiated;
        // Change-count predictor (Phase 1.6)
        statistics::Scalar ccPredFired;
        statistics::Scalar ccPredCorrect;
        statistics::Scalar ccPredWrong;
        // Perceptron predictor (Phase 1.7)
        statistics::Scalar percPredFired;
        statistics::Scalar percPredCorrect;
        statistics::Scalar percPredWrong;
    } stats;

  private:
    bool canResolve(const DynInstPtr &inst, ThreadID tid);
    bool evaluateCondition(const DynInstPtr &inst, ThreadID tid) const;
    void rebuildCounts(ThreadID tid);

    CPU *cpu;

    // Tracks (seqNum -> [destIntRegIdx...]) and loop-branch info in flight.
    struct InFlightEntry {
        InstSeqNum seqNum;
        std::vector<int> destIntRegs;
        // Loop predictor
        int  loopPcIdx = -1;   // -1 means not a tracked loop branch
        // CC predictor (Phase 1.6)
        int16_t ccPcIdx      = -1;  // -1 = not a tracked branch
        uint8_t ccDelta      = 0;   // (d0 XOR d1) % CC_DELTA_MOD
        bool    ccFirst      = true;
        bool    ccPredMade   = false;
        bool    ccPredTaken  = false;
        int8_t  ccSrc0       = -1;
        int8_t  ccSrc1       = -1;
        // GHR perceptron (Phase 1.7)
        int16_t percPcIdx     = -1;  // path-sensitive table index (separate from ccPcIdx)
        int16_t percY         = 0;   // dot product at fetch time
        bool    percPredMade  = false;
        bool    percPredTaken = false;
        uint32_t ghrSnap      = 0;   // speculative GHR before this branch
        uint32_t ghrAfter     = 0;   // GHR after prediction (for squash restore)
        bool     ghrAfterSet  = false;
        bool     isGhrBranch  = false; // true = conditional direct branch
        bool     actualTaken  = false; // actual outcome (set at wakeup)
        bool     actualTakenSet = false;
    };
    std::deque<InFlightEntry> inFlight[MaxThreads];

    // Number of in-flight writes to each architectural integer register.
    static constexpr int MaxArchIntRegs = 32;
    int pendingWrites[MaxThreads][MaxArchIntRegs];

    // ── Loop predictor (Phase 1.5) ────────────────────────────────────────
    // Per-PC table: learns the trip count (number of taken outcomes before
    // a not-taken exit) and predicts "not-taken" on the exit iteration.
    static constexpr int LOOP_TABLE_BITS = 10;
    static constexpr int LOOP_TABLE_SIZE = 1 << LOOP_TABLE_BITS;
    static constexpr uint8_t LOOP_CONF_MAX  = 3;   // 2-bit saturating
    static constexpr uint8_t LOOP_CONF_HIGH = 2;   // threshold to predict

    struct LoopEntry {
        uint16_t tripCount = 0;   // learned exit iteration (taken count)
        uint16_t specIter  = 0;   // speculative fetch-time iter counter
        uint8_t  conf      = 0;   // 2-bit saturating confidence
        bool     valid     = false;
    };
    LoopEntry loopTable[MaxThreads][LOOP_TABLE_SIZE];

    int loopPcIdx(Addr pc) const { return (pc >> 1) & (LOOP_TABLE_SIZE - 1); }

    // ── Change-count predictor (Phase 1.6) ───────────────────────────────
    // For each conditional branch, counts how many committed writes occurred
    // to each source register since the last visit to this branch PC.
    // Indexes a 2D saturating-counter table with (pcIdx, delta_hash) to
    // predict taken/not-taken based on register mutation patterns.
    static constexpr int CC_PC_BITS   = 10;
    static constexpr int CC_PC_SIZE   = 1 << CC_PC_BITS;
    static constexpr int CC_DELTA_MOD = 16;            // 4-bit delta hash
    static constexpr uint8_t CC_CONF_MAX  = 7;         // 3-bit saturating
    static constexpr uint8_t CC_CONF_HIGH = 4;         // midpoint
    static constexpr uint8_t CC_CONF_INIT = 4;         // neutral start

    struct CCVisitEntry {
        uint32_t count0 = 0;   // regCommitCount for src0 at last commit
        uint32_t count1 = 0;   // regCommitCount for src1 at last commit
        bool     valid  = false;
    };
    CCVisitEntry ccVisit[MaxThreads][CC_PC_SIZE];
    uint8_t      ccPred [MaxThreads][CC_PC_SIZE][CC_DELTA_MOD];

    // Per-architectural-register monotonically increasing commit count.
    uint32_t regCommitCount[MaxThreads][MaxArchIntRegs];

    // ── GHR perceptron predictor (Phase 1.7) ─────────────────────────────
    // Per-branch-PC weight table indexed by PC hash.
    // Feature vector: [1, h0, h1, ..., h_{N-1}] where h_i = ±1 from GHR bit i.
    // y = Σ w[i]*h[i]. Confident if |y| > PERC_THETA; train if wrong OR low-conf.
    static constexpr int    PERC_TABLE_BITS  = 12;              // 4K entries
    static constexpr int    PERC_TABLE_SIZE  = 1 << PERC_TABLE_BITS;
    static constexpr int    PERC_N_HIST      = 16;              // GHR bits used
    static constexpr int    PERC_N_WEIGHTS   = PERC_N_HIST + 1; // +1 for bias
    static constexpr int8_t PERC_W_MAX       = 127;  // int8_t weight bound
    static constexpr int    PERC_THETA       = 500;  // confidence threshold

    struct PercEntry {
        int8_t w[PERC_N_WEIGHTS];
        PercEntry() { for (int i = 0; i < PERC_N_WEIGHTS; i++) w[i] = 0; }
    };
    PercEntry percTable[MaxThreads][PERC_TABLE_SIZE];

    // Speculative GHR: updated at fetch via notifyPredicted(), rolled back on squash.
    uint32_t specGhr  [MaxThreads] = {};
    uint32_t commitGhr[MaxThreads] = {};
};

} // namespace o3
} // namespace gem5

#endif // __CPU_O3_EARLY_BRANCH_RESOLVER_HH__
