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
        // Loop predictor: if this is a conditional branch, record the
        // PC table index so squashAfter can decrement the specIter.
        int  loopPcIdx = -1;   // -1 means not a tracked loop branch
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
};

} // namespace o3
} // namespace gem5

#endif // __CPU_O3_EARLY_BRANCH_RESOLVER_HH__
