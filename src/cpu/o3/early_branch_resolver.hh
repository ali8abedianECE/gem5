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
     * Phase 2 — wakeup time (called from IQ::wakeDependents).
     * All source regs are now in the physical register file.
     * Reads values directly and evaluates the branch condition.
     * Returns true and sets `taken` if branch type is known.
     * Also sets `mispredicted` if the result disagrees with predTaken.
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
        statistics::Scalar resolvedAtWakeup;
        statistics::Scalar wakeupMispredCorrections;
    } stats;

  private:
    bool canResolve(const DynInstPtr &inst, ThreadID tid);
    bool evaluateCondition(const DynInstPtr &inst, ThreadID tid) const;
    void rebuildCounts(ThreadID tid);

    CPU *cpu;

    // Tracks (seqNum -> [destIntRegIdx...]) for instructions in flight.
    struct InFlightEntry {
        InstSeqNum seqNum;
        std::vector<int> destIntRegs;
    };
    std::deque<InFlightEntry> inFlight[MaxThreads];

    // Number of in-flight writes to each architectural integer register.
    static constexpr int MaxArchIntRegs = 32;
    int pendingWrites[MaxThreads][MaxArchIntRegs];
};

} // namespace o3
} // namespace gem5

#endif // __CPU_O3_EARLY_BRANCH_RESOLVER_HH__
