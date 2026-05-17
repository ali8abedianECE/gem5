#include "cpu/o3/early_branch_resolver.hh"

#include <cstdint>
#include <deque>
#include <string>

#include "cpu/o3/cpu.hh"
#include "cpu/o3/dyn_inst.hh"
#include "debug/EarlyBranchResolver.hh"
#include "params/BaseO3CPU.hh"

namespace gem5
{
namespace o3
{

// ── File-scope state ──────────────────────────────────────────────────────────
namespace {

// ── RegId cache ───────────────────────────────────────────────────────────────
// Populated lazily in notifyFetched (where the full RegId is available from
// staticInst->destRegIdx).  Used in commitUpTo so we can call getArchReg
// without constructing a RegId from a raw integer index (no public ctor).
RegId intRegIdCache  [MaxThreads][32];
bool  intRegIdCached [MaxThreads][32] = {};

// ── Stride predictor ─────────────────────────────────────────────────────────
static constexpr uint8_t STRIDE_CONF_MAX  = 3;
static constexpr uint8_t STRIDE_CONF_HIGH = 2;

RegVal  strideLastVal[MaxThreads][32] = {};  // last committed value
int64_t strideVal    [MaxThreads][32] = {};  // learned stride
uint8_t strideConf   [MaxThreads][32] = {};  // 2-bit saturating confidence

// ── Loop predictor pending-exit queue ────────────────────────────────────────
struct PendingLoopExit {
    InstSeqNum seqNum;
    int        loopPcIdx;
};
std::deque<PendingLoopExit> pendingExits[MaxThreads];

} // anonymous namespace

// ── Constructor ───────────────────────────────────────────────────────────────

EarlyBranchResolver::EarlyBranchResolver(CPU *_cpu,
                                          const BaseO3CPUParams &params)
    : stats(_cpu), cpu(_cpu)
{
    for (int tid = 0; tid < MaxThreads; tid++) {
        for (int r = 0; r < MaxArchIntRegs; r++)
            pendingWrites[tid][r] = 0;
        for (int i = 0; i < LOOP_TABLE_SIZE; i++)
            loopTable[tid][i] = LoopEntry{};
    }
}

// ── notifyFetched ─────────────────────────────────────────────────────────────

void
EarlyBranchResolver::notifyFetched(const DynInstPtr &inst, ThreadID tid)
{
    InFlightEntry entry;
    entry.seqNum = inst->seqNum;

    for (int i = 0; i < inst->staticInst->numDestRegs(); i++) {
        const RegId &reg = inst->staticInst->destRegIdx(i);
        if (reg.classValue() == IntRegClass) {
            int idx = reg.index();
            if (idx != 0) {
                entry.destIntRegs.push_back(idx);
                pendingWrites[tid][idx]++;
                if (!intRegIdCached[tid][idx]) {
                    intRegIdCache[tid][idx]  = reg;
                    intRegIdCached[tid][idx] = true;
                }
                // Detect register-reset writes (addi rd, x0, imm / mv rd, x0).
                // Any write sourced from the zero register sets the dest to a
                // constant, breaking the stride sequence.  Invalidate at fetch
                // so the stride predictor doesn't fire on stale history during
                // the reset transition (before the reset write commits).
                const StaticInstPtr &si = inst->staticInst;
                for (int s = 0; s < si->numSrcRegs(); s++) {
                    const RegId &sr = si->srcRegIdx(s);
                    if (sr.classValue() == IntRegClass && sr.index() == 0) {
                        strideConf[tid][idx] = 0;  // reset → invalidate
                        break;
                    }
                }
            }
        }
    }

    // Loop predictor: record the speculative iteration for conditional direct branches.
    const StaticInstPtr &si = inst->staticInst;
    if (si->isCondCtrl() && si->isDirectCtrl() && si->numSrcRegs() >= 1) {
        int idx = loopPcIdx(inst->pcState().instAddr());
        int inFlightCount = 0;
        for (const auto &e : inFlight[tid])
            if (e.loopPcIdx == idx) inFlightCount++;
        uint16_t fetchIter =
            loopTable[tid][idx].specIter + (uint16_t)inFlightCount + 1u;
        inst->setLoopFetchIter(fetchIter);
        entry.loopPcIdx = idx;
    }

    inFlight[tid].push_back(std::move(entry));
}

// ── commitUpTo ────────────────────────────────────────────────────────────────

void
EarlyBranchResolver::commitUpTo(ThreadID tid, InstSeqNum doneSeqNum)
{
    while (!inFlight[tid].empty() &&
           inFlight[tid].front().seqNum <= doneSeqNum) {
        const InFlightEntry &front = inFlight[tid].front();

        for (int r : front.destIntRegs) {
            if (pendingWrites[tid][r] > 0) pendingWrites[tid][r]--;

            // Update stride predictor from committed register value.
            if (!intRegIdCached[tid][r]) continue;
            RegVal newVal = cpu->getArchReg(intRegIdCache[tid][r], tid);
            int64_t newStride = (int64_t)newVal - (int64_t)strideLastVal[tid][r];
            if (strideConf[tid][r] == 0 || newStride == strideVal[tid][r]) {
                strideVal[tid][r] = newStride;
                if (strideConf[tid][r] < STRIDE_CONF_MAX) strideConf[tid][r]++;
            } else {
                // Stride changed — degrade and re-learn.
                strideConf[tid][r] = 0;
                strideVal[tid][r]   = newStride;
            }
            strideLastVal[tid][r] = newVal;
        }

        // Loop predictor: advance or reset committedIter.
        int lpIdx = front.loopPcIdx;
        if (lpIdx >= 0) {
            bool isExit = !pendingExits[tid].empty() &&
                          pendingExits[tid].front().seqNum == front.seqNum &&
                          pendingExits[tid].front().loopPcIdx == lpIdx;
            if (isExit) {
                pendingExits[tid].pop_front();
                loopTable[tid][lpIdx].specIter = 0;
            } else {
                loopTable[tid][lpIdx].specIter++;
            }
        }

        inFlight[tid].pop_front();
    }
}

// ── squashAfter ───────────────────────────────────────────────────────────────

void
EarlyBranchResolver::squashAfter(ThreadID tid, InstSeqNum squashSeqNum)
{
    while (!inFlight[tid].empty() &&
           inFlight[tid].back().seqNum > squashSeqNum)
        inFlight[tid].pop_back();

    while (!pendingExits[tid].empty() &&
           pendingExits[tid].back().seqNum > squashSeqNum)
        pendingExits[tid].pop_back();

    rebuildCounts(tid);
}

// ── rebuildCounts ─────────────────────────────────────────────────────────────

void
EarlyBranchResolver::rebuildCounts(ThreadID tid)
{
    // Only rebuild pendingWrites; stride tables and committedIter are
    // based on committed state and are unaffected by squash.
    for (int r = 0; r < MaxArchIntRegs; r++)
        pendingWrites[tid][r] = 0;

    for (const auto &entry : inFlight[tid])
        for (int r : entry.destIntRegs)
            pendingWrites[tid][r]++;
}

// ── tryResolve (Phase 1 + Phase 1b stride) ───────────────────────────────────

bool
EarlyBranchResolver::tryResolve(const DynInstPtr &inst, ThreadID tid,
                                 bool &taken)
{
    if (!canResolve(inst, tid))
        return false;

    taken = evaluateCondition(inst, tid);

    {
        // Determine whether stride prediction influenced the result by
        // checking if any source reg had in-flight writes up to this branch.
        const StaticInstPtr &si2 = inst->staticInst;
        bool strideUsed = false;
        for (int i = 0; i < si2->numSrcRegs() && i < 2 && !strideUsed; i++) {
            const RegId &reg = si2->srcRegIdx(i);
            if (reg.classValue() == IntRegClass && reg.index() != 0) {
                for (const auto &e : inFlight[tid]) {
                    if (e.seqNum > inst->seqNum) break;
                    for (int r : e.destIntRegs)
                        if (r == reg.index()) { strideUsed = true; break; }
                    if (strideUsed) break;
                }
            }
        }

        if (strideUsed) {
            if (taken) {
                // Taken-with-stride: let BPU decide (unreliable).
                return false;
            }
            // Not-taken-with-stride: only override for blt/bltu when the
            // predicted counter exactly equals the bound (v1 == v2).
            // If v1 > v2 (overshoot from reset-transition), fall back.
            const std::string &mnem = si2->getName();
            if (mnem == "blt" || mnem == "bltu") {
                // Re-read values using writes-up-to-this-branch.
                auto readVal = [&](const RegId &r) -> RegVal {
                    int idx = r.index();
                    if (idx == 0) return 0;
                    int wb = 0;
                    for (const auto &e : inFlight[tid]) {
                        if (e.seqNum > inst->seqNum) break;
                        for (int rr : e.destIntRegs)
                            if (rr == idx) wb++;
                    }
                    if (wb > 0 && strideConf[tid][idx] >= STRIDE_CONF_HIGH)
                        return (RegVal)((int64_t)strideLastVal[tid][idx] +
                                       (int64_t)wb * strideVal[tid][idx]);
                    return cpu->getArchReg(r, tid);
                };
                RegVal v1 = readVal(si2->srcRegIdx(0));
                RegVal v2 = (si2->numSrcRegs() >= 2) ?
                             readVal(si2->srcRegIdx(1)) : RegVal(0);
                if (v1 != v2)
                    return false;  // overshoot — likely reset transition
            }
        }
    }

    ++stats.resolvedAtFetch;

    DPRINTF(EarlyBranchResolver,
            "[tid:%i] [sn:%llu] EBR resolved %s -> %s\n",
            tid, inst->seqNum,
            inst->staticInst->getName().c_str(),
            taken ? "taken" : "not taken");

    return true;
}

// ── canResolve ────────────────────────────────────────────────────────────────
//
// A source register is "resolvable" if:
//   (a) it has no pending writes, OR
//   (b) it has pending writes but the stride predictor is confident — then
//       evaluateCondition() will substitute the stride-predicted value.

bool
EarlyBranchResolver::canResolve(const DynInstPtr &inst, ThreadID tid)
{
    const StaticInstPtr &si = inst->staticInst;

    if (!si->isCondCtrl() || !si->isDirectCtrl()) {
        ++stats.fallbackNotCond;
        return false;
    }

    int srcs = si->numSrcRegs();
    if (srcs < 1) return false;

    // Count writes-before-this-branch for each source reg.
    // This correctly excludes speculated instructions fetched *after* this
    // branch, which would otherwise inflate the stride prediction.
    auto writesUpTo = [&](int regIdx) -> int {
        int count = 0;
        for (const auto &e : inFlight[tid]) {
            if (e.seqNum > inst->seqNum) break;
            for (int r : e.destIntRegs)
                if (r == regIdx) count++;
        }
        return count;
    };

    int busyCount = 0;
    for (int i = 0; i < srcs && i < 2; i++) {
        const RegId &reg = si->srcRegIdx(i);
        if (reg.classValue() != IntRegClass) return false;
        int idx = reg.index();
        if (idx != 0 && writesUpTo(idx) > 0)
            busyCount++;
    }

    // Only stride-predict when exactly one source is busy (loop counter vs.
    // constant).  Both busy → data-dependent comparison → fallback to BPU.
    if (busyCount >= 2) {
        ++stats.fallbackBusy;
        return false;
    }

    if (busyCount == 1) {
        for (int i = 0; i < srcs && i < 2; i++) {
            const RegId &reg = si->srcRegIdx(i);
            if (reg.classValue() != IntRegClass) return false;
            int idx = reg.index();
            if (idx != 0 && writesUpTo(idx) > 0 &&
                strideConf[tid][idx] < STRIDE_CONF_HIGH) {
                ++stats.fallbackBusy;
                return false;
            }
        }
    }

    return true;
}

// ── evaluateCondition ─────────────────────────────────────────────────────────
//
// For registers with pending writes AND confident stride, substitutes:
//   predicted_value = lastCommittedValue + pendingWrites * stride

bool
EarlyBranchResolver::evaluateCondition(const DynInstPtr &inst,
                                        ThreadID tid) const
{
    const StaticInstPtr &si = inst->staticInst;

    // Count writes to each source register from instructions *at or before*
    // this branch in program order.  pendingWrites[] includes speculated
    // instructions fetched *after* this branch, which would inflate the
    // stride prediction and produce wrong not-taken overrides.
    auto writesBeforeBranch = [&](int regIdx) -> int {
        int count = 0;
        for (const auto &e : inFlight[tid]) {
            if (e.seqNum > inst->seqNum) break;
            for (int r : e.destIntRegs)
                if (r == regIdx) count++;
        }
        return count;
    };

    auto readReg = [&](const RegId &r) -> RegVal {
        int idx = r.index();
        if (idx == 0) return 0;
        int wb = writesBeforeBranch(idx);
        if (wb > 0 && strideConf[tid][idx] >= STRIDE_CONF_HIGH) {
            // Stride prediction using only writes up to this branch.
            return (RegVal)((int64_t)strideLastVal[tid][idx] +
                            (int64_t)wb * strideVal[tid][idx]);
        }
        return cpu->getArchReg(r, tid);
    };

    const RegId &r1 = si->srcRegIdx(0);
    RegVal v1 = readReg(r1);

    RegVal v2 = 0;
    if (si->numSrcRegs() >= 2)
        v2 = readReg(si->srcRegIdx(1));

    const std::string &mnemonic = si->getName();

    if (mnemonic == "beq"   || mnemonic == "c.beqz") return v1 == v2;
    if (mnemonic == "bne"   || mnemonic == "c.bnez") return v1 != v2;
    if (mnemonic == "blt")  return (int64_t)v1 <  (int64_t)v2;
    if (mnemonic == "bge")  return (int64_t)v1 >= (int64_t)v2;
    if (mnemonic == "bltu") return v1 <  v2;
    if (mnemonic == "bgeu") return v1 >= v2;

    return false;
}

// ── tryResolveLoop (Phase 1.5) ────────────────────────────────────────────────

bool
EarlyBranchResolver::tryResolveLoop(const DynInstPtr &inst, ThreadID tid,
                                     bool &taken)
{
    const StaticInstPtr &si = inst->staticInst;
    if (!si->isCondCtrl() || !si->isDirectCtrl()) return false;
    if (si->numSrcRegs() < 1) return false;

    int idx = loopPcIdx(inst->pcState().instAddr());
    const LoopEntry &le = loopTable[tid][idx];

    if (!le.valid || le.conf < LOOP_CONF_HIGH || le.tripCount == 0)
        return false;

    // Compute true speculative iteration: committedIter + in-flight count.
    int inFlightCount = 0;
    for (const auto &e : inFlight[tid])
        if (e.loopPcIdx == idx) inFlightCount++;

    uint16_t trueIter = le.specIter + (uint16_t)inFlightCount;
    if (trueIter != le.tripCount)
        return false;

    taken = false;
    ++stats.loopPredOverride;

    DPRINTF(EarlyBranchResolver,
            "[tid:%i] [sn:%llu] EBR loop pred exit at iter %u "
            "(committedIter=%u inFlight=%d tripCount=%u)\n",
            tid, inst->seqNum, trueIter, le.specIter, inFlightCount,
            le.tripCount);

    return true;
}

// ── tryResolveWakeup (Phase 2) ────────────────────────────────────────────────

bool
EarlyBranchResolver::tryResolveWakeup(const DynInstPtr &inst, ThreadID tid,
                                       bool &taken, bool &mispredicted)
{
    const StaticInstPtr &si = inst->staticInst;

    if (!si->isCondCtrl() || !si->isDirectCtrl()) return false;
    if (si->numSrcRegs() < 1) return false;

    RegVal v1 = inst->getRegOperand(si.get(), 0);
    RegVal v2 = (si->numSrcRegs() >= 2) ? inst->getRegOperand(si.get(), 1) : 0;

    const std::string &mnemonic = si->getName();
    bool resolved = true;

    if      (mnemonic == "beq"   || mnemonic == "c.beqz") taken = (v1 == v2);
    else if (mnemonic == "bne"   || mnemonic == "c.bnez") taken = (v1 != v2);
    else if (mnemonic == "blt")  taken = ((int64_t)v1 <  (int64_t)v2);
    else if (mnemonic == "bge")  taken = ((int64_t)v1 >= (int64_t)v2);
    else if (mnemonic == "bltu") taken = (v1 <  v2);
    else if (mnemonic == "bgeu") taken = (v1 >= v2);
    else resolved = false;

    if (!resolved) return false;

    mispredicted = (taken != inst->readPredTaken());
    ++stats.resolvedAtWakeup;
    if (mispredicted) ++stats.wakeupMispredCorrections;

    DPRINTF(EarlyBranchResolver,
            "[tid:%i] [sn:%llu] EBR wakeup resolved %s -> %s "
            "(pred was %s, %s)\n",
            tid, inst->seqNum, si->getName().c_str(),
            taken ? "taken" : "not taken",
            inst->readPredTaken() ? "taken" : "not taken",
            mispredicted ? "MISPRED" : "correct");

    // Update loop predictor.
    int idx = loopPcIdx(inst->pcState().instAddr());
    LoopEntry &le = loopTable[tid][idx];
    uint16_t fetchIter = inst->getLoopFetchIter();

    bool loopPredFired = le.valid && le.conf >= LOOP_CONF_HIGH &&
                         le.tripCount > 0 && fetchIter == le.tripCount;

    if (!taken) {
        pendingExits[tid].push_back({inst->seqNum, idx});
        if (!le.valid || fetchIter == le.tripCount) {
            le.tripCount = fetchIter;
            le.valid     = true;
            if (le.conf < LOOP_CONF_MAX) le.conf++;
        } else {
            le.tripCount = fetchIter;
            if (le.conf > 0) le.conf--;
            if (le.conf == 0) le.valid = false;
        }
        if (loopPredFired) ++stats.loopPredCorrect;
    } else {
        if (loopPredFired) ++stats.loopPredWrong;
    }

    return true;
}

// ── Stats ─────────────────────────────────────────────────────────────────────

EarlyBranchResolver::EBRStats::EBRStats(CPU *cpu)
    : statistics::Group(cpu, "earlyBranchResolver"),
      ADD_STAT(resolvedAtFetch, statistics::units::Count::get(),
               "Branches resolved early at fetch (Phase 1 + stride pred)"),
      ADD_STAT(fallbackBusy, statistics::units::Count::get(),
               "Branches falling back to BPU: source reg busy, no stride"),
      ADD_STAT(fallbackNotCond, statistics::units::Count::get(),
               "Non-conditional or indirect branches skipped by EBR"),
      ADD_STAT(overrideTaken, statistics::units::Count::get(),
               "EBR overrode BPU from not-taken to taken"),
      ADD_STAT(overrideNotTaken, statistics::units::Count::get(),
               "EBR overrode BPU from taken to not-taken"),
      ADD_STAT(loopPredOverride, statistics::units::Count::get(),
               "EBR loop predictor overrode BPU (predicted loop exit)"),
      ADD_STAT(loopPredCorrect, statistics::units::Count::get(),
               "EBR loop predictor override was correct (actual not-taken)"),
      ADD_STAT(loopPredWrong, statistics::units::Count::get(),
               "EBR loop predictor override was wrong (actual taken)"),
      ADD_STAT(resolvedAtWakeup, statistics::units::Count::get(),
               "Branches resolved at IQ wakeup time (Phase 2)"),
      ADD_STAT(wakeupMispredCorrections, statistics::units::Count::get(),
               "Wakeup resolutions that caught a BPU misprediction early"),
      ADD_STAT(earlySquashesInitiated, statistics::units::Count::get(),
               "Squashes initiated at wakeup time (1 cycle before execute)")
{}

} // namespace o3
} // namespace gem5
