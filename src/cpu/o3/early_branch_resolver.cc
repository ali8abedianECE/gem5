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
RegId intRegIdCache  [MaxThreads][32];
bool  intRegIdCached [MaxThreads][32] = {};

// ── Stride predictor ─────────────────────────────────────────────────────────
static constexpr uint8_t STRIDE_CONF_MAX  = 3;
static constexpr uint8_t STRIDE_CONF_HIGH = 2;

RegVal  strideLastVal[MaxThreads][32] = {};  // last committed value
int64_t strideVal    [MaxThreads][32] = {};  // learned stride
uint8_t strideConf   [MaxThreads][32] = {};  // saturating confidence

// ── In-flight write classification ───────────────────────────────────────────
// Separate in-flight writes into two classes:
//   consistent  — rd is also a source (in-place: addi k, k, 1).
//                 Predicted value = committed_val + count * stride.
//   breaking    — rd is NOT a source (copy/init: mv j, lo; lw t0, mem).
//                 Signals that the stride history is stale; block prediction.
//
// Per-register counters are rebuilt from writeKinds[] after squash.
int8_t strideConsistentInFlight[MaxThreads][32] = {};
int8_t strideBreakingInFlight  [MaxThreads][32] = {};

struct InFlightWriteKind {
    InstSeqNum          seqNum;
    std::vector<int>    destIntRegs;  // register indices (parallel to consistency)
    std::vector<int8_t> consistency;  // +1 = consistent, -1 = breaking
};
std::deque<InFlightWriteKind> writeKinds[MaxThreads];

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
        for (int r = 0; r < MaxArchIntRegs; r++) {
            pendingWrites[tid][r]  = 0;
            regCommitCount[tid][r] = 0;
        }
        for (int i = 0; i < LOOP_TABLE_SIZE; i++)
            loopTable[tid][i] = LoopEntry{};
        for (int i = 0; i < CC_PC_SIZE; i++) {
            ccVisit[tid][i] = CCVisitEntry{};
            for (int d = 0; d < CC_DELTA_MOD; d++)
                ccPred[tid][i][d] = CC_CONF_INIT;
        }
        for (int i = 0; i < PERC_TABLE_SIZE; i++)
            percTable[tid][i] = PercEntry{};
    }
}

// ── notifyFetched ─────────────────────────────────────────────────────────────

void
EarlyBranchResolver::notifyFetched(const DynInstPtr &inst, ThreadID tid)
{
    InFlightEntry entry;
    entry.seqNum = inst->seqNum;

    InFlightWriteKind kindEntry;
    kindEntry.seqNum = inst->seqNum;

    const StaticInstPtr &si0 = inst->staticInst;
    for (int i = 0; i < si0->numDestRegs(); i++) {
        const RegId &reg = si0->destRegIdx(i);
        if (reg.classValue() == IntRegClass) {
            int idx = reg.index();
            if (idx != 0) {
                entry.destIntRegs.push_back(idx);
                pendingWrites[tid][idx]++;
                if (!intRegIdCached[tid][idx]) {
                    intRegIdCache[tid][idx]  = reg;
                    intRegIdCached[tid][idx] = true;
                }
                // Classify: consistent = rd is also a source (in-place update,
                // e.g. addi k,k,1); breaking = rd is not a source (copy/init/
                // load, e.g. mv j,lo_reg or lw t0,mem).  Breaking writes make
                // the committed stride value stale; block stride prediction
                // while any breaking write to that register is in-flight.
                bool isConsistent = false;
                for (int s = 0; s < si0->numSrcRegs(); s++) {
                    const RegId &sr = si0->srcRegIdx(s);
                    if (sr.classValue() == IntRegClass && sr.index() == idx) {
                        isConsistent = true;
                        break;
                    }
                }
                if (isConsistent)
                    strideConsistentInFlight[tid][idx]++;
                else
                    strideBreakingInFlight[tid][idx]++;

                kindEntry.destIntRegs.push_back(idx);
                kindEntry.consistency.push_back(isConsistent ? int8_t(1) : int8_t(-1));
            }
        }
    }

    if (!kindEntry.destIntRegs.empty())
        writeKinds[tid].push_back(std::move(kindEntry));

    // Loop predictor + CC predictor: only for conditional direct branches.
    if (si0->isCondCtrl() && si0->isDirectCtrl() && si0->numSrcRegs() >= 1) {
        int idx = loopPcIdx(inst->pcState().instAddr());

        // Loop predictor (Phase 1.5)
        int inFlightCount = 0;
        for (const auto &e : inFlight[tid])
            if (e.loopPcIdx == idx) inFlightCount++;
        uint16_t fetchIter =
            loopTable[tid][idx].specIter + (uint16_t)inFlightCount + 1u;
        inst->setLoopFetchIter(fetchIter);
        entry.loopPcIdx = idx;

        // CC predictor (Phase 1.6): compute register change delta from
        // committed state since the last time this branch PC committed.
        entry.ccPcIdx = (int16_t)idx;  // same hash as loop table

        int8_t s0 = -1, s1 = -1;
        const RegId &r0 = si0->srcRegIdx(0);
        if (r0.classValue() == IntRegClass && r0.index() != 0)
            s0 = (int8_t)r0.index();
        if (si0->numSrcRegs() >= 2) {
            const RegId &r1 = si0->srcRegIdx(1);
            if (r1.classValue() == IntRegClass && r1.index() != 0)
                s1 = (int8_t)r1.index();
        }
        entry.ccSrc0 = s0;
        entry.ccSrc1 = s1;

        const CCVisitEntry &cv = ccVisit[tid][idx];
        if (!cv.valid) {
            entry.ccFirst = true;
        } else {
            uint32_t c0    = (s0 > 0) ? regCommitCount[tid][s0] : 0u;
            uint32_t c1    = (s1 > 0) ? regCommitCount[tid][s1] : 0u;
            uint32_t d0raw = c0 - cv.count0;
            uint32_t d1raw = c1 - cv.count1;
            entry.ccDelta = (uint8_t)((d0raw ^ d1raw) % CC_DELTA_MOD);
            entry.ccFirst = false;
        }

        // Phase 1.7 GHR perceptron: dot product against current speculative
        // history bits.  Computed for ALL conditional branches so the weight
        // table trains on every visit, not just post-first-commit visits.
        // Table index XORs PC hash with folded GHR for path-sensitivity.
        entry.isGhrBranch = true;
        entry.ghrSnap     = specGhr[tid];
        {
            Addr pc = inst->pcState().instAddr();
            uint32_t ghr = entry.ghrSnap;
            // Fold 32-bit GHR down to PERC_TABLE_BITS by XOR-folding.
            uint32_t foldedGhr = (ghr ^ (ghr >> PERC_TABLE_BITS)) &
                                  (PERC_TABLE_SIZE - 1);
            int percIdx = (int)(((pc >> 1) ^ foldedGhr) & (PERC_TABLE_SIZE - 1));
            entry.percPcIdx = (int16_t)percIdx;

            const PercEntry &pe = percTable[tid][percIdx];
            int16_t y = pe.w[0];  // bias term
            for (int i = 0; i < PERC_N_HIST; i++) {
                int16_t h = ((ghr >> i) & 1u) ? int16_t(1) : int16_t(-1);
                y += (int16_t)pe.w[i + 1] * h;
            }
            entry.percY = y;
        }
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
            regCommitCount[tid][r]++;

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

        // GHR perceptron: advance committed GHR when a conditional branch commits.
        if (front.isGhrBranch && front.actualTakenSet)
            commitGhr[tid] = (commitGhr[tid] << 1) | (front.actualTaken ? 1u : 0u);

        // CC predictor: update last-visit counts when a tracked branch commits.
        // Runs after regCommitCount increments so counts are current.
        if (front.ccPcIdx >= 0) {
            int s0 = front.ccSrc0;
            int s1 = front.ccSrc1;
            CCVisitEntry &cv = ccVisit[tid][front.ccPcIdx];
            cv.count0 = (s0 > 0) ? regCommitCount[tid][s0] : 0u;
            cv.count1 = (s1 > 0) ? regCommitCount[tid][s1] : 0u;
            cv.valid  = true;
        }

        // Drain writeKinds entry for this instruction (seqNum-matched).
        if (!writeKinds[tid].empty() &&
            writeKinds[tid].front().seqNum == front.seqNum) {
            const auto &wk = writeKinds[tid].front();
            for (int j = 0; j < (int)wk.destIntRegs.size(); j++) {
                int r = wk.destIntRegs[j];
                if (wk.consistency[j] > 0) {
                    if (strideConsistentInFlight[tid][r] > 0)
                        strideConsistentInFlight[tid][r]--;
                } else {
                    if (strideBreakingInFlight[tid][r] > 0)
                        strideBreakingInFlight[tid][r]--;
                }
            }
            writeKinds[tid].pop_front();
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

    while (!writeKinds[tid].empty() &&
           writeKinds[tid].back().seqNum > squashSeqNum)
        writeKinds[tid].pop_back();

    // Restore speculative GHR to the last surviving conditional branch's state.
    specGhr[tid] = commitGhr[tid];
    for (auto it = inFlight[tid].rbegin(); it != inFlight[tid].rend(); ++it) {
        if (it->isGhrBranch && it->ghrAfterSet) {
            specGhr[tid] = it->ghrAfter;
            break;
        }
    }

    rebuildCounts(tid);
}

// ── rebuildCounts ─────────────────────────────────────────────────────────────

void
EarlyBranchResolver::rebuildCounts(ThreadID tid)
{
    for (int r = 0; r < MaxArchIntRegs; r++) {
        pendingWrites[tid][r] = 0;
        strideConsistentInFlight[tid][r] = 0;
        strideBreakingInFlight[tid][r] = 0;
    }
    for (const auto &wk : writeKinds[tid]) {
        for (int j = 0; j < (int)wk.destIntRegs.size(); j++) {
            int r = wk.destIntRegs[j];
            pendingWrites[tid][r]++;
            if (wk.consistency[j] > 0)
                strideConsistentInFlight[tid][r]++;
            else
                strideBreakingInFlight[tid][r]++;
        }
    }
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
            if (reg.classValue() == IntRegClass && reg.index() != 0 &&
                strideConsistentInFlight[tid][reg.index()] > 0)
                strideUsed = true;
        }

        if (strideUsed) {
            if (taken) {
                // Taken-with-stride: let BPU decide (unreliable).
                return false;
            }
            // Count stride-busy sources to choose the right safety check.
            int strideCount = 0;
            for (int i = 0; i < si2->numSrcRegs() && i < 2; i++) {
                const RegId &r2 = si2->srcRegIdx(i);
                if (r2.classValue() == IntRegClass && r2.index() != 0 &&
                    strideConsistentInFlight[tid][r2.index()] > 0)
                    strideCount++;
            }
            if (strideCount < 2) {
                // Single-busy: require exact boundary (v1_pred == v2_committed).
                // This guards against stride overrides far from the exit point.
                auto readVal = [&](const RegId &r) -> RegVal {
                    int idx = r.index();
                    if (idx == 0) return 0;
                    int wb = strideConsistentInFlight[tid][idx];
                    if (wb > 0 && strideConf[tid][idx] >= STRIDE_CONF_HIGH)
                        return (RegVal)((int64_t)strideLastVal[tid][idx] +
                                       (int64_t)wb * strideVal[tid][idx]);
                    return cpu->getArchReg(r, tid);
                };
                RegVal v1 = readVal(si2->srcRegIdx(0));
                RegVal v2 = (si2->numSrcRegs() >= 2) ?
                             readVal(si2->srcRegIdx(1)) : RegVal(0);
                if (v1 != v2)
                    return false;
            }
            // Dual-busy: both values are stride-predicted with high confidence;
            // evaluateCondition already computed the not-taken result from them.
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

    int busyCount = 0;
    for (int i = 0; i < srcs && i < 2; i++) {
        const RegId &reg = si->srcRegIdx(i);
        if (reg.classValue() != IntRegClass) return false;
        int idx = reg.index();
        if (idx == 0) continue;

        // Any breaking write in-flight (copy/load/init) means the stride
        // history is stale for this register — block prediction entirely.
        if (strideBreakingInFlight[tid][idx] > 0) {
            ++stats.fallbackBusy;
            return false;
        }

        if (strideConsistentInFlight[tid][idx] > 0)
            busyCount++;
    }

    if (busyCount > 2) {
        ++stats.fallbackBusy;
        return false;
    }

    if (busyCount == 2) {
        // Dual-counter loop: both sources have consistent in-flight writes
        // (e.g. bge a1,s0 where addiw a1,a1,-1 and addiw s0,s0,1 are in-flight).
        // Allow stride resolution when both have high confidence.
        for (int i = 0; i < srcs && i < 2; i++) {
            const RegId &reg = si->srcRegIdx(i);
            if (reg.classValue() != IntRegClass) { ++stats.fallbackBusy; return false; }
            int idx = reg.index();
            if (idx == 0) continue;
            if (strideConsistentInFlight[tid][idx] > 0 &&
                strideConf[tid][idx] < STRIDE_CONF_HIGH) {
                ++stats.fallbackBusy;
                return false;
            }
        }
        return true;
    }

    if (busyCount == 1) {
        // For stride-based override, require high confidence.
        for (int i = 0; i < srcs && i < 2; i++) {
            const RegId &reg = si->srcRegIdx(i);
            if (reg.classValue() != IntRegClass) return false;
            int idx = reg.index();
            if (idx == 0) continue;
            if (strideConsistentInFlight[tid][idx] > 0 &&
                strideConf[tid][idx] < STRIDE_CONF_HIGH) {
                ++stats.fallbackBusy;
                return false;
            }
        }
        // Stride-based not-taken overrides are only safe for branch types where
        // the not-taken condition is an exact POINT (not a range).
        // blt/bltu  not-taken: a >= b — exact boundary = a == b ✓
        // bne/c.bnez not-taken: a == b — exact point ✓
        // beq/bge/bgeu not-taken: range conditions — stride overrides unreliable.
        const std::string &mnem = si->getName();
        if (mnem != "blt" && mnem != "bltu" &&
            mnem != "bne" && mnem != "c.bnez") {
            ++stats.fallbackBusy;
            return false;
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

    // strideConsistentInFlight counts in-place writes (addi k,k,1) fetched
    // before this branch — correct to use for stride-predicted value.
    auto readReg = [&](const RegId &r) -> RegVal {
        int idx = r.index();
        if (idx == 0) return 0;
        int wb = strideConsistentInFlight[tid][idx];
        if (wb > 0 && strideConf[tid][idx] >= STRIDE_CONF_HIGH)
            return (RegVal)((int64_t)strideLastVal[tid][idx] +
                            (int64_t)wb * strideVal[tid][idx]);
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

// ── tryResolveChangeCount (Phase 1.6) ────────────────────────────────────────

bool
EarlyBranchResolver::tryResolveChangeCount(const DynInstPtr &inst,
                                            ThreadID tid, bool &taken)
{
    // Locate the InFlightEntry written by notifyFetched; search from back
    // since this inst was just pushed (fetch order).
    for (auto it = inFlight[tid].rbegin(); it != inFlight[tid].rend(); ++it) {
        if (it->seqNum != inst->seqNum) continue;
        if (it->ccPcIdx < 0 || it->ccFirst) return false;

        uint8_t ctr = ccPred[tid][it->ccPcIdx][it->ccDelta];
        // Only predict when counter is saturated (0 = strongly not-taken,
        // 3 = strongly taken); intermediate values are too uncertain.
        if (ctr != 0 && ctr != CC_CONF_MAX) return false;

        taken             = (ctr >= CC_CONF_HIGH);
        it->ccPredMade    = true;
        it->ccPredTaken   = taken;
        ++stats.ccPredFired;

        DPRINTF(EarlyBranchResolver,
                "[tid:%i] [sn:%llu] EBR CC pred delta=%u ctr=%u -> %s\n",
                tid, inst->seqNum, it->ccDelta, ctr,
                taken ? "taken" : "not taken");
        return true;
    }
    return false;
}

// ── tryResolvePerceptron (Phase 1.7) ─────────────────────────────────────────

bool
EarlyBranchResolver::tryResolvePerceptron(const DynInstPtr &inst,
                                           ThreadID tid, bool &taken)
{
    for (auto it = inFlight[tid].rbegin(); it != inFlight[tid].rend(); ++it) {
        if (it->seqNum != inst->seqNum) continue;
        if (!it->isGhrBranch || it->percPcIdx < 0) return false;

        int16_t y = it->percY;
        if (y <= PERC_THETA && y >= -PERC_THETA) return false;  // not confident

        taken              = (y > 0);
        it->percPredMade   = true;
        it->percPredTaken  = taken;
        ++stats.percPredFired;

        DPRINTF(EarlyBranchResolver,
                "[tid:%i] [sn:%llu] EBR perceptron y=%d -> %s\n",
                tid, inst->seqNum, y, taken ? "taken" : "not taken");
        return true;
    }
    return false;
}

// ── notifyPredicted ───────────────────────────────────────────────────────────

void
EarlyBranchResolver::notifyPredicted(const DynInstPtr &inst, ThreadID tid,
                                      bool predicted_taken)
{
    for (auto it = inFlight[tid].rbegin(); it != inFlight[tid].rend(); ++it) {
        if (it->seqNum != inst->seqNum) continue;
        if (!it->isGhrBranch) return;
        it->ghrAfter    = (it->ghrSnap << 1) | (predicted_taken ? 1u : 0u);
        it->ghrAfterSet = true;
        specGhr[tid]    = it->ghrAfter;
        return;
    }
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

    // Update CC + perceptron predictors with the actual outcome.
    for (auto &e : inFlight[tid]) {
        if (e.seqNum != inst->seqNum || e.ccPcIdx < 0) continue;

        // CC saturating counter update.
        uint8_t &ctr = ccPred[tid][e.ccPcIdx][e.ccDelta];
        if (taken && ctr < CC_CONF_MAX) ctr++;
        else if (!taken && ctr > 0) ctr--;
        if (e.ccPredMade) {
            if (e.ccPredTaken == taken) ++stats.ccPredCorrect;
            else                        ++stats.ccPredWrong;
        }

        // GHR perceptron weight update (Phase 1.7).
        // Record actual outcome for committed GHR update.
        e.actualTaken    = taken;
        e.actualTakenSet = true;

        if (e.isGhrBranch && e.percPcIdx >= 0) {
            int16_t y      = e.percY;
            bool wrong     = e.percPredMade && (e.percPredTaken != taken);
            bool low_conf  = (y >= -PERC_THETA && y <= PERC_THETA);
            if (wrong || low_conf) {
                int8_t sign = taken ? int8_t(1) : int8_t(-1);
                PercEntry &pe = percTable[tid][e.percPcIdx];
                auto clamp = [](int16_t v) -> int8_t {
                    return (int8_t)(v > PERC_W_MAX ? PERC_W_MAX :
                                    v < -PERC_W_MAX ? -PERC_W_MAX : v);
                };
                pe.w[0] = clamp((int16_t)pe.w[0] + sign);  // bias
                uint32_t ghr = e.ghrSnap;
                for (int i = 0; i < PERC_N_HIST; i++) {
                    int16_t h = ((ghr >> i) & 1u) ? int16_t(1) : int16_t(-1);
                    pe.w[i + 1] = clamp((int16_t)pe.w[i + 1] + sign * h);
                }
            }
            if (e.percPredMade) {
                if (e.percPredTaken == taken) ++stats.percPredCorrect;
                else                          ++stats.percPredWrong;
            }
        }
        break;
    }

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
               "Squashes initiated at wakeup time (1 cycle before execute)"),
      ADD_STAT(ccPredFired, statistics::units::Count::get(),
               "CC predictor overrode BPU (Phase 1.6 prediction)"),
      ADD_STAT(ccPredCorrect, statistics::units::Count::get(),
               "CC predictor override was correct"),
      ADD_STAT(ccPredWrong, statistics::units::Count::get(),
               "CC predictor override was wrong"),
      ADD_STAT(percPredFired, statistics::units::Count::get(),
               "Perceptron predictor overrode BPU (Phase 1.7 prediction)"),
      ADD_STAT(percPredCorrect, statistics::units::Count::get(),
               "Perceptron predictor override was correct"),
      ADD_STAT(percPredWrong, statistics::units::Count::get(),
               "Perceptron predictor override was wrong")
{}

} // namespace o3
} // namespace gem5
