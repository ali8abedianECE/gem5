#include "cpu/o3/early_branch_resolver.hh"

#include <cstdint>
#include <string>

#include "cpu/o3/cpu.hh"
#include "cpu/o3/dyn_inst.hh"
#include "debug/EarlyBranchResolver.hh"
#include "params/BaseO3CPU.hh"

namespace gem5
{
namespace o3
{

EarlyBranchResolver::EarlyBranchResolver(CPU *_cpu,
                                          const BaseO3CPUParams &params)
    : stats(_cpu), cpu(_cpu)
{
    for (int tid = 0; tid < MaxThreads; tid++)
        for (int r = 0; r < MaxArchIntRegs; r++)
            pendingWrites[tid][r] = 0;
}

void
EarlyBranchResolver::notifyFetched(const DynInstPtr &inst, ThreadID tid)
{
    InFlightEntry entry;
    entry.seqNum = inst->seqNum;

    for (int i = 0; i < inst->staticInst->numDestRegs(); i++) {
        const RegId &reg = inst->staticInst->destRegIdx(i);
        if (reg.classValue() == IntRegClass) {
            int idx = reg.index();
            if (idx != 0) {  // x0 is hardwired zero, never written
                entry.destIntRegs.push_back(idx);
                pendingWrites[tid][idx]++;
            }
        }
    }

    inFlight[tid].push_back(std::move(entry));
}

void
EarlyBranchResolver::commitUpTo(ThreadID tid, InstSeqNum doneSeqNum)
{
    while (!inFlight[tid].empty() &&
           inFlight[tid].front().seqNum <= doneSeqNum) {
        for (int r : inFlight[tid].front().destIntRegs) {
            if (pendingWrites[tid][r] > 0)
                pendingWrites[tid][r]--;
        }
        inFlight[tid].pop_front();
    }
}

void
EarlyBranchResolver::squashAfter(ThreadID tid, InstSeqNum squashSeqNum)
{
    while (!inFlight[tid].empty() &&
           inFlight[tid].back().seqNum > squashSeqNum) {
        inFlight[tid].pop_back();
    }
    rebuildCounts(tid);
}

void
EarlyBranchResolver::rebuildCounts(ThreadID tid)
{
    for (int r = 0; r < MaxArchIntRegs; r++)
        pendingWrites[tid][r] = 0;

    for (const auto &entry : inFlight[tid])
        for (int r : entry.destIntRegs)
            pendingWrites[tid][r]++;
}

bool
EarlyBranchResolver::tryResolve(const DynInstPtr &inst, ThreadID tid,
                                 bool &taken)
{
    if (!canResolve(inst, tid))
        return false;

    taken = evaluateCondition(inst, tid);
    ++stats.resolvedAtFetch;

    DPRINTF(EarlyBranchResolver,
            "[tid:%i] [sn:%llu] EBR resolved %s -> %s\n",
            tid, inst->seqNum,
            inst->staticInst->getName(),
            taken ? "taken" : "not taken");

    return true;
}

bool
EarlyBranchResolver::canResolve(const DynInstPtr &inst, ThreadID tid)
{
    const StaticInstPtr &si = inst->staticInst;

    if (!si->isCondCtrl() || !si->isDirectCtrl()) {
        ++stats.fallbackNotCond;
        return false;
    }

    // RISC-V B-type branches have exactly 2 integer source registers.
    // Compressed branches (c.beqz/c.bnez) have 1 source vs zero.
    int srcs = si->numSrcRegs();
    if (srcs < 1)
        return false;

    for (int i = 0; i < srcs && i < 2; i++) {
        const RegId &reg = si->srcRegIdx(i);
        if (reg.classValue() != IntRegClass)
            return false;
        int idx = reg.index();
        // x0 is always ready (hardwired zero)
        if (idx != 0 && pendingWrites[tid][idx] > 0) {
            ++stats.fallbackBusy;
            return false;
        }
    }

    return true;
}

bool
EarlyBranchResolver::evaluateCondition(const DynInstPtr &inst,
                                        ThreadID tid) const
{
    const StaticInstPtr &si = inst->staticInst;
    const RegId &r1 = si->srcRegIdx(0);
    RegVal v1 = (r1.index() == 0) ? 0 : cpu->getArchReg(r1, tid);

    // Compressed branches compare rs1 against zero
    RegVal v2 = 0;
    if (si->numSrcRegs() >= 2) {
        const RegId &r2 = si->srcRegIdx(1);
        v2 = (r2.index() == 0) ? 0 : cpu->getArchReg(r2, tid);
    }

    const std::string &name = si->getName();

    if (name == "beq"   || name == "c.beqz") return v1 == v2;
    if (name == "bne"   || name == "c.bnez") return v1 != v2;
    if (name == "blt")  return (int64_t)v1 <  (int64_t)v2;
    if (name == "bge")  return (int64_t)v1 >= (int64_t)v2;
    if (name == "bltu") return v1 <  v2;
    if (name == "bgeu") return v1 >= v2;

    // Unknown branch type — can't resolve, treated as fallback
    return false;
}

EarlyBranchResolver::EBRStats::EBRStats(CPU *cpu)
    : statistics::Group(cpu, "earlyBranchResolver"),
      ADD_STAT(resolvedAtFetch, statistics::units::Count::get(),
               "Branches resolved early at fetch (source regs were ready)"),
      ADD_STAT(fallbackBusy, statistics::units::Count::get(),
               "Branches falling back to BPU: source reg had pending write"),
      ADD_STAT(fallbackNotCond, statistics::units::Count::get(),
               "Non-conditional or indirect branches skipped by EBR"),
      ADD_STAT(overrideTaken, statistics::units::Count::get(),
               "EBR overrode BPU from not-taken to taken"),
      ADD_STAT(overrideNotTaken, statistics::units::Count::get(),
               "EBR overrode BPU from taken to not-taken")
{}

} // namespace o3
} // namespace gem5
