function [DiceIndex, Precision, Recall] = evalResults(Seg,Ground)

SegMask = (Seg == 1);
GroundMask = (Ground == 1);

TP = nnz(SegMask & GroundMask);
PS = nnz(SegMask);
PG = nnz(GroundMask);

DiceIndex = (2 * TP) / (PS + PG);
Precision = TP/PS;
Recall = TP/PG;