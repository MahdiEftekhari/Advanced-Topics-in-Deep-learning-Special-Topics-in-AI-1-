# SimSiam Paper Analysis and Comparison

## Objective
Read and dissect *Exploring Simple Siamese Representation Learning* (Chen & He), then write a 4–6 page report analyzing how SimSiam achieves competitive self-supervised representations without negative pairs, large batches, or a momentum encoder. Unlike the other projects in this repo, there's no code to implement — the deliverable is a written analysis, originally followed by a short in-person lecture where each student presented their report back to the instructor.

## Background / Paper
- **Paper:** Exploring Simple Siamese Representation Learning — Chen & He (CVPR 2021)
- **Focus:** Siamese-network self-supervised learning, and specifically what actually prevents representation collapse when so many of the usual safeguards (negative pairs, momentum encoders, clustering) are removed.
- **Originally assigned by:** Dr. Mahdi Eftekhari, Shahid Bahonar University of Kerman — the original handout gave a two-week window; that deadline doesn't apply here, this is an archived version of the assignment for self-study.

## Files in this folder
- `assignment.pdf` — original assignment handout (report structure, requirements, and grading criteria)

## How to attempt it
The report should cover five sections:

1. **Introduction** (~0.5 page) — why Siamese self-supervised learning matters, and why SimSiam's simplicity is notable.
2. **Self-supervised Siamese architectures** (~1–1.5 pages) — for each of SimCLR, BYOL, and SwAV: architecture, and the specific mechanism each uses to avoid collapse (negative pairs + contrastive loss for SimCLR, momentum encoder + predictor for BYOL, online clustering + Sinkhorn-Knopp for SwAV).
3. **The SimSiam approach** (~1.5–2 pages) — architecture (reproduce Figure 1 from the paper), the stop-gradient operation, and a full walkthrough of Algorithm 1: encoder, projection MLP, prediction MLP, negative cosine similarity loss, symmetrization, and how gradients actually flow.
4. **Comparative analysis** (~1–1.5 pages) — SimSiam as "SimCLR without negative pairs," "BYOL without the momentum encoder," and "SwAV without online clustering"; what the paper's ablations show about batch size, batch norm, and predictor design; stop-gradient vs. the implicit constraints in the other three methods.
5. **Discussion** (~0.5 page) — the alternating-optimization hypothesis for why stop-gradient prevents collapse, why the Siamese architecture underlies all four methods, simplicity/performance trade-offs, and open directions.

Use tables or diagrams where they clarify the differences between methods — the point is conceptual clarity, not reproducing benchmark numbers.

## Format & evaluation
No reference solution is included — this was a discussion-based assignment, and the "answer" was each student's own report and their ability to explain it live. Grading (per the original handout) weighed:
- technical accuracy and depth
- clarity of the algorithmic comparisons
- how well stop-gradient is explained
- quality of the cross-method comparative analysis
- critical thinking about *why* SimSiam works despite doing less than its predecessors
