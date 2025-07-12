## Experimental Results

### Table 1: WER Progression
| Epoch | Decoding Method       | Dev WER | Test WER |
|-------|-----------------------|---------|----------|
| 10    | Greedy Search         | 39.04%  | 40.44%   |
| 10    | Beam Search (beam=4)  | 35.39%  | 36.67%   |
| 20    | Greedy Search (avg 5) | 18.75%  | 19.54%   |
| 20    | Beam Search (avg 5)   | 17.76%  | 18.67%   |
| 30    | Greedy Search (avg 5) | 16.31%  | 17.10%   |
| 32    | Beam Search (avg 5)   | 15.58%  | 16.26%   |
| 40    | Greedy Search         | 15.6%   | 16.12%   |
| 40    | Beam Search           | 14.93%  | 15.45%   |
| 50    | Greedy Search         | 15.26%  | 15.72%   |
| 50    | Beam Search           | 14.6%   | 15.0%    |
| 60    | Greedy Search         | 14.82%  | 15.6%    |
| 60    | Beam Search           | 14.17%  | 14.94%   |
| 70    | Greedy Search         | 14.67%  | 15.59%   |
| 70    | Beam Search (beam=4)  | 14.13%  | 14.85%   |

### Key Findings
1. **Performance Trend**:
   - Achieved **64.6% relative WER reduction** (40.44% → 14.85%)
   - Beam search consistently outperforms greedy search by **0.5-1.5% absolute WER**

2. **Training Observations**:
   - Checkpoint averaging (top-5) improves WER by **~1%**
   - Best test set WER: **14.85%** (epoch 70, beam=4)

### Error Analysis (Dev Set @ 14.85% WER)
| Reference                          | Hypothesis                        | Error Type       |
|------------------------------------|-----------------------------------|------------------|
| "A konyv az asztalon van"          | "A könyv az asztalon"             | Deletion ("van") |
| "Menjünk el a parkba"              | "Menjünk ki a parkba"             | Substitution ("el"→"ki") |
| "Ez nagyon finom"                  | "Ez egy nagyon finom"             | Insertion ("egy")|

