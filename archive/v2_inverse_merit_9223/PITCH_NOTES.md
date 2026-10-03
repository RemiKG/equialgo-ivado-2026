# SOTA Overfitters ? V2 five-minute pitch

**0:00?0:35, slide 1.** Our first submission scored 94.63% accuracy and 94.40% macro F1, as reported from HxBuddy. We preserved that complete method and restarted the inference problem. The new CSV changes 194 decisions while keeping exactly 1,600 grants. Its official score is pending.

**0:35?1:15, slide 2.** Deleting geographic fields did not remove bias. Our first correction also inherited the committee?s work coefficient. That could be another way of retaining its priorities. We examined nonlinear income and work effects, but better agreement with committee decisions did not establish the true merit rule. We decided to model the uncertainty explicitly.

**1:15?2:05, slide 3.** We construct 2,048 merit hypotheses with different academic, work, need, first-generation and commute contributions. We weight them by compatibility with the aggregate result of the old submission. The supplied baseline opportunity gap is only a weak clue because we do not know its exact evaluation cohort. We average compatible hypotheses and select 1,600. No private reference labels are accessed.

**2:05?2:50, slide 4.** The primary assumes that omitted criteria explain more errors than random merit noise. A second model allows much more noise. For each, we sweep opportunity tolerances under a fixed budget. These are model-implied frontiers, not validated scores. Equal opportunity against independent merit remains unknown.

**2:50?3:30, slide 5.** The primary awards 970 grants to centre applicants and 630 to remote applicants. It replaces 97 recipients with 97 others. New recipients have higher average academic scores, lower income and fewer work hours. That demonstrates a real policy change, not proof that the hidden labels agree. The next portal result tests the hypothesis.

**3:30?4:20, slide 6.** One aggregate score cannot determine a fair rule. The low-noise prior and the support criteria can all be wrong. Before deployment, an independent panel must label a representative sample including rejected applicants. We require budget and schema controls, intersection monitoring, human appeals and a reviewed fallback. Inferred probabilities are not production-ready eligibility probabilities.

**4:20?5:00, slide 7.** We have made a testable change, preserved every previous prediction, and kept measured results separate from simulations. The repository contains the model, executed audit, code tests, pitch and governance plan. We will keep or revise the new hypothesis using actual scoring evidence, not an internally optimistic number.
