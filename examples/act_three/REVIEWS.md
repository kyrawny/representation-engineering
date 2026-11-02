\# REVIEW 1

## Interesting use of ACT to steer model behaviour



Review:



I found it interesting that using ACT to steer the model led to responses that people rated as less sycophancy. Using sociological theory to decide how a model should respond in different situations and social roles is an interesting idea. The human evaluation is also a strength, as it examines how people actually perceive these responses.



However, steering method comes from existing RepE techniques. The main new part seems to be how ACT is used to set the steering targets, but the experiments do not clearly show what this adds over existing methods. The main comparisons are between unsteered, prompt-only, RepE-only, and hybrid conditions. Methods such as CAA and persona vectors would be useful baselines. Related work, such as Sycophancy Detection in Large Language Models, also deserves more discussion.



Another concern is that the RepE-only condition still uses ACT-derived targets, so it does not show how much ACT itself contributes. Comparing fixed EPA targets with ACT-derived targets while keeping the steering method unchanged would help answer this question. I would also like to see comparisons with existing anti-sycophancy methods in the human evaluation. Measuring appropriateness and naturalness alongside non-sycophancy would help show whether the proposed approach offers a better balance between these qualities.



Rating: 5: Marginally below acceptance threshold

Confidence: 3: The reviewer is fairly confident that the evaluation is correct



\# REVIEW 2



\## Interesting proposal with dubious empirical support due to the weaknesses highlighted below



Review:



Strengths



&#x20;   Interesting interdisciplinary approach of combining Affect Control Theory (ACT) with Representation Engineering (RepE) for socially grounded LLM steering.

&#x20;   Sufficient experiments and ablations



Weaknesses



&#x20;   Primarily integrates existing ACT and RepE methods without substantial technical innovation.

&#x20;   The same EPA reading framework informs steering and evaluates its effectiveness, potentially biasing results toward the proposed method.

&#x20;   Limited generalizability as evaluation focuses on predefined social identities and single-turn interactions, without demonstrating effectiveness in realistic multi-turn settings.

&#x20;   Most importantly, it is unclear whether the selected evaluation metrics meaningfully capture genuine affective alignment, social appropriateness, or sycophancy reduction. The motivation for the metric choices and experimental design, including the selection of scenarios and social identities, is insufficiently justified.

&#x20;   Evaluation appears somewhat ad hoc, raising concerns about the validity of the findings. Lack of comparison / reference to existing work in this space.



Questions



&#x20;   How does the proposed approach distinguish between reducing substantive sycophancy and simply generating more assertive responses?

&#x20;   Why does the hybrid method achieve the best automated performance but the worst human-rated appropriateness and naturalness?

&#x20;   How can the authors rule out evaluation bias when the same EPA framework is used for steering and assessment?

&#x20;   What additional benefits does ACT provide compared with simpler prompting or conventional representation-steering baselines?

&#x20;   How applicable is the framework to realistic, multi-turn interactions involving uncertain identities and culturally diverse social expectations?

&#x20;   What informed your metric choices and experimental setting ? What are the other relevant work in this space and how does your method compare to theirs ?





Rating: 3: Clear rejection

Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct



\# REVIEW 3



\## Review comments



Review:



This paper uses Affect Control Theory to compute socially appropriate affective targets, and steers LLM responses toward them via representation engineering, with the aim of reducing sycophancy. Grounding steering targets in an established sociological theory is original and relevant to human-AI interaction.



&#x20;   The automated evaluation is circular: the same EPA reader defines the targets and scores the outputs. The reader itself is noisy, and the gains are modest.



&#x20;   The human study does not clearly support the central claim. Unsteered responses are rated most appropriate, while the hybrid condition, best on the automated metric, is rated worst on both appropriateness and naturalness.



&#x20;   Sycophancy is measured with a single Likert item rather than standard sycophancy benchmarks.



&#x20;   The statistical tests treat repeated ratings from 48 participants as independent samples; a mixed-effects analysis is needed.





Rating: 5: Marginally below acceptance threshold

Confidence: 3: The reviewer is fairly confident that the evaluation is correct

