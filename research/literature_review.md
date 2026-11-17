
# Brain Tumor Detection — Literature Review

## 1. Research Objective

The objective of this project is to study existing machine learning
and deep learning methods for brain tumor classification using MRI
images, investigate the limitations of existing approaches, and
develop an evidence-based methodology for improving model reliability.

The project will initially focus on classifying MRI images into
available tumor categories. The final task and methodology will be
confirmed after reviewing suitable datasets and recent research.

## 2. Literature Search Period

Primary search period: 2021–2026.

Earlier studies may be included when required to understand the
development of existing methods.

The review will prioritize original research papers, systematic
reviews, publicly documented datasets, and reproducible evaluation
methods.

---

## Paper 1 — Publication Verification Pending

### Title
AI in MRI Brain Tumor Diagnosis: A Systematic Review of
Machine Learning and Deep Learning Advances (2010–2025)

### Publication Details
- Reported year: 2025
- Reported journal: Chemometrics and Intelligent Laboratory Systems
- Reported DOI: 10.1016/j.chemolab.2025.105414
- Reported paper type: Systematic literature review
- DOI link: https://doi.org/10.1016/j.chemolab.2025.105414
- Verification status: Pending independent confirmation

### Research Objective
The previously drafted entry describes a review of machine learning
and deep learning methods for brain tumor analysis using MRI images.

### Methodology
The following details appeared in the initial draft but have not
yet been independently confirmed:
- Use of PRISMA review guidelines.
- Review of 155 studies.
- Coverage of research published between 2010 and 2025.
- Discussion of CNNs, SVMs, ensemble methods and Vision Transformers.
- Discussion of datasets such as BraTS and TCIA.

These details must not be treated as verified until the publication
record and full text are checked.

### Main Findings
No definitive findings are recorded here because the publication
details and full text have not yet been verified.

### Potential Research Relevance
A verified systematic review could help identify established
approaches, common evaluation practices, dataset limitations,
and research directions.

### Verification Requirement
Confirm the exact title, author list, journal, DOI, publication year,
methodology and findings before using this paper in a final report.

---

## Paper 2 — Vision Transformer Classification

### Title
A fine-tuned vision transformer based enhanced multi-class brain
tumor classification using MRI scan imagery

### Publication Details
- Authors: C. Kishor Kumar Reddy and colleagues
- Year: 2024
- Journal: Frontiers in Oncology
- Paper type: Original research
- DOI: 10.3389/fonc.2024.1400341
- Published: 18 July 2024
- Article: https://doi.org/10.3389/fonc.2024.1400341

### Research Objective
The study investigates fine-tuned Vision Transformer models for
classifying brain tumor MRI images and compares their performance
with established deep learning models.

### Dataset
The study reports using 7,023 MRI images across four categories:
- Glioma
- Meningioma
- Pituitary tumor
- No tumor

### Methodology
The authors evaluate several fine-tuned Vision Transformer models:
- FTVT-b16
- FTVT-b32
- FTVT-l16
- FTVT-l32

They compare these models with ResNet50, MobileNetV2 and
EfficientNet-B0.

### Evaluation Metrics
The paper reports:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

### Reported Results
The paper reports 98.70% accuracy for the FTVT-l16 model.
The reported accuracies for ResNet50, MobileNetV2 and
EfficientNet-B0 are 96.5%, 94.9% and 95.1%, respectively.

These are the results reported by the authors under their
experimental setup. They are not independently reproduced results.

### Limitations and Questions to Investigate
The following are evaluation questions for our review, not
confirmed defects in the paper:
- How well would the model perform on an independent MRI dataset?
- Were training and testing images split in a way that prevents
  related or duplicate images from appearing across both sets?
- How sensitive is performance to differences in image sources?
- How much computational cost is required to train and use the model?

### Research Opportunity
Evaluate a suitable baseline classifier on the original test split
and, where compatible data are available, on an independent dataset.
Compare the results using multiple evaluation metrics.

### Relevance to Our Project
This study provides a useful reference for Vision Transformer
classification and comparison with established CNN architectures.

---

## Paper 3 — Hybrid ViT-GRU and Explainable AI

### Title
Brain tumor detection and classification in MRI using hybrid ViT
and GRU model with explainable AI in Southern Bangladesh

### Publication Details
- Authors: Md. Mahfuz Ahmed and colleagues
- Year: 2024
- Journal: Scientific Reports
- Paper type: Original research
- DOI: 10.1038/s41598-024-71893-3
- Published: 1 October 2024
- Article: https://doi.org/10.1038/s41598-024-71893-3

### Research Objective
The study proposes a hybrid Vision Transformer and Gated Recurrent
Unit model for brain tumor classification and investigates the use
of explainable AI techniques to interpret predictions.

### Dataset
The authors report a hospital-collected dataset named BrTMHD-2023
containing 1,166 MRI scans from Bangabandhu Sheikh Mujib Medical
College Hospital in Faridpur, Bangladesh.

The study also reports an evaluation using an additional
brain tumor Kaggle dataset.

### Methodology
The proposed framework combines:
- Vision Transformer (ViT) feature extraction
- Gated Recurrent Unit (GRU)
- Different optimization methods, including SGD, Adam and AdamW
- Explainability methods, including attention maps, SHAP and LIME
- Ten-fold cross-validation on the primary dataset

### Evaluation Metrics
The paper reports:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- ROC-AUC analysis

### Reported Results
The authors report:
- 98.97% accuracy with the AdamW optimizer in their primary
  dataset experiments.
- 96.08% accuracy on the additional Kaggle dataset.
- Approximately 97% precision, recall and F1-score in the
  reported primary-dataset results.

These values belong to the paper's own experimental setup and
should not be directly compared with other papers without
considering dataset and evaluation differences.

### Limitations and Questions to Investigate
The following questions require further investigation:
- How representative is the hospital dataset?
- How does performance change across different MRI sources?
- Are the explanations stable across correct and incorrect predictions?
- How much computational complexity does the hybrid architecture add?
- Does the additional dataset evaluation adequately demonstrate
  generalization to unseen data?

These are questions for critical evaluation, not claims that the
authors failed to address them.

### Research Opportunity
Investigate whether a simpler classifier can provide competitive
performance while offering reliable confidence estimates and
transparent evaluation on independent data.

### Relevance to Our Project
This paper is relevant to cross-dataset evaluation, hybrid models
and explainability in MRI classification.

---

## 3. Cross-Paper Comparison

| Paper | Year | Approach | Dataset | Main investigation |
|---|---|---|---|---|
| Paper 1 | Reported 2025; unverified | Systematic review as described in the initial draft | Not confirmed | Verify bibliographic details and findings |
| Paper 2 | 2024 | Fine-tuned Vision Transformers | 7,023 MRI images, four classes | Independent-dataset performance and evaluation protocol |
| Paper 3 | 2024 | Hybrid ViT-GRU with explainable AI | 1,166 hospital MRI scans plus an additional Kaggle dataset | Cross-dataset performance, confidence and explanation reliability |

Important: The datasets, task definitions, splits and evaluation
procedures differ. Reported accuracy alone is not sufficient to
establish which method is best.

---

## 4. Candidate Research Questions

The following questions will guide further investigation:

1. How well does a brain tumor classifier generalize to an
   independent MRI dataset?

2. How much does model performance change when images come from
   different sources or acquisition conditions?

3. Can confidence calibration help identify predictions that
   may be unreliable?

4. Can a model abstain from making a confident prediction when
   its estimated uncertainty is high?

5. How do model complexity, classification performance and
   reliability compare across baseline architectures?

6. Can the experimental results be reproduced using documented
   datasets, preprocessing steps and evaluation procedures?

---

## 5. Candidate Research Gap

### Current Status
Candidate only — not yet established as novel.

### Proposed Direction
Evaluate the cross-dataset generalization and prediction reliability
of brain tumor MRI classifiers under distribution shift.

The project may investigate:
- Performance on a held-out test set.
- Performance on an independent dataset, when suitable data are available.
- Changes in accuracy, macro-F1, sensitivity and specificity.
- Confidence calibration and uncertainty estimation.
- The relationship between model confidence and prediction errors.
- The trade-off between model complexity and reliability.

### Why Further Research Is Needed
High accuracy on one dataset does not, by itself, establish reliable
performance on data from other sources.

However, cross-dataset generalization, explainability and uncertainty
are already active research areas. The proposed direction must
therefore be compared with recent work before claiming novelty.

### Criteria for Confirming the Gap
Before finalizing the research gap:
1. Review additional recent original research papers.
2. Record the datasets, model architectures and evaluation protocols.
3. Identify which limitations are explicitly acknowledged by authors.
4. Check whether closely related methods already address the proposed gap.
5. Define a specific, testable contribution supported by the evidence.

---

## 6. Preliminary Experimental Plan

### Phase 1 — Literature Review
Verify references and compare recent studies.

### Phase 2 — Dataset Selection
Choose a documented, appropriately licensed MRI dataset.
Inspect its class distribution, image formats, labels and split protocol.

### Phase 3 — Baseline Model
Train a manageable baseline classifier using the available
computational resources.

### Phase 4 — Reliable Evaluation
Measure accuracy, macro-precision, macro-recall, macro-F1 and
class-wise performance. Use an independent dataset only when
the labels and task definitions are compatible.

### Phase 5 — Confidence and Error Analysis
Investigate calibration and uncertainty estimation if the baseline
and dataset support these experiments.

### Phase 6 — Comparison and Reporting
Compare results against appropriate baselines using consistent
evaluation procedures. Document limitations and reproducibility details.

---

## 7. Reproducibility and Research Integrity

- Record dataset names, sources, licences and access dates.
- Keep training, validation and test data separate.
- Check for duplicate or closely related images across splits.
- Fit preprocessing operations using training data only when applicable.
- Fix and record random seeds where possible.
- Report multiple metrics instead of accuracy alone.
- Do not present literature-reported results as our own experiments.
- Do not claim novelty without reviewing related research.
- Document software versions, configuration and experiment settings.
- Do not publish private medical data or restricted dataset files.

---

## 8. References

1. Paper 1: Publication details are pending verification.
   Reported DOI in the initial draft:
   https://doi.org/10.1016/j.chemolab.2025.105414

2. Reddy, C. K. K., Reddy, P. A., Janapati, H., et al. (2024).
   A fine-tuned vision transformer based enhanced multi-class brain
   tumor classification using MRI scan imagery.
   Frontiers in Oncology, 14, 1400341.
   https://doi.org/10.3389/fonc.2024.1400341

3. Ahmed, M. M., Hossain, M. M., Islam, M. R., et al. (2024).
   Brain tumor detection and classification in MRI using hybrid
   ViT and GRU model with explainable AI in Southern Bangladesh.
   Scientific Reports, 14, 22797.
   https://doi.org/10.1038/s41598-024-71893-3

---

## 9. Final Research Status

- Project structure: Created.
- Python environment: Set up.
- Core ML libraries: Installed and import-tested.
- Literature review: Initial draft prepared.
- Paper 1: Bibliographic verification pending.
- Papers 2 and 3: Identified and summarized from publisher records.
- Research gap: Candidate only; further literature review required.
- Dataset: Not finalized.
- Model: Not selected.
- Experiments: Not started.

This project is an educational research prototype. It is not a
clinically validated system and must not be used to make medical
diagnoses or treatment decisions.
