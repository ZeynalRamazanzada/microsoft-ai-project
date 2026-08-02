# Comprehensive Literature Review
## Systematic Literature Review of Decision Fatigue, DDM, and ML for Cognitive Load
### A Systematic Review of Behavioral Decision Fatigue Scoring (BDFS)

**Project:** BDFS — A Behavioral Decision Fatigue Scoring Framework Using Sensor-Free Multi-Signal Machine Learning on Synthetic Sequential Choice Data  
**Prepared for:** Academic Evaluation (Graduate/Master's Level)  
**Date:** June 2026  
**Team:** BDFS Research Team  

---

## Table of Contents

1. [Phase 1 — Project Analysis](#phase-1--project-analysis)
2. [Phase 2 — Literature Review Strategy](#phase-2--literature-review-strategy)
3. [Phase 3 — Critical Literature Analysis](#phase-3--critical-literature-analysis)
4. [Phase 4/5 — Detailed Source Cards](#phase-45--detailed-source-cards)
5. [Phase 6 — Final Deliverables](#phase-6--final-deliverables)
   - 6.1 [Professionally Written Literature Review](#61-professionally-written-literature-review)
   - 6.2 [Complete References — APA 7](#62-complete-references--apa-7)
   - 6.3 [Categorized Summary of Literature](#63-categorized-summary-of-literature)
   - 6.4 [Research Gaps Identified](#64-research-gaps-identified)
   - 6.5 [Suggestions for Novel Contributions](#65-suggestions-for-novel-contributions)
   - 6.6 [Recommendations for Future Research](#66-recommendations-for-future-research)

---

# Phase 1 — Project Analysis

## 1.1 Main Research Problem

The central research problem addresses a fundamental gap in the intersection of cognitive science and machine learning: **Can cognitive fatigue states during sequential decision-making be reliably predicted from purely behavioral signals — without physiological sensors — using classical machine learning models enhanced by computationally derived latent cognitive features?**

Decision fatigue is a well-documented psychological phenomenon. However, existing detection methods are bifurcated into two isolated paradigms: (1) sensor-dependent approaches using EEG, fNIRS, or ECG with deep learning, and (2) observational behavioral studies in social science that lack computational modeling. The BDFS project explicitly targets this unoccupied methodological space.

## 1.2 Academic Disciplines

| Discipline | Relevance |
|:---|:---|
| **Cognitive Psychology** | Decision fatigue theory, ego depletion, choice overload |
| **Mathematical Psychology** | Drift Diffusion Model (DDM), evidence accumulation |
| **Machine Learning / Data Science** | Classification algorithms, feature engineering, model evaluation |
| **Explainable AI (XAI)** | SHAP values, model interpretability |
| **Behavioral Science** | Observable behavioral markers, reaction time analysis |
| **Computational Cognitive Science** | EZ-diffusion parameter estimation, synthetic data generation |

## 1.3 Related Theories, Technologies, and Frameworks

### Theoretical Frameworks
- **Ego Depletion / Strength Model of Self-Control** (Baumeister et al., 1998)
- **Process Model of Depletion** (Inzlicht & Schmeichel, 2012)
- **Choice Overload Hypothesis** (Iyengar & Lepper, 2000)
- **Sequential Sampling / Evidence Accumulation** (Ratcliff & McKoon, 2008)

### Computational Frameworks
- **Drift Diffusion Model (DDM)**: Latent parameters (drift rate *v*, boundary separation *a*, non-decision time *T*_er)
- **EZ-Diffusion Model** (Wagenmakers et al., 2007)
- **Ornstein-Uhlenbeck Process**: Stochastic mean-reverting process

### Machine Learning Methodologies
- **Ensemble Methods**: Random Forest, XGBoost
- **SMOTE**: Synthetic Minority Over-sampling Technique
- **SHAP**: SHapley Additive exPlanations

## 1.4 Potential Research Questions

1. To what extent can cognitive fatigue during sequential decision-making be predicted from observable behavioral patterns without physiological sensors?
2. Do latent features derived from the EZ-Diffusion Model provide significant additional predictive power compared to raw behavioral features alone?

## 1.5 Innovative/Novel Aspects

- **Sensor-free cognitive state detection**: Eliminates the hardware barrier that limits current cognitive load monitoring systems.
- **DDM-derived latent features as ML inputs**: Bridges computational cognitive science and applied ML.
- **Systematic ablation design**: Quantifies the marginal contribution of DDM features, temporal features, and behavioral baselines.
- **Controlled synthetic data with ground truth**: Enables precise evaluation of detection capabilities.

## 1.6 Keywords and Search Strategy

### Main Keywords
`decision fatigue`, `ego depletion`, `cognitive load detection`, `drift diffusion model`, `EZ-diffusion`, `mental workload classification`, `sensor-free fatigue detection`

### Alternative Academic Keywords
`choice overload`, `sequential decision making`, `cognitive depletion`, `mental fatigue machine learning`, `reaction time fatigue`, `digital biomarker fatigue`, `explainable AI cognitive`

### Boolean Search Combinations

```
("decision fatigue" OR "ego depletion") AND ("machine learning" OR "classification")

("drift diffusion model" OR "EZ-diffusion") AND ("fatigue" OR "cognitive load" OR "sleep deprivation")

("cognitive load" OR "mental workload") AND ("detection" OR "classification") AND ("sensor-free" OR "behavioral" OR "without sensors")

("reaction time" OR "response time") AND ("fatigue" OR "cognitive decline") AND ("prediction" OR "machine learning")
```

---

# Phase 2 — Literature Review Strategy

## 2.1 Database Search Plan

| Database | Focus Areas | Expected Yield |
|:---|:---|:---|
| **Google Scholar** | Broad cross-disciplinary search; citation verification | High — primary discovery tool |
| **IEEE Xplore** | Sensor-based ML, signal processing | High — engineering literature |
| **ACM Digital Library** | HCI, behavioral signals, CHI proceedings | Medium-High |
| **PubMed / PsycINFO** | Psychology, neuroscience, cognitive modeling | High — theoretical foundation |
| **Springer** | Machine Learning journal, cognitive science | Medium |
| **ScienceDirect (Elsevier)** | Neuroscience & Biobehavioral Reviews | High |
| **Nature** | Nature Machine Intelligence (SHAP) | Medium |

## 2.2 Inclusion Criteria

- Peer-reviewed journal articles or top-tier conference papers.
- Published in Q1/Q2 journals (verified via Scopus/SJR).
- High citation count relative to age (minimum ~50 citations for papers >3 years old).
- Direct relevance to decision fatigue, DDM, ML for cognitive load, or ML methodology (SMOTE/SHAP).

## 2.3 Exclusion Criteria

- Non-peer-reviewed sources (blogs, white papers, opinion pieces).
- Studies with insufficient methodological rigor.

---

# Phase 3 — Critical Literature Analysis

## 3.1 Thematic Analysis: Decision Fatigue and Ego Depletion

The concept of decision fatigue is rooted in Baumeister et al.'s (1998) strength model of self-control, which posits that self-regulation draws upon a finite, depletable resource. Vohs et al. (2008) extended this by demonstrating that the act of making choices itself is depleting. The real-world applicability was dramatically illustrated by Danziger et al. (2011) in their analysis of judicial parole decisions.

**Contradictory findings.** The theoretical foundations were significantly challenged by Hagger et al. (2016), a pre-registered multilab replication that failed to find evidence for the ego depletion effect. In response, Inzlicht and Schmeichel (2012) proposed an alternative process model, arguing that depletion reflects shifts in motivation and attention rather than resource exhaustion. 

**BDFS Implication.** BDFS is positioned to be agnostic to the underlying mechanism — it detects behavioral patterns associated with fatigue regardless of whether the cause is resource depletion or motivational shift.

## 3.2 Thematic Analysis: Drift Diffusion Models

The Drift Diffusion Model (DDM) provides the computational cognitive framework that distinguishes BDFS from purely empirical behavioral approaches. Ratcliff and McKoon (2008) provided the definitive formalization of the DDM for two-choice tasks. Wagenmakers et al. (2007) introduced the EZ-Diffusion Model, a simplified approach to estimating DDM parameters from summary statistics, making it practical for ML feature engineering.

The direct connection between DDM parameters and fatigue was established by Ratcliff and Van Dongen (2009, 2011), who demonstrated that fatigue decreases drift rate and increases non-decision time and variability. This empirically validates that DDM parameters capture distinct dimensions of fatigue.

## 3.3 Thematic Analysis: Machine Learning for Cognitive Load Detection

The dominant approach to cognitive load detection relies on physiological signals (Borghini et al., 2014; Dolmans et al., 2021). However, emerging sensor-free approaches demonstrate the feasibility of detecting cognitive states from behavioral signals alone. Fridman et al. (2018) achieved reasonable classification from camera-derived features, and Acien et al. (2022) demonstrated keystroke dynamics as a digital biomarker for mental fatigue. 

**Research Gap.** No existing study combines DDM-derived latent features with classical ML models for sensor-free cognitive fatigue detection. BDFS uniquely bridges these approaches.

## 3.4 Thematic Analysis: Machine Learning Methodology

The ML algorithm choices in BDFS are grounded in well-established foundations, notably Random Forest (Breiman, 2001) and XGBoost (Chen & Guestrin, 2016). Class imbalance is addressed through SMOTE (Chawla et al., 2002). Model interpretability is achieved through SHAP (Lundberg & Lee, 2017) and TreeSHAP (Lundberg et al., 2020), which provide exact, theoretically grounded feature importance.

---

# Phase 4/5 — Detailed Source Cards

> [!NOTE]
> The following source cards provide detailed information for the most important references used in this literature review. Each card includes title, authors, publication details, methodology, findings, strengths, limitations, project relevance, and citation information.

---

### Source Card 1: Baumeister et al. (1998)

| Field | Detail |
|:---|:---|
| **Title** | Ego depletion: Is the active self a limited resource? |
| **Authors** | Roy F. Baumeister, Ellen Bratslavsky, Mark Muraven, Dianne M. Tice |
| **Year** | 1998 |
| **Publication** | *Journal of Personality and Social Psychology*, 74(5), 1252–1265 |
| **Citations** | ~9,200+ |
| **Research Objective** | Test whether self-regulation draws upon a limited, depletable resource |
| **Methodology** | Series of laboratory experiments using sequential-task paradigm |
| **Main Findings** | Self-regulation depletes a finite resource; subsequent task performance is impaired |
| **Strengths** | Foundational study; highly influential |
| **Limitations** | Challenged by recent replication failures |
| **Relation to Project** | Provides core theoretical rationale that decision-making depletes cognitive resources |
| **DOI** | https://doi.org/10.1037/0022-3514.74.5.1252 |
| **APA 7 Citation** | Baumeister, R. F., Bratslavsky, E., Muraven, M., & Tice, D. M. (1998). Ego depletion: Is the active self a limited resource? *Journal of Personality and Social Psychology*, *74*(5), 1252–1265. https://doi.org/10.1037/0022-3514.74.5.1252 |

---

### Source Card 2: Vohs et al. (2008)

| Field | Detail |
|:---|:---|
| **Title** | Making choices impairs subsequent self-control: A limited-resource account of decision making, self-regulation, and active initiative |
| **Authors** | Kathleen D. Vohs, Roy F. Baumeister, Brandon J. Schmeichel, Jean M. Twenge, Noelle M. Nelson, Dianne M. Tice |
| **Year** | 2008 |
| **Publication** | *Journal of Personality and Social Psychology*, 94(5), 883–898 |
| **Citations** | ~1,900+ |
| **Research Objective** | Demonstrate that decision-making itself depletes self-regulatory resources |
| **Methodology** | Multiple experiments measuring self-control after choice-making tasks |
| **Main Findings** | The act of choosing depletes the same resource used for self-control |
| **Strengths** | Directly links decision-making to depletion |
| **Limitations** | Laboratory paradigm |
| **Relation to Project** | Validates that sequential decision count is a primary input for fatigue scoring |
| **DOI** | https://doi.org/10.1037/0022-3514.94.5.883 |
| **APA 7 Citation** | Vohs, K. D., Baumeister, R. F., Schmeichel, B. J., Twenge, J. M., Nelson, N. M., & Tice, D. M. (2008). Making choices impairs subsequent self-control: A limited-resource account of decision making, self-regulation, and active initiative. *Journal of Personality and Social Psychology*, *94*(5), 883–898. https://doi.org/10.1037/0022-3514.94.5.883 |

---

### Source Card 3: Ratcliff & McKoon (2008)

| Field | Detail |
|:---|:---|
| **Title** | The diffusion decision model: Theory and data for two-choice decision tasks |
| **Authors** | Roger Ratcliff, Gail McKoon |
| **Year** | 2008 |
| **Publication** | *Neural Computation*, 20(4), 873–922 |
| **Citations** | ~4,000+ |
| **Research Objective** | Formalize the DDM for two-choice RT tasks |
| **Methodology** | Mathematical modeling with parameter estimation across diverse cognitive tasks |
| **Main Findings** | DDM accounts for full RT distributions, accuracy, and speed-accuracy tradeoffs |
| **Strengths** | Definitive formalization; comprehensive demonstration |
| **Limitations** | Computationally expensive full parameter estimation |
| **Relation to Project** | Defines the complete DDM parameter space that BDFS extracts as ML features |
| **DOI** | https://doi.org/10.1162/neco.2008.12-06-420 |
| **APA 7 Citation** | Ratcliff, R., & McKoon, G. (2008). The diffusion decision model: Theory and data for two-choice decision tasks. *Neural Computation*, *20*(4), 873–922. https://doi.org/10.1162/neco.2008.12-06-420 |

---

### Source Card 4: Wagenmakers et al. (2007)

| Field | Detail |
|:---|:---|
| **Title** | An EZ-diffusion model for response time and accuracy |
| **Authors** | Eric-Jan Wagenmakers, Han L. J. van der Maas, Raoul P. P. P. Grasman |
| **Year** | 2007 |
| **Publication** | *Psychonomic Bulletin & Review*, 14(1), 3–22 |
| **Citations** | ~2,500+ |
| **Research Objective** | Propose a simplified method-of-moments DDM estimation approach |
| **Methodology** | Closed-form equations estimating v, a, T_er from mean RT, RT variance, and accuracy |
| **Main Findings** | EZ-diffusion provides rapid, accessible parameter estimation |
| **Strengths** | Computational simplicity; requires only summary statistics |
| **Limitations** | Less precise than full DDM |
| **Relation to Project** | Direct methodological foundation: BDFS uses EZ-diffusion for feature extraction |
| **DOI** | https://doi.org/10.3758/BF03194023 |
| **APA 7 Citation** | Wagenmakers, E.-J., van der Maas, H. L. J., & Grasman, R. P. P. P. (2007). An EZ-diffusion model for response time and accuracy. *Psychonomic Bulletin & Review*, *14*(1), 3–22. https://doi.org/10.3758/BF03194023 |

---

### Source Card 5: Acien et al. (2022)

| Field | Detail |
|:---|:---|
| **Title** | Detection of mental fatigue in the general population: Feasibility study of keystroke dynamics as a real-world biomarker |
| **Authors** | Alejandro Acien, Aythami Morales, Ruben Vera-Rodriguez, Julian Fierrez, Ijah Mondesire-Crump, Teresa Arroyo-Gallego |
| **Year** | 2022 |
| **Publication** | *JMIR Biomedical Engineering* |
| **Citations** | Emerging |
| **Research Objective** | Demonstrate keystroke dynamics as a non-intrusive digital biomarker for mental fatigue |
| **Methodology** | Keystroke feature extraction with SVM and Random Forest |
| **Main Findings** | 70–90% accuracy for fatigue detection from everyday typing patterns |
| **Strengths** | Fully sensor-free; ecologically valid |
| **Limitations** | Keystroke-specific |
| **Relation to Project** | Validates sensor-free ML fatigue detection from behavioral timing signals |
| **DOI** | https://doi.org/10.2196/41003 |
| **APA 7 Citation** | Acien, A., Morales, A., Vera-Rodriguez, R., Fierrez, J., Mondesire-Crump, I., & Arroyo-Gallego, T. (2022). Detection of mental fatigue in the general population: Feasibility study of keystroke dynamics as a real-world biomarker. *JMIR Biomedical Engineering*. https://doi.org/10.2196/41003 |

---

### Source Card 6: Breiman (2001)

| Field | Detail |
|:---|:---|
| **Title** | Random Forests |
| **Authors** | Leo Breiman |
| **Year** | 2001 |
| **Publication** | *Machine Learning*, 45(1), 5–32 |
| **Citations** | ~187,000+ |
| **Research Objective** | Introduce the Random Forest ensemble learning algorithm |
| **Methodology** | Bootstrap-aggregated decision trees with random feature subsets |
| **Main Findings** | RF achieves competitive accuracy with noise robustness |
| **Strengths** | Foundational for ensemble ML |
| **Limitations** | Variable importance biased toward high-cardinality features |
| **Relation to Project** | RF is a core classifier in the BDFS pipeline |
| **DOI** | https://doi.org/10.1023/A:1010933404324 |
| **APA 7 Citation** | Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/A:1010933404324 |

---

### Source Card 7: Chawla et al. (2002)

| Field | Detail |
|:---|:---|
| **Title** | SMOTE: Synthetic minority over-sampling technique |
| **Authors** | Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, W. Philip Kegelmeyer |
| **Year** | 2002 |
| **Publication** | *Journal of Artificial Intelligence Research*, 16, 321–357 |
| **Citations** | ~30,000+ |
| **Research Objective** | Address class imbalance through synthetic minority oversampling |
| **Methodology** | Feature-space interpolation between minority samples |
| **Main Findings** | SMOTE improves minority class detection |
| **Strengths** | Foundational work |
| **Limitations** | Can generate noisy samples |
| **Relation to Project** | BDFS uses SMOTE for training set balancing |
| **DOI** | https://doi.org/10.1613/jair.953 |
| **APA 7 Citation** | Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, *16*, 321–357. https://doi.org/10.1613/jair.953 |

---

### Source Card 8: Lundberg & Lee (2017)

| Field | Detail |
|:---|:---|
| **Title** | A unified approach to interpreting model predictions |
| **Authors** | Scott M. Lundberg, Su-In Lee |
| **Year** | 2017 |
| **Publication** | *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4768–4777 |
| **Citations** | ~55,000+ |
| **Research Objective** | Unify feature attribution methods under game-theoretic Shapley value framework |
| **Methodology** | SHAP framework connecting various methods |
| **Main Findings** | SHAP provides theoretically grounded feature importance |
| **Strengths** | Theoretical elegance; unifying framework |
| **Limitations** | Computationally expensive for large models without optimizations |
| **Relation to Project** | Core interpretability method for BDFS models |
| **DOI** | https://doi.org/10.48550/arXiv.1705.07874 |
| **APA 7 Citation** | Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 4768–4777). |

---

# Phase 6 — Final Deliverables

## 6.1 Professionally Written Literature Review

*(Excerpt of Synthesis)*
The integration of Drift Diffusion Models with applied Machine Learning represents a frontier in cognitive state detection. While physiological sensor-based approaches have dominated mental workload classification, their ecological validity is limited by hardware constraints. BDFS circumvents this by utilizing behavioral signals — reaction times and choice patterns — augmented by DDM parameters derived via the EZ-Diffusion model. This approach is theoretically grounded in cognitive psychology yet pragmatically aligned with scalable ML classification techniques like Random Forest and XGBoost. The interpretability provided by SHAP further ensures that the resulting models are not black boxes, but rather diagnostic tools capable of revealing the cognitive markers of fatigue.

## 6.2 Complete References — APA 7

Acien, A., Morales, A., Vera-Rodriguez, R., Fierrez, J., Mondesire-Crump, I., & Arroyo-Gallego, T. (2022). Detection of mental fatigue in the general population: Feasibility study of keystroke dynamics as a real-world biomarker. *JMIR Biomedical Engineering*. https://doi.org/10.2196/41003

Baumeister, R. F., Bratslavsky, E., Muraven, M., & Tice, D. M. (1998). Ego depletion: Is the active self a limited resource? *Journal of Personality and Social Psychology*, *74*(5), 1252–1265. https://doi.org/10.1037/0022-3514.74.5.1252

Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, *16*, 321–357. https://doi.org/10.1613/jair.953

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM.

Danziger, S., Levav, J., & Avnaim-Pesso, L. (2011). Extraneous factors in judicial decisions. *Proceedings of the National Academy of Sciences*, *108*(17), 6889–6892.

Hagger, M. S., et al. (2016). A multilab preregistered replication of the ego-depletion effect. *Perspectives on Psychological Science*, *11*(4), 546–573.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* (Vol. 30).

Ratcliff, R., & McKoon, G. (2008). The diffusion decision model: Theory and data for two-choice decision tasks. *Neural Computation*, *20*(4), 873–922.

Wagenmakers, E.-J., van der Maas, H. L. J., & Grasman, R. P. P. P. (2007). An EZ-diffusion model for response time and accuracy. *Psychonomic Bulletin & Review*, *14*(1), 3–22.

*(Note: Full 25 references provided in the RIS file)*

## 6.3 Categorized Summary of Literature

| Category | Key Papers | Primary Insight |
|:---|:---|:---|
| **Decision Fatigue Theory** | Baumeister (1998), Vohs (2008) | Sequential decisions degrade cognitive resources |
| **Drift Diffusion Models** | Ratcliff & McKoon (2008), Wagenmakers (2007) | DDM parameters serve as cognitive state indicators |
| **Sensor-Free ML** | Acien et al. (2022), Fridman et al. (2018) | Behavioral signals can classify fatigue without sensors |
| **ML Methodology** | Breiman (2001), Chawla (2002), Lundberg (2017) | Ensembles + SMOTE + SHAP form robust classification pipelines |

## 6.4 Research Gaps Identified

> [!WARNING]
> ### Gap 1: No DDM-ML Integration
> No existing study uses DDM-derived parameters as engineered features within an ML classification pipeline for cognitive fatigue detection.

> [!IMPORTANT]
> ### Gap 2: Lack of Ablation Studies in Fatigue ML
> Existing fatigue detection studies rarely perform systematic ablation to quantify the contribution of individual feature groups.

## 6.5 Suggestions for Novel Contributions

1. **First DDM-ML fatigue detection framework**: Bridging cognitive science and applied ML.
2. **Systematic ablation quantification**: Rigorous evidence for DDM feature value.
3. **Theory-agnostic detection**: Capturing fatigue patterns independent of the resource vs. process model debate.

## 6.6 Recommendations for Future Research

1. **Real-world validation**: Deploy BDFS on real human behavioral data collected through web-based decision tasks.
2. **Online/streaming detection**: Extend the framework to real-time fatigue monitoring using sliding windows.
3. **Deep learning sequential models**: Process the sequence as a time series using LSTM or Transformer architectures.

---
*End of Literature Review*
