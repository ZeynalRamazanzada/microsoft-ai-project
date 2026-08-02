# 🎤 BDFS — 10-Minute Presentation Speaker Script
### Behavioral Decision Fatigue Scoring
**Speakers:** Zeynal (Z) · Alperen (A)
**Total Runtime:** ~10 minutes

---

## ⏱️ SECTION 1 — TIMING PLAN TABLE

| # | Slide Title | Speaker | Duration | Cumulative | Purpose |
|---|---|---|---|---|---|
| 1 | Title Slide | Zeynal | 0:35 | 0:35 | Opening hook, introduce team |
| 2 | The Invisible Threat | Zeynal | 0:50 | 1:25 | Establish the problem |
| 3 | Why This Matters | Zeynal | 0:45 | 2:10 | Raise the stakes |
| 4 | The BDFS Solution | Zeynal | 0:40 | 2:50 | Hand off after solution intro |
| 5 | Dataset & Features | Alperen | 0:55 | 3:45 | Technical depth on data |
| 6 | Model Evaluation | Alperen | 1:00 | 4:45 | Results, why XGBoost won |
| 7 | System Pipeline | Alperen | 0:50 | 5:35 | Architecture walkthrough |
| 8 | Backend Integration | Alperen | 0:45 | 6:20 | FastAPI & API design |
| 9 | LIVE DEMO SLIDE | Alperen | 1:30 | 7:50 | Live browser demonstration |
| 10 | The User Interface | Alperen | 0:40 | 8:30 | UI walkthrough after demo |
| 11 | Results & SHAP | Zeynal | 0:45 | 9:15 | Key scientific finding |
| 12 | Challenges & Solutions | Zeynal | 0:30 | 9:45 | Credibility & honesty |
| 13 | Future Improvements | Zeynal | 0:20 | 10:05 | Vision, quick pass |
| 14 | Conclusion & Q&A | Both | 0:25 | ~10:30 | Wrap-up + thanks |

> **Note:** The live demo slide is the flexible buffer. If you're running short, extend the demo. If running long, keep it tight to the preset profile.

---

## 🎙️ SECTION 2 — FULL SPEAKER SCRIPT

---

### 🟦 SLIDE 1 — Title Slide
**Speaker: ZEYNAL**
**Duration: ~35 seconds**

> *Stand at the front, let the slide settle for 2 seconds before speaking.*

---

**What to say:**

"Good morning, everyone. My name is Zeynal, and this is my teammate Alperen. Today we're presenting BDFS — Behavioral Decision Fatigue Scoring.

Here's the question we started with: *what if you could tell that someone was mentally exhausted — just by watching how they make decisions?* No sensors, no surveys, no wearables. Just behavior.

That's exactly what BDFS does. Let's get into it."

**🔑 Key emphasis:** Pause after the question. Let it land. The word *"just behavior"* should be said slowly and clearly — it's your hook.

---

### 🟦 SLIDE 2 — The Invisible Threat
**Speaker: ZEYNAL**
**Duration: ~50 seconds**

**What to say:**

"So, what's the actual problem? Decision fatigue is real, it's measurable, and it's dangerous. Studies link a significant share of critical errors — in surgery, in aviation, in financial trading — to cognitive exhaustion that went undetected.

But here's the frustrating part. Look at the middle card: *zero* practical real-time tools exist that work passively and at scale. What we have today are EEG headsets, biometric patches, and end-of-shift questionnaires — all of which are either too expensive, too intrusive, or too slow to be useful in the moment.

Decision fatigue doesn't announce itself. And that's exactly why it's so dangerous."

**🔑 Key emphasis:** Point at the three stat cards as you reference each one. Pause on "zero practical tools" — let the audience absorb it.

---

### 🟦 SLIDE 3 — Why This Matters
**Speaker: ZEYNAL**
**Duration: ~45 seconds**

**What to say:**

"And this isn't a niche problem. Think about the four domains on this slide. A surgeon five hours into a procedure. A pilot handling a complex approach. A long-haul driver on a motorway. A judge hearing their thirtieth case of the day. All of these people may be visibly alert — but cognitively, they're running on fumes.

Research actually shows that judges approve significantly fewer parole requests as the afternoon goes on — not because the cases change, but because their decision quality degrades. That is the invisible tax of cognitive fatigue.

Catching it early doesn't just improve performance — it saves lives."

**🔑 Key emphasis:** Gesture to each of the four domain cards as you name them. Deliver the last sentence slowly and firmly.

---

### 🟦 SLIDE 4 — The BDFS Solution
**Speaker: ZEYNAL**
**Duration: ~40 seconds**

**What to say:**

"So how does BDFS solve this? Three core design principles. First: *non-intrusive* — our system reads behavioral signals only. No hardware required. Second: *real-time* — predictions return in under ten milliseconds through an async API. Third: *explainable* — every prediction comes with SHAP values that tell you exactly which behavioral signal triggered the alert.

On the right you can see the full pipeline — from the web interface, through our FastAPI backend, through the XGBoost model, all the way to the SHAP output on screen.

I'll now hand over to Alperen, who will walk you through the data and the technical architecture."

**🔑 Key emphasis:** Hold up three fingers as you say non-intrusive, real-time, explainable. This is a classic rhetoric move that registers with audiences.

> 🔄 **HANDOVER LINE:**
> *"Alperen, over to you."*

---

### 🟩 SLIDE 5 — Dataset & Feature Engineering
**Speaker: ALPEREN**
**Duration: ~55 seconds**

> *Zeynal steps slightly back. Alperen steps forward or takes the pointer.*

---

**What to say:**

"Thanks, Zeynal. Let's talk about the data that powers everything.

We built our training set from 150,000 behavioral trial records — representing 99,000 non-fatigued and 50,000 fatigued decision instances. Each record captures 19 engineered features across three conceptual groups.

The first group is *temporal features* — things like rolling choice inconsistency over the last five and ten trials. As you can see in the distribution plot on the right, these show dramatically different patterns between fatigued and non-fatigued subjects.

The second group is *cognitive features* — we applied the Drift Diffusion Model to decompose reaction times into psychological components: drift rate, decision boundary, and non-decision time.

The third is *behavioral context* — session position, task complexity, accuracy decay rate.

Together, these 19 features give us a rich, multi-dimensional fingerprint of cognitive state."

**🔑 Key emphasis:** Point to the distribution chart on the right. Specifically gesture to the rolling_incon_5 panel where the class separation is most visible.

---

### 🟩 SLIDE 6 — Model Evaluation
**Speaker: ALPEREN**
**Duration: ~60 seconds**

**What to say:**

"We trained and evaluated five different classifiers under identical conditions. The table here shows the results — and the winner is clear: XGBoost, with a ROC-AUC of 0.967 and an F1-Score of 0.866.

But I want to explain *why* XGBoost beat the others — because it's not just a number. Behavioral data is highly non-linear. Fatigue doesn't manifest as a straight line between features — it emerges from combinations and thresholds. Tree-based models naturally capture that. Logistic Regression, despite being surprisingly competitive at 0.948 AUC, couldn't model the more complex interaction patterns.

SVM and KNN? They struggled significantly. SVM reached only 0.771 AUC — nearly 20 points behind XGBoost. KNN was similarly weak. The ROC curve on the right shows this gap visually — the XGB and RF curves hug the top-left corner, while SVM and KNN trail well behind.

We chose XGBoost specifically because it balances precision and recall well — 91.6% precision means minimal false alarms, which is critical in a real deployment."

**🔑 Key emphasis:** Tap the table row for XGBoost when you say "the winner is clear." Refer to the ROC chart when you mention the visual gap.

---

### 🟩 SLIDE 7 — System Pipeline
**Speaker: ALPEREN**
**Duration: ~50 seconds**

**What to say:**

"Let me show you how the system actually works end-to-end — six clean stages.

A user opens the web interface and sets the 19 behavioral sliders — or loads a preset profile. On submission, those values are sent as a JSON payload via HTTP POST to our FastAPI backend.

Step three is the critical engineering decision: preprocessing. We serialize the StandardScaler *fitted on training data only* and load it at API startup. This is not a small detail — if you re-fit the scaler at inference time, your predictions will be meaningless. We learned that the hard way.

Step four: XGBoost inference. The model returns a class prediction and a probability score. Step five: SHAP computation — feature contributions are calculated and attached to the response. Step six: the UI renders the gauge, the verdict, and the top factors — all in under ten milliseconds total.

This box at the bottom is the key design note: every production prediction is mathematically identical to what we validated in training."

**🔑 Key emphasis:** Walk through the six cards sequentially with your hand or pointer. Slow down and emphasize the note about the scaler — it's a strong credibility signal.

---

### 🟩 SLIDE 8 — Robust Backend Integration
**Speaker: ALPEREN**
**Duration: ~45 seconds**

**What to say:**

"The backend is built on FastAPI — one of Python's fastest async frameworks. Everything runs asynchronously, so it can handle concurrent requests without queuing bottlenecks.

The API exposes five clean endpoints: a health check, a schema endpoint, a single prediction at /predict, and a batch endpoint at /predict/batch for processing multiple trials in one call. The Swagger UI on the right is auto-generated — this isn't just documentation, it's a live interactive testing interface we used during development.

Every input is validated through Pydantic models — if a field is missing or out of range, the request fails fast with a structured error before it ever reaches the model. And every prediction response includes SHAP contributions — so the API isn't just an inference engine, it's a full explainability service.

Alright — enough talking. Let me show you the thing actually running."

**🔑 Key emphasis:** Reference the Swagger screenshot directly. The last line is your transition into the demo — say it with a slight smile and natural energy shift.

---

### 🟩 SLIDE 9 — LIVE DEMO
**Speaker: ALPEREN**
**Duration: ~90 seconds**

> *Switch to browser. This slide stays on screen as a backdrop.*

---

*(See Section 4 below for the full LIVE DEMO SCRIPT.)*

---

### 🟩 SLIDE 10 — The User Interface
**Speaker: ALPEREN**
**Duration: ~40 seconds**

> *Return to slide deck after demo.*

---

**What to say:**

"So what you just saw is the full interface. Let me highlight a few design decisions that were deliberate.

The dark glassmorphic aesthetic isn't just visual style — it signals precision and focus. The Simple-to-Expert mode toggle means a non-technical operator can use it with zero training, while a researcher gets access to all 19 raw values and the DDM parameters.

The tooltip you see in the top-right screenshot? Every single slider has one. Plain-English explanations. No jargon. Because a system that only domain experts can use isn't actually useful in the field.

And notice the probability gauge animates in real time — every slider change triggers a live API call. There is zero batch processing delay."

**🔑 Key emphasis:** Point to the tooltip screenshot. Emphasize the live API call behavior — it visually impressed people during the demo.

> 🔄 **HANDOVER LINE:**
> *"Now I'll hand back to Zeynal to talk about what our SHAP analysis revealed — which was honestly the most surprising finding of this whole project."*

---

### 🟦 SLIDE 11 — Results & Explainability (SHAP)
**Speaker: ZEYNAL**
**Duration: ~45 seconds**

> *Zeynal steps forward again.*

---

**What to say:**

"Thank you, Alperen. And yes — this is the part that surprised us.

When we ran SHAP analysis, we expected the DDM cognitive parameters — drift rate, decision boundary — to dominate. These are derived from a well-validated psychological model. We thought they'd be the strongest signal.

But look at the chart. The top two features by SHAP importance are rolling_incon_5 and rolling_incon_10 — pure temporal inconsistency. How erratically someone has been choosing over their *last five and ten trials*. These beat every static cognitive metric.

And our ablation study confirmed it: removing all DDM features costs less than 0.1% AUC. But removing temporal features drops performance by over four points.

The insight? Fatigue doesn't show up in a single moment. It reveals itself through *accumulated erratic behavior over time*. That's the core scientific contribution of this work."

**🔑 Key emphasis:** Point to the SHAP chart and the top two features. The last sentence is your thesis — deliver it clearly and confidently.

---

### 🟦 SLIDE 12 — Challenges & Solutions
**Speaker: ZEYNAL**
**Duration: ~30 seconds**

**What to say:**

"No project goes smoothly, and we want to be transparent about that.

The biggest technical challenge was a data scaling mismatch. Early versions of our API re-fitted the scaler per request — completely breaking production predictions. The fix was straightforward once we understood it: serialize the fitted scaler at training time, load it once at startup.

The second challenge was UX. Nineteen raw feature variables including things like 'ez_drift_rate' and 'drift_boundary_ratio' are meaningless to most users. Our solution was the Simple/Expert toggle — abstracting complexity without hiding it.

These weren't just bugs. They were lessons in building a system that actually works outside the notebook."

**🔑 Key emphasis:** The last line is a confidence signal — say it with conviction. It shows you understand the gap between research and deployment.

---

### 🟦 SLIDE 13 — Future Improvements
**Speaker: ZEYNAL**
**Duration: ~20 seconds**

**What to say:**

"Quickly on the future — the short-term path is passive input integration: keyboard telemetry, mouse dynamics, webcam-based eye tracking. No new hardware required, just richer behavioral streams.

Medium-term: personalized baselines per user, and adaptive alert thresholds. Long-term: clinical pilots in real healthcare and aviation environments, with federated learning so we can improve the model without centralizing sensitive data.

The infrastructure is already there. It just needs better data."

**🔑 Key emphasis:** Keep this brisk and forward-looking. You're painting a vision, not defending a plan.

---

### 🟦 SLIDE 14 — Conclusion & Q&A
**Speaker: ZEYNAL + ALPEREN**
**Duration: ~25 seconds**

**What to say (ZEYNAL leads):**

"So to summarize: we built a complete, production-ready system that detects cognitive decision fatigue in real time — using behavioral data alone, with a ROC-AUC of 0.967 and sub-ten-millisecond latency.

The key scientific finding is that temporal behavioral inconsistency is a stronger fatigue signal than any static cognitive metric.

Before we take questions — we genuinely want to thank our advisor, Assistant Professor Dr. Denizhan Demirkol. His guidance shaped this project from day one, and we wouldn't be standing here with these results without him."

**ALPEREN adds:**

"Thank you all for your time. We're happy to take any questions."

**🔑 Key emphasis:** The thank-you to Dr. Demirkol should be warm and sincere — make eye contact with him if he's in the room. The final line from Alperen should be calm and confident, not rushed.

---

## 🔄 SECTION 3 — ALL TRANSITIONS

| # | From | To | Handover Line |
|---|---|---|---|
| 1 | Zeynal (Slide 4) | Alperen (Slide 5) | *"I'll now hand over to Alperen, who will walk you through the data and the technical architecture."* |
| 2 | Alperen (Slide 10) | Zeynal (Slide 11) | *"Now I'll hand back to Zeynal to talk about what our SHAP analysis revealed — which was honestly the most surprising finding of this whole project."* |
| 3 | Zeynal (Slide 13) | Both (Slide 14) | *(Zeynal speaks conclusion, Alperen closes with Q&A invitation — no explicit verbal handoff needed, natural double-close)* |

> 💡 **Tip:** When handing over, the incoming speaker should not start immediately. Take one breath-length pause. It signals control and composure to the audience.

---

## 🖥️ SECTION 4 — LIVE DEMO SCRIPT

**Speaker: ALPEREN**
**Duration: ~90 seconds**
**Starting point:** Slide 9 is on screen. Alperen opens the browser.

---

### Step 1 — Open the Application
> *Open browser to the BDFS web app.*

**Say:**
> *"Alright, this is the live application — running right now, connected to our FastAPI backend. You can see the model badge in the top right: XGBoost, F1 of 0.866, AUC of 0.967, and the green 'Connected' indicator — the API is live."*

---

### Step 2 — Load the Fatigued Profile
> *Click the "Load Fatigued Profile" button (orange emoji button, top left).*

**Say:**
> *"I'll start by loading our pre-built fatigued profile — this is a real example from our test set. Watch the gauge on the right."*

> *[Pause 1 second for the gauge animation to complete.]*

> *"87%. The system classified this as FATIGUED with moderate-to-high confidence. And look at the top contributing features below the gauge: Short Inconsistency, Long Inconsistency, and Decision Switching — exactly what our SHAP analysis predicted would dominate."*

---

### Step 3 — Show a Tooltip
> *Hover over the ⓘ icon next to "Reaction Speed" or "Short Inconsistency".*

**Say:**
> *"Each slider has a plain-English tooltip. This one reads: 'Average time to respond per trial — higher means slower reactions.' This is designed so someone with zero ML background can understand what they're looking at."*

---

### Step 4 — Manually Adjust a Slider
> *Switch to Expert mode using the toggle. Then slowly drag the "Short Inconsistency" (rolling_incon_5) slider from its current value down toward zero.*

**Say:**
> *"Now watch what happens when I reduce the inconsistency score — essentially simulating a more consistent decision-maker. The gauge drops in real time. Every slider movement is a live API call to our backend. There's no batching, no delay."*

> *[Move slider down. Pause as the gauge animates.]*

> *"Down to around 40%. Same session, same everything else — just more consistent choices. The system picks that up immediately."*

---

### Step 5 — Load the Alert Profile (Optional, if time allows)
> *Click the "Load Alert Profile" button (lightning bolt button, top).*

**Say:**
> *"And here's the opposite — an alert, non-fatigued profile. Gauge drops to single digits. System says not fatigued. This is the dynamic range the model is working with."*

---

### Step 6 — Close and Return to Slides
> *Switch back to the presentation.*

**Say:**
> *"That's BDFS in action. What you just saw — that full cycle from input to prediction to explanation — happens in under ten milliseconds on the server side. Let me go back to the slides for a moment."*

---

### ⚠️ BACKUP LINES — If Something Goes Wrong

| Problem | What to Say |
|---|---|
| API not responding / "Disconnected" badge | *"The live backend isn't responding — this happens sometimes in demo environments. I have a recorded walkthrough here — but let me show you the interface regardless, because the UI behavior is what matters most."* |
| Browser won't load | *"The browser's being uncooperative — so let me walk you through the screenshots on the next slide, which capture exactly what you'd see. The important thing is the architecture, which we've validated thoroughly."* |
| Slider values look wrong after loading | *"The values loaded slightly off — let me reset and reload the profile manually."* *(calmly click Reset, re-click Fatigued Profile)* |
| Gauge doesn't animate | *"The animation seems to have stalled — but the probability value updated. The backend is returning predictions correctly, this is a front-end render timing quirk."* |
| Someone asks a question mid-demo | *"Great question — let me finish this one demo step and I'll come back to that directly."* |

> 💡 **General backup rule:** Never apologize excessively. Acknowledge, pivot, move on. The audience follows your energy, not your technical glitches.

---

## 📌 FINAL CHECKLIST — Day of Presentation

- [ ] API backend is running before you walk in the room
- [ ] Browser is pre-opened to the app (not just the URL typed)
- [ ] "Load Fatigued Profile" tested once in the last 30 minutes
- [ ] Slide deck is in **Presenter View** so you can see notes without audience seeing them
- [ ] Both speakers know exactly which slide they're taking over on
- [ ] Zeynal has slides 1–4, 11–14 mentally rehearsed
- [ ] Alperen has slides 5–10 + demo mentally rehearsed
- [ ] You've both timed a full run-through at least once

---

*Alperen Sümeroğlu (231805023) · Zeynalabidin Ramazanzade (231805121)*
