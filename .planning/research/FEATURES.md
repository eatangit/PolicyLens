# Feature Landscape: PolicyLens

**Domain:** AI-powered energy/environmental policy impact evaluator
**Target users:** Politicians, general public, journalists, think tank analysts
**Researched:** 2026-04-18
**Confidence:** HIGH (EPS/NREL feature research verified; visualization best-practice research MEDIUM via multi-source cross-check)

---

## Table Stakes

Features users expect from a policy analysis tool. Missing any of these and the product feels unfinished or untrustworthy.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Plain-English Policy Input** | The entire value proposition. Without it you are just a dashboard. Users (politicians, public) cannot configure sliders like EPS requires. | Medium | Claude parses → structured action vector. Input field + submit button. |
| **Baseline ("Status Quo") State** | Users need a reference point to judge impact. "How much CO2 does the US emit today?" must be answered before projections mean anything. | Low | Pull from EIA; display as a labeled fact card before results. |
| **Projected State After Policy** | The core output. What changes, and by how much, if this policy is enacted? | High | GRU or linear fallback trajectory over 10 steps (years). |
| **Environmental Metrics (energy-scoped)** | CO2 emissions, renewable energy share, fossil fuel consumption. These are the EIA-native metrics and are expected for an energy policy tool. | Low | EIA routes already identified in STACK.md. Do not expand to air quality, water, land use — out of scope. |
| **Economic Metrics (two or three only)** | Users expect to see the cost side. Minimal set: energy cost to consumers ($/MWh or equivalent), government revenue/spending change, jobs in energy sector. More than three metrics overwhelms non-experts. | Medium | Derive from state-vector changes scaled to dollar values via EIA price data. |
| **Time Horizon Display** | Show results over time (years 1–10), not just a single endpoint. Policy effects compound; showing only Year 10 hides the trajectory. | Medium | Line chart with year on x-axis. |
| **Before/After Comparison** | Users need to see the delta, not just the projected value in isolation. "CO2 drops from 5,200 Mt to 4,100 Mt" is comprehensible; "4,100 Mt" alone is not. | Low | Side-by-side or overlaid baseline vs. projected line. |
| **Policy Summary Card** | What did the system understand about this policy? Non-experts need to verify the AI parsed their input correctly before they trust the output. | Low | Display Claude's `reasoning` field from the structured output. Critical for trust. |
| **Fallback Mode Disclosure** | At demo time the GRU is untrained. The UI must be honest: "This projection uses a simplified model. Results are illustrative." Hiding this is a trust liability. | Low | Banner or footnote. One sentence. Never buried. |
| **Exportable / Shareable Report** | Journalists, politicians, and analysts need to take findings out of the tool. Minimum: a static screenshot-friendly layout. | Low | Streamlit's native layout is printable. A "Download as PDF" button is a bonus but not required for demo. |

---

## Differentiators

Features that distinguish PolicyLens from existing tools. Not universally expected, but high-value when done well.

| Feature | Value Proposition | Complexity | Achievable at Hackathon? |
|---------|-------------------|------------|--------------------------|
| **Natural Language as the primary interface** | Every comparable tool (EPS, LEAP, NREL Scenario Viewer) requires users to configure sliders, percentages, and checkboxes for 50+ policy levers. PolicyLens accepts "Ban new coal plants by 2030" and handles the translation. This is a genuine differentiator — no major tool does this at a public-facing level. | High (already planned) | Yes — core architecture |
| **Economic vs. Environmental Tradeoff Score** | Reduce the full report to two scalar scores: an "economic health delta" and an "environmental health delta." Politicians and public want "good or bad?" before they want detail. A radar chart or a 2x2 quadrant plot of (Econ, Env) scores is instantly legible. | Medium | Yes — compute from weighted metric deltas |
| **Policy Ambiguity Warning** | When Claude's confidence in parsing is low (vague or out-of-domain policy text), surface an explicit warning: "This policy is too vague to model reliably — consider specifying a sector or target." No current public tool does this. | Low | Yes — inspect `reasoning` field and/or action vector sparsity |
| **Year-by-Year Narrative** | Instead of only showing charts, generate a one-sentence description for each projected year: "By 2028, renewable share reaches 38%, reducing consumer energy costs by 4%." Makes the report readable as prose, not just as charts. | Medium | Yes — Claude generation from trajectory data. Use streaming for live feel. |
| **State-Level Anchoring** | Pull the baseline from a user-selected US state via EIA SEDS, not just national averages. A California senator cares about California CO2 numbers. | Medium | Yes — EIA SEDS supports all 48 continental states |
| **Side-by-Side Policy Comparison** | Run two policies at once and show which is better on each metric. "Carbon tax vs. renewable mandate — which reduces CO2 more by 2030?" | High | Scope risk — cut if time is short. Implement as stretch goal. |

---

## Anti-Features

Features to deliberately exclude. Each one has a reason beyond "we ran out of time."

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Air quality / health metrics (PM2.5, NOx, ozone)** | EIA does not provide these directly. Linking CO2 reductions to health outcomes requires epidemiological modeling that is a separate research domain. Showing health numbers without this chain would be misleading. | Keep environmental metrics strictly to energy: CO2, renewable %, fossil fuel consumption, grid fuel mix. Label the section "Energy Impact." |
| **Interactive sliders for policy parameters** | Every existing tool already does this (EPS has 50+ sliders). It is not differentiating and it abandons the natural-language value proposition. Adding sliders signals "we didn't trust our AI." | Natural language input only. If a user wants to refine, they re-type with more specificity. |
| **Confidence intervals / uncertainty bands** | Monte Carlo or bootstrap uncertainty quantification requires either a trained ensemble model or many simulation runs. At demo time the GRU is untrained; confidence bands on a linear fallback are statistically meaningless. Showing fake uncertainty bands is worse than showing none. | Instead, use the fallback disclosure banner. "Illustrative projection" is honest. |
| **International / cross-country comparison** | EIA data is US-specific. Extending to IEA or World Bank data multiplies data integration complexity with no hackathon benefit. | US-only. State-level is sufficient scope. |
| **Real-time / live grid data** | EIA's real-time RTO fuel-type endpoint has 5-minute granularity. Building a live-updating dashboard is an engineering distraction and irrelevant to policy evaluation (policy operates on annual timescales). | Use annual SEDS historical data for baseline. No polling, no live updates. |
| **User accounts / saved reports** | Auth and persistence are multi-hour additions with no demo value. Judges will not log in. | Stateless session. Results exist while the browser tab is open. |
| **Regulatory/legal text analysis** | PolicyLens evaluates impact, not legality or constitutional compliance. Adding "Is this policy legal?" is a different product (legal AI). | Scope to impact only. Add a disclaimer: "This tool models projected energy outcomes, not legal feasibility." |
| **Demographic impact breakdown** | Breaking out impact by income quintile, race, or geography requires demographic weighting models that do not come from EIA data. Without this modeling, showing demographic numbers would be fabricated. | Stick to aggregate economic and environmental metrics. |

---

## Feature Dependencies

```
Plain-English Input
    └── Policy Summary Card (requires parsed reasoning field)
    └── Policy Ambiguity Warning (requires parsed action vector quality check)
    └── Projected State After Policy
            └── Baseline State (projection delta requires baseline)
            └── Time Horizon Display (requires trajectory, not point estimate)
            └── Before/After Comparison (requires both baseline and projection)
            └── Economic vs. Environmental Tradeoff Score (derived from trajectory)
            └── Year-by-Year Narrative (derived from trajectory)

Baseline State
    └── State-Level Anchoring (parametrizes which EIA baseline to pull)
```

---

## MVP Recommendation

For a hackathon demo that runs in 24–36 hours of build time:

**Prioritize (build these first, demo breaks without them):**
1. Plain-English input field + Claude parser + Policy Summary Card (trust anchor)
2. Baseline state display (EIA pull for 2–3 key metrics, national level)
3. Linear fallback projection (10 years, applied action deltas)
4. Time-series line chart (baseline vs. projected, at minimum CO2 and renewable %)
5. Before/After metric cards (big numbers, color-coded green/red delta)
6. Fallback disclosure banner

**Build second (high differentiator value, moderate effort):**
7. Economic vs. Environmental Tradeoff Score (two-number summary, ideally with a 2x2 quadrant plot)
8. Policy Ambiguity Warning
9. Year-by-Year Narrative (Claude streaming, impressive live demo effect)

**Stretch goals (only if core is done):**
10. State-Level Anchoring (state selector dropdown)
11. Side-by-Side Policy Comparison

**Defer entirely:**
- Anything in Anti-Features above
- Download/export (Streamlit's print-to-PDF is sufficient)
- GRU training during demo (use fallback; GRU can be dropped in if trained beforehand)

---

## Visualization Patterns

Specific chart recommendations for showing economic vs. environmental tradeoffs over time to a non-expert audience.

### Primary: Overlaid Line Chart (Time Series)
Use a single Plotly line chart per metric group. X-axis: years 1–10. Two lines per chart: "Status Quo (baseline)" in gray, "With Policy" in green (environmental metrics) or blue (economic metrics). Do NOT use dual-axis line charts on a single canvas — research consistently shows they mislead non-experts by implying false correlation strength. Keep axes separate.

### Summary: 2x2 Quadrant / Scorecard
A scatter point on two axes: X = Economic Health Delta (positive = better), Y = Environmental Health Delta (positive = better). Four quadrants: "Win-Win" (top-right), "Env Win / Econ Cost" (top-left), "Econ Win / Env Cost" (bottom-right), "Lose-Lose" (bottom-left). This is the most legible single-screen summary for a politician or journalist. Plot the policy as a labeled dot.

### Supporting: Metric Cards (Big Numbers)
Above the charts, show 3–5 `st.metric()` cards: CO2 Delta, Renewable % Delta, Fossil Fuel Delta, Consumer Energy Cost Delta, Energy Jobs Delta. `st.metric` has built-in delta arrow formatting (green up/down). Use these as the "headline" before the charts.

### Avoid
- Pie/donut charts for time-series data (no temporal dimension)
- Stacked area charts (hard to read individual series for non-experts)
- Heatmaps (wrong for 10-year projections across 3–5 metrics)
- Dual-axis line charts (mislead about correlation; Flourish's own documentation flags this as the primary misuse case)

---

## Sources

- [Energy Policy Simulator — Documentation](https://docs.energypolicy.solutions/) — Feature inventory for comparable tools (HIGH confidence)
- [Energy Innovation — EPS Overview](https://energyinnovation.org/report/the-energy-policy-simulator/) — Economic + environmental output metrics (HIGH confidence)
- [RMI — State EPS Analysis](https://rmi.org/energy-policy-simulator/) — State-level scoping precedent (HIGH confidence)
- [EIA OpenData — API Documentation](https://www.eia.gov/opendata/documentation.php) — Available state/national metrics (HIGH confidence)
- [EIA State Energy Data System (SEDS)](https://www.eia.gov/state/seds/) — State-level baseline data source (HIGH confidence)
- [Flourish — Dual Axis Charts](https://flourish.studio/blog/dual-axis-charts/) — Visualization anti-pattern evidence (MEDIUM confidence)
- [Number Analytics — Data Visualization in Policy](https://www.numberanalytics.com/blog/ultimate-data-visualization-policy-guide) — Non-expert audience design principles (MEDIUM confidence)
- [PMC — Innovation in Data Visualisation for Public Policy](https://pmc.ncbi.nlm.nih.gov/articles/PMC7933940/) — Policy visualization research (MEDIUM confidence)
- [Rapid Innovation — AI Policy Impact Analyzer Features](https://www.rapidinnovation.io/post/ai-agent-public-policy-impact-analyzer) — Feature landscape for AI policy tools (LOW confidence — marketing source)
- [ClimateWatch](https://www.climatewatchdata.org/) — Emissions scenario tool reference (MEDIUM confidence)
