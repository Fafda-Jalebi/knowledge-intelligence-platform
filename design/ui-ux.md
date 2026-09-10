# UI/UX Design System & Experience Specifications — AI Research OS

This document specifies the visual aesthetic, layout structures, and interaction paradigms for the **Agentic Multimodal Research Platform** (AI Research OS).

---

## 1. Core Experience & Visual Philosophy

1. **AI Research Operating System (Not a Simple Chatbot)**:
   - The interface is structured as an interactive research cockpit rather than a basic text bubble conversation.
   - Live research progress is visually deconstructed into sequential agent steps with real-time feedback.
2. **First-Class Evidence & Citation Layer**:
   - Every claim in the generated research report is clickable, opening a dedicated side sheet inspector showing the exact source snippet, document/web origin, page number, and confidence score.
   - Clear visual indicators for contradictions, evidence coverage percentages (e.g. `82% Evidence Coverage`), and source reliability.

---

## 2. Page Layouts & Workspaces

### 2.1 Research Cockpit & Dashboard
```
┌────────────────────────────────────────────────────────────────────────┐
│  AI RESEARCH PLATFORM                    [🔔] [User: Om | Quota: 84%]  │
├───────────────┬────────────────────────────────────────────────────────┤
│ 📊 Dashboard  │  What do you want to research?                         │
│ 🔬 Research   │  ┌──────────────────────────────────────────────────┐  │
│ 📚 Knowledge  │  │ Ask anything...                                  │  │
│ 📄 Documents  │  └──────────────────────────────────────────────────┘  │
│ 📁 Projects   │  [+ Add Documents]  [✔ Web] [✔ Knowledge] [✔ Data]     │
│ 📑 Reports    │                                                        │
│ ⚙️ Settings   │  Recent Research:                                      │
│               │  • Sustainable Food Packaging 10-Year Outlook  [82% ✅]│
│               │  • EV Commercial Fleet Adoption in India       [94% ✅]│
│               │  • MTech Thesis Literature Review & Datasets   [78% ⚠️]│
└───────────────┴────────────────────────────────────────────────────────┘
```

### 2.2 Live Research Progress Streaming Experience
When a research task executes, the UI shows real-time agent state transitions:
```
[●] Understanding Question...       [Done]
[●] Planning Research (10 tasks)... [Done]
[●] Checking Private Knowledge...   [Done] (Retrieved 8 chunks)
[●] Searching External Web...       [In Progress] (Scanned 14 sources)
[○] Analyzing Data & Images...      [Pending]
[○] Cross-Checking Evidence...      [Pending]
[○] Critiquing Findings...          [Pending]
[○] Synthesizing Final Report...    [Pending]
```

### 2.3 Final Structured Report View
The finalized output renders as a rich, structured research document:
- **Executive Summary**: High-level synthesis of findings.
- **Key Findings**: Numbered critical discoveries.
- **Evidence Matrix**:
  - `Claim 1` $\rightarrow$ `[Paper 1, Page 12]`, `[Industry Report 2026]`
  - `Claim 2` $\rightarrow$ `[Uploaded Dataset B, Section 4.2]`
- **Contradictions & Gaps**: Highlighted areas of scientific or market uncertainty.
- **Methodology & Data Analysis**: Embedded data charts and tables computed by deterministic tools.
- **Confidence Rating**: Overall evidence coverage gauge (e.g. `82% Evidence Coverage`).
- **Traceability Inspector**: "Why do you believe this?" button revealing raw underlying evidence.

---

## 3. Design Tokens & Color Palette

### Dark Mode (Primary Theme)
- **Base Background**: `#080b11` (Obsidian space)
- **Sidebar & Surface**: `#0e1420` (Navy slate)
- **Elevated Card / Drawer**: `#151e30`
- **Border / Outline**: `#23314d` / `rgba(255, 255, 255, 0.08)`
- **Accent Primary**: `#6366f1` (Indigo Glow) / Hover: `#818cf8`
- **Accent Secondary (Evidence / Verification)**: `#10b981` (Emerald) / `#06b6d4` (Cyan)
- **Warning / Contradiction**: `#f59e0b` (Amber)
- **Refusal / Error**: `#ef4444` (Rose)

---

## 4. Typography & Micro-Interactions

- **Typeface**: `Inter` / `Outfit` for display typography; `JetBrains Mono` for code and metadata counters.
- **Interactive Badges**: Glowing citation pill tags with tooltip previews on hover and instant source drawer expansion on click.
- **Live Stream Animations**: Shimmer pulse and progress bar indicating active agent execution.
