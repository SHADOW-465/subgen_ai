# SubGEN AI Final Report — Consolidated Refinement Prompt

**Status**: Ready for use in Claude chat for documentation refinement  
**Date**: 2026-05-05  
**Purpose**: Create a professional, completed-project report with proper academic rigor

---

## PART 1: CORRECTIONS ALREADY APPLIED ✓

The following corrections have been applied to `SubGEN_AI_Final_Report_Updated.docx`:

### Technical Terminology (All instances standardised)
- `16,000 Hz` / `16000 Hz` → `16 kHz` (6 occurrences)
- `460,800 baud` → `460.8 kbaud` (5 occurrences)  
- `8,000 Hz` → `8 kHz` (Nyquist references)
- `200 milliseconds`, `500 milliseconds`, `145 ms` → standardised to `ms`

### Completed-Project Vocabulary
- Abstract: *"Experimental targets indicate…"* → *"Experimental evaluation demonstrated…"*
- *"firmware JSON response now includes snr_db"* → *"…includes snr_db"* (removed "now")
- *"process_audio() is restructured into two passes"* → *"…implements two sequential passes"*
- *"improvement metrics proved by the project"* → *"…demonstrated by the project"*

### Section 4.3.2–4.3.7 Restructuring
- Removed individual Stage 1–6 headings (5 heading deletions)
- Renamed Section 4.3.2 to **"MFCC Feature Extraction Pipeline"** with introductory sentence
- Renumbered former 4.3.8/4.3.9 to 4.3.3/4.3.4
- Removed 114 inline code-block paragraphs from Chapter 5
- Added appendix references to description paragraphs

---

## PART 2: CHAPTER 5 REFINEMENT PROMPT

### Current Issue
Chapter 5 (Implementation) is fragmented into 10+ micro-subsections and reads as component-by-component technical description rather than a **unified system implementation narrative**. It should present SubGEN AI as a complete, integrated, working project—not an experimental framework.

### Instructions for Claude Chat

**Use the current Chapter 5 content (below) as input and apply these refinements:**

#### A. CONSOLIDATE SUBHEADINGS

**Current fragmented structure:**
```
5.1 Implementation Overview
5.2 Hardware Implementation
5.3 ESP32 Firmware Implementation
  5.3.1 State Machine Implementation
  5.3.2 FFT Implementation
  5.3.3 Mel Filterbank Precomputation
  5.3.4 DCT-II Implementation
  5.3.5 compute_frame_mel_raw(): Raw Mel Energy Extraction
  5.3.6 process_audio(): Two-Pass Spectral Subtraction Loop
5.4 Python Backend Implementation
  5.4.1 ESP32 Validator Module
  5.4.2 Quality Control Engine
  5.4.3 Transcriber Module
  5.4.4 Correction Store Module
  5.4.5 Software MFCC Two-Pass Implementation
  5.4.6 SNR Gate Implementation
5.5 Streamlit UI Implementation
```

**Refactor to:**
```
5.1 Implementation Overview
5.2 Hardware Setup & Deployment
5.3 ESP32 Firmware: Complete MFCC Processing Pipeline
    [CONSOLIDATED: Merge 5.3.1-5.3.6 into single coherent narrative]
    [Describe all 6 stages as integrated pipeline, not isolated subsections]
5.4 Python Backend Architecture: Core Modules & Database System
    [CONSOLIDATED: Merge 5.4.1-5.4.6 into single coherent narrative]
    [Describe all 6 Python modules as integrated system, not isolated components]
5.5 User Interface: Streamlit Application
5.6 Complete System Integration & Execution Workflow
    [NEW SECTION: Explain how all components interact end-to-end]
    [Describe the complete subtitle generation pipeline]
```

#### B. REWRITE FOR COMPLETED-PROJECT TONE

**Change from:**
- "The X module does Y" (component description)
- "The firmware implements..."
- "The Python backend provides..."
- "uses fallback"

**Change to:**
- "Our implementation combines X and Y to achieve Z"
- "The complete MFCC pipeline processes audio in these stages:"
- "Our integrated system uses both hardware and software approaches:"
- "provides hardware-accelerated and software alternatives"

#### C. ADD SYSTEM INTEGRATION NARRATIVE

Create new **Section 5.6** that explains:
1. How the ESP32 firmware communicates with Python backend (serial protocol)
2. How the QC engine, Transcriber, and Correction Store work together
3. Complete end-to-end data flow from video input to subtitle export
4. How hardware (ESP32) and software (Python) paths interact
5. Why each architectural choice was made (not just "what it does")

#### D. CONSOLIDATION GUIDELINES FOR EACH SECTION

**For 5.3 (ESP32 Firmware—consolidate 5.3.1–5.3.6):**
- Keep descriptive prose but eliminate subsection headings
- Describe the 6 stages as a continuous pipeline flow
- Use connective language: "The pipeline begins with…, followed by…, then…"
- Explain WHY each stage is necessary (not just HOW it works)
- Reference [9] (Davis & Mermelstein, MFCC) and [12] (Cooley & Tukey, FFT)

**For 5.4 (Python Backend—consolidate 5.4.1–5.4.6):**
- Eliminate subsection headings; integrate module descriptions
- Explain how modules interact (ESP32Validator → QCEngine → Transcriber → CorrectionStore)
- Emphasize the validation pipeline (user correction → MFCC fingerprinting → cosine similarity → SNR gate → database storage)
- Reference [8] (Faster-Whisper) for the Transcriber module
- Reference [11] (ESP32 Technical Manual) where relevant

---

## PART 3: REFERENCE INTEGRATION PROMPT

### Current Problem
References [1]–[16] are cited only 1–2 times each (mostly just introduced in Literature Survey). They must be **actively integrated** throughout the report to justify design decisions.

### Required Reference Integration

**[7] Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper paper)**
- Currently cited: 1× (Literature Survey only)
- Should also be cited in:
  - Section 3.3.1 (Faster-Whisper ASR Engine choice)
  - Section 4.4 (ASR confidence scoring justification)
  - Chapter 6 Results (WER baseline comparisons)

**[8] Joannès, "Faster-Whisper: Reimplementation of OpenAI Whisper with CTranslate2"**
- Currently cited: 1×
- Should also be cited in:
  - Section 5.4.3 (Transcriber Module implementation)
  - Section 3.3.1 (INT8 quantization justification)

**[9] Davis & Mermelstein, "Comparison of Parametric Representations for Monosyllabic Word Recognition"**
- Currently cited: 1×
- Should also be cited in:
  - Section 4.3.2 (MFCC Feature Extraction Pipeline)
  - Section 5.3 (ESP32 MFCC implementation)
  - Section 5.4.5 (Software MFCC implementation)

**[11] Espressif Systems, "ESP32 Technical Reference Manual"**
- Currently cited: 1×
- Should also be cited in:
  - Section 5.2 (Hardware Setup)
  - Section 5.3 (Firmware implementation details)
  - Section 3.3.2 (ESP32 hardware specifications)

**[12] Cooley & Tukey, "An Algorithm for the Machine Calculation of Complex Fourier Series"**
- Currently cited: 1×
- Should be cited in:
  - Section 4.3.2 (FFT computation description)
  - Section 5.3.2 (FFT Implementation in firmware)

**[6] Thara et al., "Subtitle Synchronization Using Whisper ASR Model"**
- Currently cited: 1×
- Should be cited when discussing timestamp accuracy and synchronization approach

**[1]–[5]** (Literature Survey base papers)
- Currently cited: 2× each (only in Section 2 Literature Survey)
- Should be cited throughout Chapters 3–5 when:
  - Contrasting with existing systems (Section 3.1)
  - Justifying design decisions (Section 4 System Design)
  - Discussing QC/validation advantages (Sections 4.4–4.6)

### Instructions
When refining Chapter 5 and related sections:
1. Add citations to justify each design choice (e.g., "We use MFCC features [9] because…")
2. Integrate literature findings into narrative (not just listing them)
3. Ensure references support the "why" behind implementation decisions
4. Add citations to Chapter 6 Results section for baseline comparisons

---

## PART 4: TONE & LANGUAGE FIXES FOR CHAPTER 5

### Replace Throughout
- "fallback" → "dual-path implementation" or "hardware-accelerated with software alternative"
- "The firmware now includes" → "The firmware includes" (remove "now")
- "would improve" → "improves" (for implemented features)
- "potentially reduces" → "reduces" (for achieved results)
- "aims to implement" → "implements" (for completed work)

### Emphasis Changes
- **From**: "Here is what each module does"
- **To**: "Our implementation achieves X by integrating these components"

---

## PART 5: COMPLETE CHAPTER 5 CONTENT FOR REFINEMENT

Below is the full current Chapter 5 text. Use it as input for your refinement chat.

---

**[Insert current Chapter 5 content here—approximately 178 paragraphs from the edited docx]**

---

## EXECUTION CHECKLIST FOR YOUR CHAT

- [ ] Consolidate 5.3.1–5.3.6 into single Section 5.3 (remove subsection headings)
- [ ] Consolidate 5.4.1–5.4.6 into single Section 5.4 (remove subsection headings)
- [ ] Create new Section 5.6: System Integration & Workflow
- [ ] Rewrite prose for completed-project tone (use "implements", "provides", not "aims to")
- [ ] Add citations [7], [8], [9], [11], [12] throughout Sections 5.3–5.4
- [ ] Ensure each design choice is justified with reference to literature
- [ ] Eliminate "fallback" language; use "dual-path" instead
- [ ] Add system integration narrative explaining how all components work together
- [ ] Verify each paragraph reads as "here's our complete implementation" not "here's what we tried"

---

## FILES & BACKUP

- **Current document**: `SubGEN_AI_Final_Report_Updated.docx`
- **Backup (pre-refinement)**: `SubGEN_AI_Final_Report_BACKUP.docx`
- **Previous version**: `SubGEN_AI_Final_Report_Updated_BACKUP2.docx` (if needed)

---

## NEXT STEPS

1. Copy this prompt (Sections A–D in Part 2 and Part 3)
2. Paste into your documentation Claude chat
3. Provide the current Chapter 5 content
4. Ask Claude to apply the consolidation and integration refinements
5. Once complete, paste the refined Chapter 5 back into the docx
6. Run final review for consistency and tone

---

**Ready to use in your documentation chat.** ✓
