# Answer Extraction & Matching Methodology

## Overview

This document defines the standardized process for evaluating model-generated answers
against ground truth (GT) in the self-reflective multi-turn VLM evaluation pipeline.

**Dataset**: `outputs/inference_v5/qwen-thought-mt-v1.jsonl` (8,717 samples)
**Model**: Qwen2.5-VL SFT checkpoint (multi-turn self-refinement)
**Evaluation targets**: `initial_answer` (turn 1) and `final_answer` (last turn) vs `gt_final_answer`

---

## Two-Tier Verification Process

### Tier 1: Rule-Based Extraction + Matching
- Automated script classifies question type, extracts core answers, compares them
- Each result is assigned a confidence level: `HIGH` or `NEEDS_REVIEW`

### Tier 2: Opus 4.6 Verification
- **ALL** rule-based results are verified by Claude Opus 4.6 in batches
- `HIGH` confidence results: spot-checked in random batches to validate rules
- `NEEDS_REVIEW` results: every sample reviewed individually
- Final verdict comes from Opus, not the script

---

## Question Type Classification

Classification is determined by the **hint** embedded in the question text.

| Type ID | Detection Rule | Count | Example Hint |
|---------|---------------|-------|-------------|
| `MCQ` | `"option letter"` in question (case-insensitive) | 2,776 | "provide the correct option letter, e.g., A, B, C, D" |
| `INTEGER` | `"integer answer"` in question | 335 | "requiring an integer answer and provide the final value, e.g., 1, 2, 3" |
| `FLOAT` | `"floating-point"` in question | 89 | "requiring a floating-point number with one decimal" |
| `FREE_FORM` | None of the above | 5,517 | "provide the final value, e.g., 1, 2.5, 300" or no hint |

---

## Extraction Rules by Type

### MCQ Extraction

**Goal**: Extract a single option letter (A-H) from both model answer and GT.

#### From Model Answer (priority order):
1. `\boxed{X}` — LaTeX boxed answer
2. `Answer: (X)` or `Answer: X` — explicit answer marker
3. `the answer is (X)` / `the answer is X` — natural language
4. `the correct answer is (X)` — variation
5. `Therefore, (X)` or `Therefore, X` at end
6. Last standalone option letter `(X)` in text
7. Last isolated capital letter A-H at end of text

#### From GT Answer (priority order):
1. Single letter: `"A"`, `"B"`, etc.
2. Parenthesized: `"(A)"`, `"(D) 21.6"`
3. Letter-comma-text: `"A, 15"`, `"C, 27°"`
4. Text-comma-letter: `"The deer tick..., C."`
5. Text with `(X)` embedded: `"...the answer is (C) 3mm."`
6. Yes/No mapping: Match `"Yes"`/`"No"` to the choice text
7. Embedded letter: Find option letter that matches a choice keyword
8. **No letter found → NEEDS_REVIEW**: GT is descriptive text or numeric without clear mapping

#### MCQ Matching Rule:
- **Exact letter match** after extraction
- If GT is `"Yes"`/`"No"`: map to option letter using choices, then compare
- If GT contains the answer VALUE (e.g., `"BF = 4"`) and model selected an option: match option's value to GT value

### INTEGER Extraction

**Goal**: Extract an integer from both sides.

#### From Model Answer:
1. `\boxed{N}` — LaTeX boxed
2. `Answer: N` or `the answer is N`
3. Last integer appearing in the text (after "is", "=", or at end of sentence)

#### From GT Answer:
- GT is typically clean: `"30"`, `"9"`, `"0"`
- Parse directly as integer

#### INTEGER Matching Rule:
- **Exact numeric match**: `extracted_model == extracted_gt`
- No tolerance

### FLOAT Extraction

**Goal**: Extract a decimal number from both sides.

#### From Model Answer:
1. `\boxed{N.N}` — LaTeX boxed
2. `Answer: N.N`
3. Last decimal number in text

#### From GT Answer:
- Can be clean (`"63.4"`) or verbose (`"The sum is 1765.0."`)
- Extract the last decimal number from GT text
- If GT contains a calculation chain, use the **final result** number

#### FLOAT Matching Rule:
- **Tolerance-based match**: `|model - gt| <= max(0.01, 0.005 * |gt|)`
- This allows 0.5% relative tolerance or 0.01 absolute tolerance, whichever is larger
- Handles rounding differences (e.g., 1765.0 vs 1765)

### FREE_FORM Extraction

**Goal**: Extract the core answer — may be numeric, expression, or text.

#### Sub-classification of FREE_FORM:
1. **Pure numeric**: GT is just a number → extract number, compare with tolerance
2. **Short expression**: GT is a math expression (`"y = x - 5"`, `"4 + 4√3"`) → normalize and compare
3. **Short text**: GT is brief text (`"even"`, `"ECONOMIC GROWTH"`) → case-insensitive string match
4. **Verbose GT**: GT contains reasoning + answer → extract final value, then compare
5. **Complex/Ambiguous**: Cannot reliably extract → NEEDS_REVIEW

#### FREE_FORM Matching Rules:
- **Numeric**: Same as FLOAT tolerance
- **Expression**: Normalize (remove spaces, standardize symbols), string match
- **Text**: Case-insensitive, strip punctuation, check if one contains the other
- **Ambiguous**: Flag as NEEDS_REVIEW for Opus verification

---

## Confidence Classification

### HIGH Confidence (rule-based result trusted, spot-check only):
- MCQ: Both model and GT yielded a clear single letter
- INTEGER: Both sides yielded a clean integer
- FLOAT: Both sides yielded a clean number
- FREE_FORM: Both sides yielded a clean number or exact text match

### NEEDS_REVIEW (Opus must verify):
- MCQ: GT has no clear option letter, or model answer is ambiguous
- MCQ: GT is Yes/No or descriptive text requiring choice mapping
- FLOAT: GT is verbose with multiple numbers
- FREE_FORM: Expression comparison, verbose GT, or text-only comparison
- Any case where extraction regex did not match cleanly

---

## Opus 4.6 Verification Ground Rules

When Opus reviews a sample, apply these rules consistently:

### Rule 1: Answer Equivalence, Not Explanation Quality
- We are comparing the **final answer value** only, NOT the reasoning
- A wrong explanation with a correct final answer = CORRECT
- A correct explanation with a wrong final answer = INCORRECT

### Rule 2: MCQ Option Letter Matching
- `"B"` and `"No"` are equivalent if choice (B) is "No"
- `"(B) No"` and `"B"` are equivalent
- Model saying `"Answer: B"` with verbose reasoning = B
- If model's extracted answer matches the GT option letter = CORRECT

### Rule 3: Numeric Tolerance
- Integers: Must be exact (30 ≠ 29)
- Floats: Allow ±0.5% relative or ±0.01 absolute tolerance
- Units don't need to match if the number is correct
- Scientific notation equivalence: `6.51 kN` = `6510 N` = `6.51` (if unit in question)

### Rule 4: Expression Equivalence
- `4 + 4√3` = `4 + 4\sqrt{3}` = `4(1 + √3)` (mathematically equivalent)
- `y = x - 5` = `x - 5` (if question asks for y or the function)
- For complex expressions: evaluate numerically if possible

### Rule 5: Partial Answer Handling
- If GT is `"9"` and model says `"9 cities"`: CORRECT (extracted value matches)
- If GT is `"BF = 4"` and model says `"4"`: CORRECT
- If GT is multi-part and model only answers part: NEEDS careful judgment

### Rule 6: Verbose GT with Reasoning
- Many GT answers contain reasoning chains from the multi-turn process
- Extract the **final numerical/categorical answer** from GT reasoning
- Compare that to the model's extracted answer
- Example: GT `"The sum is 1765.0"` → extract `1765.0`

### Rule 7: When GT Itself Appears Wrong
- Some GT answers may contain errors (e.g., hallucinated feedback led to wrong correction)
- Still evaluate against GT as given — we measure agreement, not absolute correctness
- Flag suspicious GT for separate analysis

### Rule 8: Edge Cases
- Model says "I cannot determine" or similar → INCORRECT (unless GT also says so)
- Model repeats the question → INCORRECT
- Model gives multiple candidate answers → Use the LAST one stated
- Empty or truncated answers → INCORRECT

---

## Metrics to Compute

### Primary Metrics
| Metric | Formula | What it measures |
|--------|---------|-----------------|
| Initial Accuracy | correct_initial / total | Base model capability |
| Final Accuracy | correct_final / total | Post-refinement capability |
| Self-Refinement Gain | final_acc - initial_acc | Does refinement help? |

### Per-Type Breakdown
- Accuracy by question type (MCQ, INTEGER, FLOAT, FREE_FORM)
- Gain by question type

### Transition Analysis
| Transition | Meaning |
|-----------|---------|
| Wrong → Right | Successful self-correction |
| Right → Right | Maintained correct answer |
| Right → Wrong | Regression (refinement hurt) |
| Wrong → Wrong | Failed to self-correct |

### Refinement Effectiveness
- Correction Rate = (Wrong→Right) / (Wrong initial) — how often does it fix mistakes?
- Regression Rate = (Right→Wrong) / (Right initial) — how often does it break correct answers?
- Net Gain = Correction Rate - Regression Rate

---

## Verification Workflow

1. **Run rule-based script** → produces per-sample results with confidence levels
2. **Opus spot-checks HIGH confidence** → validates rules on random batches (~100 samples per type)
3. **Opus reviews ALL NEEDS_REVIEW** → makes final correct/incorrect judgment
4. **Merge results** → combine rule-based HIGH + Opus-verified NEEDS_REVIEW
5. **Compute metrics** → final accuracy, gain, transitions
6. **Error analysis** → examine regressions and persistent failures

---

## File Outputs

| File | Contents |
|------|----------|
| `eval_results_rulebased.jsonl` | Per-sample: extracted answers, match result, confidence |
| `eval_needs_review.jsonl` | Samples flagged for Opus review |
| `eval_opus_verified.jsonl` | Opus judgments on reviewed samples |
| `eval_final_metrics.json` | Aggregate metrics |
| `eval_transitions.json` | Transition matrix (initial→final) |
