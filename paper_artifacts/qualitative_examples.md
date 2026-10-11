# Qualitative Examples for Paper

## 1. Adjudication Success (Hallucination Removal)
### Example 1.1 (ID: 0)
**Transcript Snippet**: ...[Clinician] Okay, let's see here. We've got a 67-year-old patient admitted with a recent history of seizures. Uh, we need to be mindful of the fall risks. Um, during the examination, we observed unequal pupil response, which might, uh, indicate a past head trauma or some neurological issue. Now, let...

**Hallucination (V3 Union)**:
```json
{
  "id": "78",
  "value": "Yes"
}
```
**Outcome**: Adjudication correctly **removed** this observation (not present in Gold).

### Example 1.2 (ID: 0)
**Transcript Snippet**: ...[Clinician] Okay, let's see here. We've got a 67-year-old patient admitted with a recent history of seizures. Uh, we need to be mindful of the fall risks. Um, during the examination, we observed unequal pupil response, which might, uh, indicate a past head trauma or some neurological issue. Now, let...

**Hallucination (V3 Union)**:
```json
{
  "id": "107",
  "value": "High"
}
```
**Outcome**: Adjudication correctly **removed** this observation (not present in Gold).

### Example 1.3 (ID: 0)
**Transcript Snippet**: ...[Clinician] Okay, let's see here. We've got a 67-year-old patient admitted with a recent history of seizures. Uh, we need to be mindful of the fall risks. Um, during the examination, we observed unequal pupil response, which might, uh, indicate a past head trauma or some neurological issue. Now, let...

**Hallucination (V3 Union)**:
```json
{
  "id": "21",
  "value": "walker"
}
```
**Outcome**: Adjudication correctly **removed** this observation (not present in Gold).

### Example 1.4 (ID: 0)
**Transcript Snippet**: ...[Clinician] Okay, let's see here. We've got a 67-year-old patient admitted with a recent history of seizures. Uh, we need to be mindful of the fall risks. Um, during the examination, we observed unequal pupil response, which might, uh, indicate a past head trauma or some neurological issue. Now, let...

**Hallucination (V3 Union)**:
```json
{
  "id": "89",
  "value": "Mildly impaired"
}
```
**Outcome**: Adjudication correctly **removed** this observation (not present in Gold).

### Example 1.5 (ID: 0)
**Transcript Snippet**: ...[Clinician] Okay, let's see here. We've got a 67-year-old patient admitted with a recent history of seizures. Uh, we need to be mindful of the fall risks. Um, during the examination, we observed unequal pupil response, which might, uh, indicate a past head trauma or some neurological issue. Now, let...

**Hallucination (V3 Union)**:
```json
{
  "id": "130",
  "value": "Disoriented to time"
}
```
**Outcome**: Adjudication correctly **removed** this observation (not present in Gold).


## 2. Repair Success (Value Correction)
### Example 2.1 (ID: 5)
**Transcript Snippet**: ...[Clinician] Patient is a 68-year-old female with a history of COPD and mild cognitive impairment, uh, recently admitted after an episode of acute bronchitis. She's, um, currently on oxygen therapy via nasal cannula, maintaining her oxygen saturation at 92%. Uh, her mobility is, uh, mildly impaired d...

**Concept ID**: 38
- **V4 Value (Before Repair)**: `True`
- **V6 Value (After Repair)**: `Yes` (Matches Gold)
- **Gold Value**: `Yes`

### Example 2.2 (ID: 14)
**Transcript Snippet**: ...[Clinician] Alright, let's see here... Um, this is a dictation for the patient in room, uh, 204. Patient is an elderly male, um, recently had a neurological event. Presenting with, uh, noticeable left facial droop. Uh, motor strength is, uh, 2 out of 5. Now, cognitive status... the patient is alert,...

**Concept ID**: 25
- **V4 Value (Before Repair)**: `borderline inadequate`
- **V6 Value (After Repair)**: `inadequate` (Matches Gold)
- **Gold Value**: `inadequate`

### Example 2.3 (ID: 20)
**Transcript Snippet**: ...[Clinician] Alright, so with this patient, we're seeing increased work of breathing, um, and they are using accessory muscles, which, ah, indicates they're really struggling a bit to breathe. Breathing pattern is labored, and, uh, you can notice the chest expansion is unequal. Now, oxygen saturation...

**Concept ID**: 13
- **V4 Value (Before Repair)**: `labored`
- **V6 Value (After Repair)**: `['labored']` (Matches Gold)
- **Gold Value**: `['labored']`

### Example 2.4 (ID: 53)
**Transcript Snippet**: ...[Clinician] Alright, let's see. Uh, the patient, um, let's start with the Glasgow Coma Score, uh, for eye openingâ€”it's, uh, spontaneous. The best motor response, uh, is, um, well, they obey commands. Uh, speaking of commands, the patient does follow them, so that's a positive, uh, command followin...

**Concept ID**: 60
- **V4 Value (Before Repair)**: `intact`
- **V6 Value (After Repair)**: `Normal` (Matches Gold)
- **Gold Value**: `Normal`

### Example 2.5 (ID: 81)
**Transcript Snippet**: ...[Clinician] Patient is, uh, alert but there's general confusion and forgetfulness. He's disorientedâ€”uh, oriented to person and place but not to time or situation, so x2. Um, memory is a bit forgetful at times, which is something we're keeping an eye on. Now, looking at the behavior, the Broset vio...

**Concept ID**: 132
- **V4 Value (Before Repair)**: `Yes`
- **V6 Value (After Repair)**: `Mild` (Matches Gold)
- **Gold Value**: `Mild`

