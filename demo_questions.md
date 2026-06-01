# Demo Questions

Use one question per demo segment. These prompts are chosen to show the main system behaviours: statute-first Q/A, case-law retrieval, uploaded-document Q/A, and case-review prediction.

## Case Q/A

### 1. Constitutional Personal Liberty

```text
Which constitutional provision protects personal liberty in India?
```

Expected behaviour: routes to official reference law and answers using Article 21 of the Constitution of India.

### 2. Constitutional Property Protection

```text
Can a person be deprived of property without authority of law under the Constitution of India?
```

Expected behaviour: routes to official reference law and answers using Article 300A.

### 3. Consumer Refund Remedy

```text
What remedy is available if an online seller refuses to refund money for a defective product?
```

Expected behaviour: routes to the Consumer Protection Act, 2019 and explains Section 39 remedies such as refund, replacement, compensation, and costs.

### 4. RTI Non-Response Remedy

```text
What can an applicant do if an RTI application is not answered within 30 days?
```

Expected behaviour: routes law-first to the RTI Act and explains the Section 19 first-appeal route after expiry of the Section 7 response period.

### 5. Similar Case Retrieval

```text
Are there similar cases where a government employee challenged dismissal because they were not given a reasonable opportunity of hearing?
```

Expected behaviour: retrieves service-law demo cases about dismissal, hearing, notice, disciplinary procedure, and natural justice.

### 6. Case Explanation

```text
demo_service_hearing_001 explain this case
```

Expected behaviour: explains the facts, issue, likely reasoning, and limits of the selected demo case record.

## Uploaded Document Q/A

Upload `sample_data/uploaded_document_demo.txt`, then ask:

```text
What is the main issue in this document?
```

Expected behaviour: answers from the uploaded-document lane and identifies the RTI non-response / first appeal issue.

## Case Review / Prediction Inputs

Use these structured facts in the Case Review tab. The prediction output is a triage signal only; it should be read together with retrieved authorities and caution text.

### 1. RTI Delay Case

```text
Case type: Information / RTI dispute
Your role: Applicant
Forum / court type: Information Commission / High Court
Core facts: The applicant filed an RTI request seeking public records. No reply was given within thirty days, and the first appeal also remained pending.
Relief sought: Direction to provide available information and consider delay.
Evidence / documents: RTI application, postal receipt, first appeal copy.
Opponent's main argument: The authority may claim that the records are not readily available or are exempt.
```

Expected behaviour: likely favorable triage signal with RTI non-response cases and RTI appeal provisions.

### 2. Service Dismissal Without Hearing

```text
Case type: Service / disciplinary proceedings
Your role: Employee / Petitioner
Forum / court type: High Court
Core facts: A government employee was dismissed after an internal complaint. No show-cause notice, charge memorandum, or personal hearing was given before the dismissal order.
Relief sought: Set aside the dismissal order and permit fresh proceedings only according to law.
Evidence / documents: Dismissal order, service record, representation copy.
Opponent's main argument: The department may argue that the misconduct was serious and urgent action was required.
```

Expected behaviour: likely favorable triage signal with similar cases about reasonable opportunity, hearing, and disciplinary fairness.

### 3. Consumer Defective Product Case

```text
Case type: Consumer dispute
Your role: Complainant
Forum / court type: Consumer Commission
Core facts: The buyer purchased an electronic product online. The product was defective on delivery, and the seller refused refund or replacement despite repeated written complaints.
Relief sought: Refund, compensation, and costs.
Evidence / documents: Invoice, delivery record, complaint emails, product photos.
Opponent's main argument: The seller may say the product was damaged after delivery or that only repair is available.
```

Expected behaviour: likely favorable triage signal with consumer remedy authorities and Section 39 reference-law support.
