# Document Templates Reference

This document provides detailed templates for each planning document. These templates are designed to be comprehensive while remaining scannable by both humans and AI agents.

---

## PROJECT-SPEC.md Template

```markdown
# [Project Name] Specification

> [One-sentence description of what this project does]

## Document Info
| Field | Value |
|-------|-------|
| **Status** | Draft / In Review / Approved |
| **Author** | [Name] |
| **Created** | [Date] |
| **Last Updated** | [Date] |
| **Reviewers** | [Names] |

---

## 1. Overview

### 1.1 Problem Statement

**Current State:**
[Describe the current situation and its problems]

**Pain Points:**
- [Pain point 1]
- [Pain point 2]
- [Pain point 3]

**Impact:**
[What is the cost of not solving this problem?]

### 1.2 Proposed Solution

[Brief description of what we're building to solve the problem]

**Key Benefits:**
- [Benefit 1]
- [Benefit 2]
- [Benefit 3]

### 1.3 Target Users

| User Type | Description | Primary Needs |
|-----------|-------------|---------------|
| [User 1] | [Who they are] | [What they need] |
| [User 2] | [Who they are] | [What they need] |

### 1.4 Success Criteria

| Metric | Target | How to Measure |
|--------|--------|----------------|
| [Metric 1] | [Target value] | [Measurement method] |
| [Metric 2] | [Target value] | [Measurement method] |

---

## 2. Requirements

### 2.1 Core Requirements (P0 - Must Have)

These are required for the project to be considered complete.

#### REQ-001: [Requirement Name]
- **Description:** [What the system must do]
- **User Story:** As a [user type], I want to [action] so that [benefit]
- **Acceptance Criteria:**
  - [ ] [Criterion 1]
  - [ ] [Criterion 2]
  - [ ] [Criterion 3]

#### REQ-002: [Requirement Name]
- **Description:** [What the system must do]
- **User Story:** As a [user type], I want to [action] so that [benefit]
- **Acceptance Criteria:**
  - [ ] [Criterion 1]
  - [ ] [Criterion 2]

### 2.2 Secondary Requirements (P1 - Should Have)

These significantly improve the solution but aren't blocking.

#### REQ-101: [Requirement Name]
- **Description:** [What the system should do]
- **Acceptance Criteria:**
  - [ ] [Criterion 1]

### 2.3 Nice-to-Have Requirements (P2 - Could Have)

These would be nice but can be deferred.

#### REQ-201: [Requirement Name]
- **Description:** [What the system could do]

### 2.4 Non-Goals (Explicitly Out of Scope)

These are things we are NOT building. This section is critical for preventing scope creep.

| Non-Goal | Rationale |
|----------|-----------|
| [Thing not building] | [Why it's out of scope] |
| [Another non-goal] | [Why it's out of scope] |

---

## 3. Technical Specification

### 3.1 Tech Stack

| Layer | Technology | Version | Rationale |
|-------|------------|---------|-----------|
| Language | [e.g., TypeScript] | [e.g., 5.0+] | [Why chosen] |
| Framework | [e.g., Next.js] | [e.g., 14.x] | [Why chosen] |
| Database | [e.g., PostgreSQL] | [e.g., 15+] | [Why chosen] |
| ORM | [e.g., Prisma] | [e.g., 5.x] | [Why chosen] |
| Testing | [e.g., Vitest] | [e.g., 1.x] | [Why chosen] |

### 3.2 Architecture Overview

```
[ASCII diagram or description of system architecture]

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   Server    │────▶│  Database   │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Components:**
- **[Component 1]:** [Description and responsibility]
- **[Component 2]:** [Description and responsibility]

### 3.3 Data Model

```
[Entity Relationship description or diagram]

User
├── id: UUID (PK)
├── email: String (unique)
├── name: String
└── created_at: DateTime

Post
├── id: UUID (PK)
├── user_id: UUID (FK → User)
├── title: String
├── content: Text
└── created_at: DateTime
```

### 3.4 API Design

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/api/resource` | GET | List resources | Yes |
| `/api/resource` | POST | Create resource | Yes |
| `/api/resource/:id` | GET | Get resource | Yes |

### 3.5 Integration Points

| System | Type | Purpose | Notes |
|--------|------|---------|-------|
| [System 1] | API | [What for] | [Any notes] |
| [System 2] | Webhook | [What for] | [Any notes] |

---

## 4. Constraints & Considerations

### 4.1 Technical Constraints

| Constraint | Impact | Mitigation |
|------------|--------|------------|
| [Constraint 1] | [How it affects us] | [How we handle it] |

### 4.2 Security Requirements

- [ ] [Security requirement 1]
- [ ] [Security requirement 2]
- [ ] [Security requirement 3]

### 4.3 Performance Requirements

| Metric | Requirement | Notes |
|--------|-------------|-------|
| Response Time | [e.g., p95 < 200ms] | [Context] |
| Throughput | [e.g., 1000 req/s] | [Context] |
| Availability | [e.g., 99.9%] | [Context] |

### 4.4 Compliance Requirements

- [ ] [Compliance requirement if any]

---

## 5. Risks & Mitigations

| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|----|------|------------|--------|------------|-------|
| R1 | [Risk description] | High/Med/Low | High/Med/Low | [How to mitigate] | [Who] |
| R2 | [Risk description] | High/Med/Low | High/Med/Low | [How to mitigate] | [Who] |

---

## 6. Open Questions

Questions that need to be resolved before or during implementation.

| ID | Question | Status | Owner | Resolution |
|----|----------|--------|-------|------------|
| Q1 | [Question] | Open/Resolved | [Who] | [Answer if resolved] |
| Q2 | [Question] | Open/Resolved | [Who] | [Answer if resolved] |

---

## 7. Appendix

### 7.1 Glossary

| Term | Definition |
|------|------------|
| [Term] | [Definition] |

### 7.2 References

- [Link to related document]
- [Link to research/inspiration]
```

---

## IMPLEMENTATION-PLAN.md Template

```markdown
# [Project Name] Implementation Plan

> Phased implementation plan for [Project Name]

## Document Info
| Field | Value |
|-------|-------|
| **Spec Version** | [Link to PROJECT-SPEC.md] |
| **Created** | [Date] |
| **Last Updated** | [Date] |
| **MVP Phase** | Phase [N] |

---

## Implementation Overview

| Phase | Name | Objective | Est. Tasks |
|-------|------|-----------|------------|
| 1 | [Name] | [One-line objective] | [N] |
| 2 | [Name] | [One-line objective] | [N] |
| 3 | [Name] | [One-line objective] | [N] |

**Dependency Flow:**
```
Phase 1 (Foundation) → Phase 2 (Core) → Phase 3 (Polish)
```

---

## Phase 1: Foundation

### Objective
[What this phase accomplishes - what's true at the end that wasn't true before]

### Prerequisites
- [What must exist before starting this phase]

### Tasks

#### Task 1.1: [Task Name]

**Priority:** P0 (Blocking) / P1 (Important) / P2 (Nice-to-have)

**Description:**
[Clear description of what needs to be done]

**Implementation Details:**
```
Files to create:
- src/path/to/new-file.ts

Files to modify:
- src/path/to/existing-file.ts
  - Add: [what to add]
  - Change: [what to change]

Patterns to follow:
- [Reference existing pattern in codebase]
```

**Acceptance Criteria:**
- [ ] [Specific, testable criterion]
- [ ] [Specific, testable criterion]

**Dependencies:** None

**Verification:**
```bash
# Commands to verify completion
npm run test:unit -- path/to/test
npm run lint
```

---

#### Task 1.2: [Task Name]

**Priority:** P0

**Description:**
[What needs to be done]

**Implementation Details:**
```
Files to create:
- [files]

Files to modify:
- [files]
```

**Acceptance Criteria:**
- [ ] [Criterion]

**Dependencies:** Task 1.1

**Verification:**
```bash
[verification commands]
```

---

### Phase 1 Summary

**Deliverables:**
- [ ] [Deliverable 1]
- [ ] [Deliverable 2]

**Phase Verification:**
```bash
# Run all phase 1 verifications
npm run test
npm run build
npm run lint
```

**Exit Criteria:**
- [ ] All phase tasks complete
- [ ] All tests passing
- [ ] No lint errors
- [ ] [Any other criteria]

---

## Phase 2: Core Features

### Objective
[What this phase accomplishes]

### Prerequisites
- Phase 1 complete
- [Any other prerequisites]

### Tasks

#### Task 2.1: [Task Name]
[Same structure as Phase 1 tasks...]

---

## Phase N: Polish & Launch

### Objective
[Final phase objectives]

### Prerequisites
- All previous phases complete

### Tasks

#### Task N.1: [Task Name]
[Same structure...]

---

## Technical Decisions

Decisions made during planning that affect implementation.

| ID | Decision | Options Considered | Choice | Rationale |
|----|----------|-------------------|--------|-----------|
| D1 | [What decision] | [Option A], [Option B] | [Choice] | [Why] |
| D2 | [What decision] | [Options] | [Choice] | [Why] |

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| [Date] | Initial plan created | [Name] |
| [Date] | [What changed] | [Name] |
```

---

## TASKS.md Template

```markdown
# [Project Name] Tasks

> Current implementation status and task tracking

**Current Phase:** [Phase N]
**Last Updated:** [Date/Time]

---

## Quick Status

| Phase | Status | Progress |
|-------|--------|----------|
| 1. Foundation | Complete / In Progress / Not Started | [N/M tasks] |
| 2. Core | Complete / In Progress / Not Started | [N/M tasks] |
| 3. Polish | Complete / In Progress / Not Started | [N/M tasks] |

**Next Action:** [Brief description of immediate next step]

---

## In Progress

Tasks currently being worked on.

### [ ] TASK-1.2: [Task Name]
**Phase:** 1 - Foundation
**Priority:** P0
**Started:** [Date]

**Context:**
[Why this task, what it achieves, any relevant background]

**Current Status:**
[Where we are in this task]

**Remaining Steps:**
1. [ ] [Remaining step 1]
2. [ ] [Remaining step 2]
3. [ ] [Remaining step 3]

**Files:**
- `src/path/file.ts` - [Status: created/modified/pending]
- `src/path/other.ts` - [Status]

**Blockers:**
- [Any blockers, or "None"]

**Verification:**
```bash
[How to verify when complete]
```

---

## Ready (Up Next)

Tasks ready to start, no blockers.

### [ ] TASK-1.3: [Task Name]
**Phase:** 1 - Foundation
**Priority:** P0
**Blocked By:** None

**Summary:**
[Brief description of what this task accomplishes]

**Quick Start:**
```bash
# First command to run when starting this task
[command]
```

**Key Files:**
- `src/path/to/main-file.ts`

---

### [ ] TASK-1.4: [Task Name]
**Phase:** 1 - Foundation
**Priority:** P1
**Blocked By:** None

**Summary:**
[Brief description]

---

## Blocked

Tasks that cannot proceed until dependencies are resolved.

### [ ] TASK-2.1: [Task Name]
**Phase:** 2 - Core
**Blocked By:** Phase 1 completion

**Waiting For:**
- [ ] TASK-1.2 completion
- [ ] TASK-1.3 completion

---

## Backlog

Future tasks not yet ready to schedule.

### [ ] TASK-2.2: [Task Name]
**Phase:** 2 - Core

### [ ] TASK-2.3: [Task Name]
**Phase:** 2 - Core

### [ ] TASK-3.1: [Task Name]
**Phase:** 3 - Polish

---

## Completed

### [x] TASK-1.1: [Task Name]
**Completed:** [Date]
**Phase:** 1 - Foundation

**Summary:** [What was accomplished]

**Changes Made:**
- Created `src/path/file.ts`
- Modified `src/path/other.ts`

**Notes:**
[Any relevant notes for future reference]

---

## Session Log

Brief log of work sessions for context continuity.

### [Date] - Session N
**Tasks Worked:** TASK-1.1, TASK-1.2
**Completed:** TASK-1.1
**Notes:** [Any context for next session]

### [Date] - Session N-1
**Tasks Worked:** TASK-1.1
**Completed:** None
**Notes:** [Context]
```

---

## Best Practices for Templates

### For AI Agent Consumption

1. **Use consistent heading levels** - AI can parse structure reliably
2. **Include file paths** - Specific paths eliminate guesswork
3. **Add verification commands** - Testable completion criteria
4. **Keep tasks atomic** - One clear outcome per task
5. **Note dependencies explicitly** - Prevents out-of-order execution

### For Human Review

1. **Executive summaries** - Quick status at the top
2. **Progress indicators** - Visual checkboxes and tables
3. **Context sections** - Why, not just what
4. **Change logs** - Track evolution over time

### Maintenance

1. **Update TASKS.md frequently** - After each task completion
2. **Keep specs in sync** - Update when requirements change
3. **Archive completed phases** - Move to bottom or separate file
4. **Review weekly** - Ensure accuracy and relevance
