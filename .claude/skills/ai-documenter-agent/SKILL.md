---
name: ai-documenter-agent
description: Extract tribal knowledge, surface hidden assumptions, and produce structured documentation for AI agents. Use when describing a problem or feature that other specialized AI agents will implement, to ensure complete context with no assumptions left undocumented.
---

# AI Documenter Agent Skill

Surface hidden assumptions, extract tribal knowledge, and decompose problems into well-documented sub-problems with dependency graphs. Produces standardized documentation in `docs/` that other specialized AI agents can consume to implement solutions without making assumptions.

## Overview

This skill operates as a structured interview process. Given a problem or feature description, it:

1. Discovers existing documentation to avoid re-asking known context
2. Iteratively asks questions to surface hidden assumptions and tribal knowledge
3. Separates business context from technical context
4. Decomposes the problem into the smallest logical sub-problems
5. Builds a dependency DAG among sub-problems (Mermaid + YAML)
6. Documents everything in standardized folders with clear known/unknown tracking
7. Produces documentation that other AI agents can directly consume

**Critical principle:** Never stop questioning until the user confirms no assumptions remain. When someone says "obviously" or "of course" — that is tribal knowledge. Probe deeper.

## Phase 1: Context Discovery

Before asking any questions, scan the existing documentation structure.

### Actions

1. **Scan `docs/` directory** for existing documentation:
   - Read `docs/readme.md` (root index) if it exists
   - Read `docs/business/readme.md` if it exists
   - Read `docs/technical/readme.md` if it exists
   - Read `docs/features/readme.md` and `docs/bugs/readme.md` if they exist
   - Follow links in these index files to catalog all existing documents and their topics

2. **Present discovery summary** to the user:
   - List what documentation already exists
   - Note what areas have coverage vs. gaps
   - This avoids asking questions already answered in existing docs

3. **If no `docs/` structure exists**, inform the user that you will create it during the documentation phase.

## Phase 2: Initial Problem Understanding

Ask the user to describe their problem or feature. Then classify and scope it.

### Required Information

- [ ] Problem or feature description (in the user's own words)
- [ ] Classification: feature, bug fix, or other
- [ ] High-level goal: what should be true when this is done?
- [ ] Who requested or needs this? (stakeholder context)
- [ ] Any known urgency or timeline constraints

### Question Strategy

Start with: "Tell me about the problem or feature you're working on."

Follow up conversationally based on what they share. Do NOT present a checklist. Listen for implicit assumptions in every answer.

## Phase 3: Knowledge Extraction — Business Context

Probe for hidden assumptions about business rules, processes, and domain knowledge. This is where tribal knowledge most often hides.

### Areas to Probe

- [ ] **Business rules**: What rules govern this domain? Are there exceptions?
- [ ] **Domain terminology**: What terms mean something specific here? (feeds the glossary)
- [ ] **Stakeholders**: Who are the users? Who else is affected? What are their concerns?
- [ ] **Current process**: How is this handled today (manually or otherwise)?
- [ ] **Business constraints**: Regulatory, compliance, contractual obligations
- [ ] **Success criteria**: How will the business measure whether this succeeded?
- [ ] **Historical context**: Why does the current system work this way? Past decisions that constrain the present?
- [ ] **Edge cases**: Unusual but valid scenarios the business handles

### Question Strategy

- Ask follow-up questions when answers contain jargon or implicit assumptions
- When the user says "obviously" or "of course" — probe deeper, that is tribal knowledge
- When the user references a process or system by name, ask what it does and why
- Check existing `docs/business/` content — skip questions already answered there
- Track every answer as **known** and every gap as **unknown**

See [questioning-framework.md](references/questioning-framework.md) for detailed tribal knowledge extraction techniques.

## Phase 4: Knowledge Extraction — Technical Context

Probe for hidden assumptions about architecture, systems, and technical constraints. Only go as deep as needed — the consuming agent may or may not need all technical detail.

### Areas to Probe

- [ ] **System architecture**: What systems are involved? How do they communicate?
- [ ] **Data model**: What data entities are relevant? Where do they live?
- [ ] **Integrations**: What external systems, APIs, or services are involved?
- [ ] **Technology stack**: Languages, frameworks, databases, infrastructure
- [ ] **Performance requirements**: Latency, throughput, scale expectations
- [ ] **Security requirements**: Authentication, authorization, data sensitivity
- [ ] **Testing approach**: How is this area tested today? What coverage exists?
- [ ] **Deployment**: How are changes deployed? Any special considerations?
- [ ] **Technical debt**: Known issues, workarounds, or fragile areas near this problem

### Question Strategy

- Check existing `docs/technical/` content — skip questions already answered there
- Ask about technical context only to the depth needed for the problem at hand
- Separate "must know to solve the problem" from "nice to have context"
- If the user doesn't know a technical answer, mark it as **unknown** — don't skip it

## Phase 5: Problem Decomposition

Once you have sufficient context, decompose the problem into the smallest logical sub-problems.

### Decomposition Process

1. **Identify atomic sub-problems** — each should be independently understandable and solvable
2. **Map dependencies** — which sub-problems require others to be completed first?
3. **Build the DAG** — create a directed acyclic graph in both Mermaid and YAML formats
4. **Validate with the user** — present the decomposition and ask if anything is missing or incorrectly split

### For Each Sub-Problem, Capture

- [ ] Clear description of what needs to be solved
- [ ] Why this is a separate sub-problem (not merged with another)
- [ ] Dependencies (which other sub-problems must be completed first)
- [ ] Acceptance criteria (concrete, testable conditions for "done")
- [ ] Known constraints specific to this sub-problem
- [ ] Examples and edge cases (concrete input/output or before/after)
- [ ] Risk level and impact if solved incorrectly
- [ ] Links to relevant existing documentation

See [decomposition-guide.md](references/decomposition-guide.md) for DAG construction methodology.

## Phase 6: Sub-Problem Deep Dive

Follow the DAG ordering — start with independent sub-problems (no dependencies) and work through the graph as dependencies are resolved.

### For Each Sub-Problem (in DAG Order)

1. **Ask targeted questions** specific to this sub-problem
2. **Surface assumptions** — what is the user taking for granted about this piece?
3. **Probe for edge cases** — "What happens when...?" scenarios
4. **Define acceptance criteria** — concrete, testable conditions
5. **Assess risk** — what breaks if this is wrong? How hard is it to fix?
6. **Capture examples** — concrete input/output or before/after scenarios

### Completion Criteria

Continue asking questions until:

- Every sub-problem has clear acceptance criteria
- No question is answered with "it depends" without clarifying what it depends on
- The user confirms there are no remaining assumptions
- All unknowns are explicitly documented as unknowns (not silently skipped)

**Important:** The DAG informs question ordering as a preference, but allow the conversation to flow naturally when the user takes it somewhere productive. Don't rigidly enforce ordering if a natural thread surfaces important information.

## Phase 7: Documentation Generation

Generate all documentation files in the standardized folder structure.

### Output Structure

```
docs/
├── readme.md                              # Root index (create or update)
├── business/
│   ├── readme.md                          # Business context index (create or update)
│   ├── glossary.md                        # Domain terminology (create or update)
│   └── {topic-specific}.md               # As needed for reusable business context
├── technical/
│   ├── readme.md                          # Technical context index (create or update)
│   ├── constraints.md                     # Non-functional requirements (create or update)
│   └── {topic-specific}.md               # As needed for reusable technical context
├── features/{feature-name}/               # For features
│   ├── overview.md                        # Problem statement, known/unknown, acceptance criteria
│   ├── decomposition.md                   # Sub-problems with DAG (Mermaid + YAML)
│   └── sub-problems/
│       ├── {sub-problem-1}.md             # Individual sub-problem detail
│       └── {sub-problem-2}.md
└── bugs/{bug-name}/                       # For bugs
    ├── overview.md                        # Bug description, known/unknown
    ├── decomposition.md                   # Root cause analysis with DAG
    └── sub-problems/
        ├── {sub-problem-1}.md
        └── {sub-problem-2}.md
```

### Documentation Rules

1. **Link, don't duplicate** — If business or technical context already exists in `docs/business/` or `docs/technical/`, link to it from the feature/bug docs. NEVER copy content.
2. **Update indexes** — Always update the relevant `readme.md` index files to include new documents.
3. **Separate concerns** — Business context in `docs/business/`, technical context in `docs/technical/`, problem-specific docs in `docs/features/` or `docs/bugs/`. A consuming agent chooses which folders to read.
4. **Update glossary** — Add any new domain terms discovered during questioning to `docs/business/glossary.md`.
5. **Track known/unknown** — Every document must have explicit **Known** and **Unknown / Open Questions** sections.
6. **Present documents for review** — Show each document to the user before writing it. Ask for corrections.

See [document-templates.md](references/document-templates.md) for complete templates for each document type.

## Wrap Up

After generating all documentation, present a summary:

1. **Documentation map** — List all files created or updated with one-line descriptions
2. **Known/Unknown summary** — Aggregate view of what is known and what remains open across all sub-problems
3. **Recommended next steps** — Which sub-problem to tackle first based on the DAG (independent nodes first)
4. **Agent consumption guide** — Brief instructions for pointing other AI agents at this documentation, including which folders are relevant for different agent types

### Follow-Up Options

1. **Deep dive on unknowns** — Investigate open questions further to convert unknowns to knowns
2. **Refine decomposition** — Adjust sub-problems or dependencies based on new understanding
3. **Add more context** — Document additional business or technical knowledge discovered later
4. **Update existing docs** — Refresh previously documented context that has changed
5. **Generate agent prompt** — Create a starter prompt that points an implementing agent at the right documentation files
