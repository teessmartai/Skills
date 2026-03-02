# Questioning Framework for Tribal Knowledge Extraction

Reference this guide when conducting knowledge extraction interviews. These techniques help surface assumptions, implicit knowledge, and undocumented context that the user may not realize they possess.

## Signal Phrases That Indicate Hidden Knowledge

When the user says any of these, it signals tribal knowledge that must be documented:

| Signal Phrase | What It Means | How to Probe |
|---|---|---|
| "Obviously..." / "Of course..." | The user considers this common knowledge, but it isn't documented | "Can you explain that as if I were a new team member?" |
| "Everyone knows..." | Widespread assumption within the team | "What specifically does everyone know? When did this become established?" |
| "We always..." / "We never..." | Undocumented process rule | "Why? What happens if that rule is broken? Is it enforced anywhere?" |
| "It's just how it works" | Historical technical or business decision | "Why was it built/decided this way? What were the alternatives?" |
| "It depends..." | Complex conditional logic not yet captured | "What does it depend on? Walk me through each scenario." |
| "Long story..." / "Historically..." | Past decision with lasting impact | "Can you give me the short version? What constraints does this create today?" |
| "You'd have to talk to [person]" | Knowledge siloed with an individual | "What do you know about it? What would they tell me?" |
| "That's an edge case" | Uncommon but real scenario | "How often does it happen? What do you do when it does?" |
| "We handle that manually" | Process gap or workaround | "Walk me through the manual process step by step." |
| "The system doesn't support that" | Known limitation with workaround | "How do users deal with it today? Is fixing it in scope?" |

## Core Questioning Techniques

### 1. The Newcomer Test

Frame questions as if a new team member needs to understand the domain from scratch. This forces the user to make implicit knowledge explicit.

**Pattern:** "If a new developer joined tomorrow and needed to work on this, what would they need to know that isn't written down anywhere?"

**Follow-ups:**
- "What mistakes would they likely make?"
- "What would surprise them about how this works?"
- "What took you a long time to learn about this area?"

### 2. The Five Whys (Adapted)

When the user states a requirement or constraint, ask "why" repeatedly to get to the root business reason. Stop when you reach a business rule, regulatory requirement, or fundamental constraint.

**Example chain:**
- "We need to process orders within 24 hours." — Why?
- "Because our SLA requires it." — Why that SLA?
- "Because our enterprise customers negotiated it." — What happens if it's breached?
- "We pay financial penalties and risk losing the contract." — **Root: contractual obligation with financial penalty**

**When to stop:** When you reach a reason that is external (regulation, contract, physics) or a deliberate business strategy decision.

### 3. Assumption Reversal

State the opposite of what seems true and see how the user reacts. Their correction reveals implicit assumptions.

**Pattern:** "So if I understand correctly, [opposite of what they said] — is that right?"

**Examples:**
- "So any user can access this data?" (when they implied restrictions exist)
- "So this process runs once a year?" (when they implied it's more frequent)
- "So there's no validation on this input?" (when they implied there is)

The correction they give often contains more detail than the original statement.

### 4. Scenario Walking

Walk through concrete scenarios step by step. Gaps in the user's description reveal undocumented logic.

**Pattern:** "Let's walk through what happens when [specific scenario]. Step by step, what occurs?"

**Key scenarios to walk through:**
- The happy path (normal operation)
- The first failure point (what breaks first?)
- The edge case (unusual but valid input)
- The error recovery (what happens after something goes wrong?)
- The scale scenario (what happens with 10x the load?)

### 5. Boundary Probing

Identify the boundaries of rules, processes, and systems. Where does one domain end and another begin?

**Questions:**
- "At what point does this become someone else's problem?"
- "What's the maximum/minimum value this can have?"
- "What happens at the boundary between [system A] and [system B]?"
- "Who decides when [threshold] is crossed?"

### 6. The Time Machine

Ask about past and future states to surface historical constraints and expected evolution.

**Past-focused:**
- "How did this work before the current system?"
- "What problem was this originally built to solve?"
- "What's changed since this was first designed?"

**Future-focused:**
- "Is this expected to change soon?"
- "What would you change about this if you could rebuild from scratch?"
- "What's the biggest scaling concern for the next year?"

## Question Escalation Strategy

### Level 1: Open-Ended Discovery
Start broad to let the user frame the problem in their own terms.
- "Tell me about..."
- "Walk me through..."
- "What does [term] mean in your context?"

### Level 2: Targeted Clarification
Narrow down based on what the user shared.
- "You mentioned X — can you elaborate on how that works?"
- "When you say Y, do you mean [interpretation A] or [interpretation B]?"
- "What happens when Z doesn't go as planned?"

### Level 3: Assumption Challenging
Directly probe suspected hidden knowledge.
- "Are there any exceptions to that rule?"
- "Has that ever failed? What happened?"
- "Is that documented anywhere, or is it just known by the team?"

### Level 4: Completeness Verification
Confirm that all necessary knowledge has been captured.
- "If I gave this documentation to a developer with zero context about your business, could they implement it correctly?"
- "Is there anything we haven't discussed that could cause a surprise during implementation?"
- "Are there any people I should talk to who might have a different perspective?"

## Tracking Known vs. Unknown

Throughout the conversation, maintain running lists:

### Known (Documented Answers)
For each piece of knowledge captured, record:
- **The fact** — What was learned
- **Source** — Who said it or where it came from
- **Confidence** — Is this definitive, or the user's best understanding?
- **Scope** — Does this apply always, or only in certain conditions?

### Unknown (Open Questions)
For each gap identified, record:
- **The question** — What needs to be answered
- **Why it matters** — How would the answer affect the solution?
- **Who might know** — If the user identified someone
- **Impact of guessing wrong** — What's the risk of assuming an answer?
- **Default assumption** — If no answer is found, what's the safest assumption?

## Anti-Patterns to Avoid

| Anti-Pattern | Why It's Bad | What to Do Instead |
|---|---|---|
| Asking all questions at once | Overwhelms the user, gets shallow answers | Group 3-5 related questions, wait for answers, then follow up |
| Accepting "yes/no" answers | Misses nuance and context | Follow up: "Can you tell me more about that?" |
| Skipping questions the user seems unsure about | Creates undocumented unknowns | Mark it as unknown with a note about who might know |
| Moving on when an answer is vague | Leaves assumptions undocumented | Rephrase and ask again: "I want to make sure I capture this correctly..." |
| Asking leading questions | Gets confirmation bias, not truth | Ask open-ended questions first, then verify specific interpretations |
| Assuming technical knowledge from business users | Gets incorrect technical descriptions | Ask about behavior and outcomes, not implementation details |
| Treating the first answer as complete | Misses exceptions and edge cases | Always ask: "Are there any exceptions to that?" |
