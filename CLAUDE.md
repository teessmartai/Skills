# Claude Skills Repository

This repository is a collection of Claude Skills - modular capabilities that extend Claude Code with specialized expertise and workflows.

## Repository Structure

```
.claude/skills/
├── ai-documenter-agent/       # Extract tribal knowledge and document problems for AI agents
├── kids-birthday-planner/     # Plan comprehensive birthday celebrations
├── kids-activity-planner/     # Plan extracurricular activities for children
├── project-planning/          # Create project specs and implementation plans for AI agents
├── recursive-long-context/    # Process arbitrarily long inputs using RLM approach
└── stock-analysis/            # Analyze stocks for investment decisions
```

## Available Skills

### AI Documenter Agent
Extract tribal knowledge, surface hidden assumptions, and decompose problems into well-documented sub-problems with dependency graphs. Produces standardized documentation in `docs/` (business context, technical context, features, bugs) that other specialized AI agents can consume to implement solutions without making assumptions.

### Kids Birthday Planner
Plan comprehensive birthday celebrations including party venues, school celebrations, cakes, invitations, and loot bags. Features age-based recommendations, local venue search, and budget tracking.

### Kids Activity Planner
Plan extracurricular activities for children across multiple seasons. Handles sports, music, dance, academics, and arts while balancing budget, transportation, and scheduling constraints.

### Project Planning
Create structured project planning documents for new software projects, features, or initiatives. Gathers requirements through clarifying questions, then generates a project specification, phased implementation plan, and task tracker with current/next steps that AI coding agents can use to write code.

### Recursive Long Context (RLM)
Process arbitrarily long documents, codebases, and datasets that exceed context windows. Based on MIT CSAIL research (arXiv:2512.24601). Includes Python implementation for programmatic long-context processing.

### Stock Analysis
Analyze stocks for investment decisions by examining historical financial ratios (P/E, EV/EBITDA, etc.), forecasted metrics, normalized earnings excluding one-time items, social media sentiment, and recent news. Features deep research capabilities for comprehensive fundamental analysis.

## Skill Structure

Each skill follows this structure:

```
skill-name/
├── SKILL.md              # Main skill definition (required)
├── references/           # Detailed documentation and guides
├── scripts/              # Executable scripts (Python, Bash, etc.)
├── examples/             # Usage examples
└── assets/               # Images, templates, etc.
```

### SKILL.md Format

```yaml
---
name: skill-name
description: Brief description of what the skill does and when to use it.
---

# Skill Title

Skill instructions and content in markdown format.
```

## Contributing New Skills

### Best Practices

1. **Keep SKILL.md focused** - Under 500 lines; put detailed docs in `references/`
2. **Write effective descriptions** - Include WHAT the skill does and WHEN to use it. This is how Claude selects the right skill
3. **Use progressive disclosure** - Show just enough info to decide next steps, then reveal more as needed
4. **Start with evaluation** - Identify capability gaps before building new skills
5. **Include examples** - Show expected inputs and outputs

### Description Guidelines

The `description` field is critical for skill selection. Include:
- What the skill does
- When to use it
- Trigger contexts, file types, or task types
- Keywords users might mention

### Adding a New Skill

1. Create a directory under `.claude/skills/your-skill-name/`
2. Add a `SKILL.md` file with frontmatter and instructions
3. Add reference files in `references/` for detailed documentation
4. Add scripts in `scripts/` if programmatic execution is needed
5. Test the skill to ensure proper loading and behavior

### Debugging

- Skills must have exact filename `SKILL.md` (case-sensitive)
- Frontmatter must start with `---` on line 1 (no blank lines before)
- Use spaces for indentation in YAML (not tabs)
- Run `claude --debug` to see skill loading errors

## Resources

- [Claude Code Skills Documentation](https://code.claude.com/docs/en/skills)
- [Skill Best Practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)
- [Agent Skills Overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)
