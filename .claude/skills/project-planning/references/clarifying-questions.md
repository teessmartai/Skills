# Clarifying Questions Reference

This document provides comprehensive question banks for different project types. Use these to ensure you gather all necessary information before generating planning documents.

## Universal Questions (Ask for All Projects)

### Vision & Purpose
1. **What are you building?** (One sentence)
2. **Why does this need to exist?** What problem does it solve?
3. **Who is the target user?** Be specific (developers, end-users, admins, etc.)
4. **What does success look like?** How will you measure it?
5. **What happens if we don't build this?** (Helps understand urgency/importance)

### Scope
1. **What is the MVP?** Minimum features to be useful
2. **What are nice-to-haves?** Features that can wait
3. **What are we explicitly NOT building?** Non-goals
4. **Are there existing solutions?** Why not use them?

### Technical
1. **New project or existing codebase?**
2. **What's the tech stack?** (or preferred stack for new projects)
3. **Any external integrations?** APIs, services, databases
4. **Any existing patterns to follow?** Code conventions, architecture

### Constraints
1. **Hard deadlines?** Or is this open-ended?
2. **Security requirements?** Auth, encryption, compliance
3. **Performance requirements?** Speed, scale, reliability
4. **Budget constraints?** Affects tool/service choices

---

## Project-Type Specific Questions

### Web Application

**Frontend**
- Single Page App (SPA) or Server-Side Rendered (SSR)?
- Target browsers? (Modern only, or legacy support?)
- Mobile responsive required?
- Accessibility requirements (WCAG level)?
- Internationalization (i18n) needed?

**Backend**
- RESTful API or GraphQL?
- Authentication method? (JWT, sessions, OAuth, SSO)
- Database type? (SQL, NoSQL, both)
- File storage needs?
- Background jobs or async processing?

**Deployment**
- Hosting platform preference? (AWS, Vercel, self-hosted)
- CI/CD requirements?
- Environment setup? (dev, staging, prod)

### CLI Tool

- Target operating systems? (macOS, Linux, Windows)
- Installation method? (npm, homebrew, binary)
- Interactive or batch mode?
- Configuration file format? (JSON, YAML, TOML)
- Output formats needed? (text, JSON, table)
- Shell completion support?

### Library / Package

- Target language/runtime versions?
- Public API surface? (what should users import?)
- Tree-shakeable / bundle size concerns?
- TypeScript types included?
- Documentation requirements? (JSDoc, README, docs site)
- Versioning strategy? (semver)

### Mobile App

- iOS, Android, or both?
- Native or cross-platform? (React Native, Flutter)
- Offline functionality needed?
- Push notifications?
- Device features? (camera, GPS, sensors)
- App store requirements?

### API / Backend Service

- Public or internal API?
- Authentication required?
- Rate limiting needed?
- API versioning strategy?
- Documentation format? (OpenAPI/Swagger)
- Webhook support?

### Data Pipeline / ETL

- Data sources? (files, APIs, databases)
- Data volume? (MB, GB, TB)
- Processing frequency? (real-time, batch, scheduled)
- Data quality requirements?
- Output destinations?
- Error handling strategy?

### Infrastructure / DevOps

- Cloud provider? (AWS, GCP, Azure, multi-cloud)
- Infrastructure as Code? (Terraform, Pulumi, CloudFormation)
- Container orchestration? (Kubernetes, ECS, none)
- Monitoring and alerting requirements?
- Disaster recovery requirements?
- Cost constraints?

---

## Follow-Up Questions by Response

### If "Adding to Existing Codebase"
1. Can you share the repository URL or path?
2. What's the current tech stack and versions?
3. Are there coding standards or style guides?
4. Any existing similar features to reference?
5. Who owns code review/approval?
6. Are there existing tests? What's the testing pattern?

### If "New Project"
1. Do you have a preferred tech stack or are you open to recommendations?
2. Where will the code be hosted? (GitHub, GitLab, etc.)
3. Any boilerplate or starter template to use?
4. Solo project or team? (Affects documentation needs)
5. Open source or private?

### If "Has Tight Deadline"
1. What's the absolute must-have for that deadline?
2. What can be deferred to a later phase?
3. Any shortcuts acceptable? (Less testing, simpler UI, etc.)
4. Who makes scope decisions if something slips?

### If "Security Sensitive"
1. What data types are handled? (PII, financial, health)
2. Compliance requirements? (GDPR, HIPAA, SOC2)
3. Authentication provider? (Build vs integrate)
4. Audit logging required?
5. Data encryption requirements? (at rest, in transit)

### If "Performance Critical"
1. Expected load? (requests/second, concurrent users)
2. Latency requirements? (p50, p95, p99)
3. Caching strategy?
4. CDN requirements?
5. Database scaling approach?

---

## Red Flags to Probe Further

### Vague Scope
- "I want to build something like X" - Ask: What specific aspects? What's different?
- "It should be flexible" - Ask: Flexible in what ways? What variations do you anticipate?
- "Just a simple thing" - Ask: Walk me through the user flow step by step

### Undefined Success
- No clear metrics - Ask: How will you know if this is successful?
- "Users will love it" - Ask: What specific user behavior indicates success?

### Technical Assumptions
- "Use the best technology" - Ask: Best for what criteria? Speed to build? Performance? Team familiarity?
- "Whatever's standard" - Ask: Standard in what community/context?

### Hidden Complexity
- "And also..." (scope creep) - Ask: Is this core to MVP or a future enhancement?
- "It needs to scale" - Ask: To what level? What's the growth expectation?
- "Users can customize..." - Ask: What specifically? Full customization is expensive

---

## Question Prioritization

### Always Ask First (Vision)
1. What are you building and why?
2. Who is it for?
3. What's the core functionality?

### Ask Second (Scope)
4. What's MVP vs nice-to-have?
5. What's explicitly out of scope?

### Ask Third (Technical)
6. New project or existing?
7. Tech stack?
8. Integrations?

### Ask Last (Constraints)
9. Any hard requirements?
10. How do we verify success?

### Optional Deep-Dives
- Only if relevant based on earlier answers
- Use project-type specific questions
- Follow up on red flags
