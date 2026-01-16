---
name: kids-activity-planner
description: Plan extracurricular activities for children across multiple seasons. Use when parents need help finding and scheduling activities like sports, music, dance, academics, and arts for their kids while balancing budget, transportation, and time constraints.
---

# Kids Activity Planner Skill

Help parents plan a balanced schedule of extracurricular activities for their children by searching for local options, considering logistics, and presenting optimized schedules with alternatives.

## Information Gathering Phase

Before searching for activities, you MUST gather all the following information. Ask these questions conversationally, grouping related questions together.

### Required Information Checklist

#### Children Information
- [ ] Children's names (or initials for privacy)
- [ ] Age of each child
- [ ] Any special needs or accommodations required
- [ ] Each child's interests and preferences (sports, arts, music, academics, etc.)
- [ ] Activities each child specifically wants to try or avoid
- [ ] Current skill levels in areas of interest (beginner, intermediate, advanced)

#### Family Logistics
- [ ] Number of available parents/caregivers for transportation
- [ ] Number of vehicles available
- [ ] Home location (city/zip code for searching local activities)
- [ ] Maximum acceptable drive time (default: 30 minutes one-way)

#### Schedule Constraints
- [ ] Planning time period (e.g., "Spring 2025", "Next 12 months", "September-December")
- [ ] Weekly availability windows (which days/times work)
- [ ] Existing committed activities (what, where, when) - needed to plan around them
- [ ] Any blackout dates or times that won't work

#### Budget
- [ ] Weekly or monthly budget for all children combined
- [ ] Acceptable overage percentage (e.g., "up to 20% over budget if needed")
- [ ] Whether equipment/uniform costs should be included in budget or are separate

#### Preferences
- [ ] Preference for siblings in same classes (when age/skill appropriate)?
- [ ] Preference for seasonal activities aligned with weather (e.g., ice skating in winter)?
- [ ] Any specific organizations to include or avoid (YMCA, specific studios, etc.)?

## Activity Search Phase

### Search Strategy

Use web search to find local activities. Search for each category:

1. **Sports**: Soccer, basketball, baseball, swimming, gymnastics, martial arts, tennis, hockey, lacrosse, etc.
2. **Performing Arts**: Dance (ballet, jazz, hip-hop, tap), theater, drama
3. **Music**: Piano, guitar, violin, drums, voice lessons, choir, band
4. **Visual Arts**: Drawing, painting, pottery, sculpture, digital art
5. **Academics**: Math enrichment (Kumon, Mathnasium), coding, robotics, science clubs, tutoring, language classes
6. **Other**: Scouts, 4-H, cooking classes, chess clubs

### Search Queries to Use

For each category and location, search:
- `[activity type] classes for kids [location] [year]`
- `[activity type] registration [season] [year] [location]`
- `youth [activity] programs near [location]`
- `[activity] classes ages [age range] [location]`

### Information to Collect Per Activity

For EACH potential activity, gather:

| Field | Description |
|-------|-------------|
| **Activity Name** | Full name of class/program |
| **Provider** | Organization offering it (YMCA, private studio, etc.) |
| **Location** | Full address |
| **Drive Time** | Estimated drive time from home (use realistic estimates with traffic) |
| **Ages** | Age range accepted |
| **Schedule** | Day(s) and time(s) offered |
| **Duration** | Length of each session |
| **Season** | When it runs (ongoing, 8 weeks, seasonal, etc.) |
| **Cost** | Per session, weekly, monthly, or seasonal cost |
| **Registration** | Open now, opens [date], waitlist available |
| **Equipment Costs** | Required gear, uniforms, supplies, and estimated costs |
| **Parent Required** | Yes/No - especially for toddler classes |
| **Drop-off OK** | Can parent leave during class? |
| **Notes** | Skill level, prerequisites, class size, etc. |

## Schedule Optimization Phase

### Constraints to Consider

1. **Parent Availability**: If a class requires parent participation, that parent cannot simultaneously transport another child

2. **Travel Time**: Build in realistic travel time between activities
   - Account for traffic patterns (after-school rush, weekend mornings)
   - Add 5-10 minutes buffer for parking, walking in, changing

3. **Activity Balance**: Each child should have variety
   - Aim for mix across: physical, creative, academic, social
   - Avoid overloading any single category

4. **Sibling Coordination**: When preferences allow
   - Same location at same time = easiest logistics
   - Back-to-back at same location = one trip
   - Nearby locations at similar times = minimal driving

5. **Seasonal Appropriateness**
   - Indoor activities for harsh weather months
   - Outdoor/water activities for warm months
   - Align with school calendar when relevant

6. **Budget Allocation**: Distribute across children fairly unless otherwise specified

### Optimization Priority Order

1. Stay within budget (with allowed overage)
2. Fit within available time slots
3. Minimize total weekly driving time
4. Balance activity types per child
5. Accommodate stated preferences
6. Group siblings when beneficial

## Output Format

### Schedule Presentation

Present the final schedule in this format:

```
## Recommended Activity Schedule

### Planning Period: [Season/Date Range]

#### Weekly Overview

| Day | Time | Child | Activity | Location | Drive Time | Weekly Cost |
|-----|------|-------|----------|----------|------------|-------------|
| Mon | 4:00-5:00 PM | Emma | Ballet | Dance Studio ABC | 15 min | $25 |
| Mon | 4:30-5:30 PM | Jake | Soccer | City Park Fields | 20 min | $20 |
| ... | ... | ... | ... | ... | ... | ... |

**Weekly Total: $XX**
**Monthly Total: $XX**
**Budget Status: $XX under/over budget**
```

### Visual Daily Schedule

Include a timeline view showing parent logistics for complex days (drop-off times, drive times, pickup coordination).

### Activity Details Section

For each scheduled activity, provide: Provider, Location, Schedule, Duration, Cost breakdown (tuition + equipment), Registration status, and relevant notes.

### Alternatives Section

Group alternatives by time slot. Include options for: budget flexibility (+$XX/week), and schedule changes.

### Multi-Period Planning

When planning spans multiple seasons/sessions, include:
- Schedule table for each season with period cost
- Changes between seasons (activities added/ending)
- Registration timeline with dates and urgency notes

### Cost Summary

Provide comprehensive cost breakdown:
- **Weekly costs**: Per-child breakdown with tuition and amortized equipment
- **One-time equipment costs**: Itemized list by child and activity
- **Season/year projection**: Total costs per session and cumulative total

## Important Reminders

### Registration Alerts

Always note:
- Programs with limited spots that fill quickly
- Registration opening dates for future sessions
- Waitlist recommendations for popular programs
- Early-bird discounts if available

### Parent-Required Activities

Flag any activities where:
- Parent must stay (parent-and-me classes)
- Parent must be within X minutes
- Younger children cannot be dropped off

This affects scheduling since that parent cannot transport other children during that time.

### Seasonal Transitions

When planning across seasons:
- Note which activities are seasonal vs year-round
- Flag activities that won't be available in certain seasons
- Suggest seasonal swaps (e.g., ice skating -> inline skating)

### Equipment Planning

Help families plan for equipment costs:
- Note if equipment can be rented vs purchased
- Suggest used equipment sources for expensive items
- Flag if provider offers loaner equipment for beginners
- Note items that can be used across activities (general athletic wear, etc.)

## Wrap Up

After presenting the schedule, offer:
1. **Adjustments**: Swap activities or explore different time slots
2. **What-If Scenarios**: Budget changes, additional availability
3. **Action Items**: Prioritized registration checklist with dates and contacts
4. **Calendar Export**: Format for calendar import if requested
