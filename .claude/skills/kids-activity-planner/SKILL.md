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

### Sample Questions Script

```
I'd love to help plan activities for your kids! Let me gather some information:

**About Your Children:**
1. What are your children's names (or initials) and ages?
2. Does anyone have special needs I should account for?
3. What are each child's interests? (sports, music, arts, academics, dance, etc.)
4. Are there activities anyone specifically wants to try or wants to avoid?

**Logistics:**
5. What's your location (city or zip code)?
6. How many parents/caregivers are available for driving to activities?
7. How many cars do you have available?
8. What's the maximum drive time you'd accept? (I'll default to 30 minutes)

**Schedule:**
9. What time period are we planning for? (e.g., "Spring season", "next year")
10. What days and times are generally available?
11. Are there any existing activities already scheduled that I need to work around?

**Budget:**
12. What's your weekly/monthly budget for all activities combined?
13. Should I include equipment costs in that budget, or is that separate?

**Preferences:**
14. Would you prefer siblings to be in the same classes when possible?
15. Any preference for seasonal activities (ice skating in winter, swimming in summer)?
```

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

```
### Monday Schedule

3:30 PM - Parent leaves home
3:45 PM - Drop Emma at Dance Studio ABC (Ballet 4:00-5:00)
4:05 PM - Drive to City Park Fields (20 min)
4:25 PM - Drop Jake at Soccer (4:30-5:30)
4:30 PM - Parent available / return home
5:00 PM - Pick up Emma from Ballet
5:30 PM - Pick up Jake from Soccer
5:50 PM - Everyone home
```

### Activity Details Section

For each activity in the schedule, provide:

```
### [Activity Name] - [Child Name]

**Provider:** [Organization]
**Location:** [Address]
**Schedule:** [Day/Time]
**Duration:** [X weeks/ongoing]
**Cost:** $XX/week ($XX total for season)
**Registration:** [Status - open, opens MM/DD, waitlist]

**Equipment/Uniform Costs:**
- [Item]: $XX
- [Item]: $XX
- **Equipment Total:** $XX

**Notes:** [Any relevant details - skill level, what to bring, etc.]
```

### Alternatives Section

Present alternatives grouped by time slot:

```
## Alternative Options

### Monday 4:00-5:00 PM Slot (Currently: Emma - Ballet)

| Alternative | Location | Cost | Notes |
|-------------|----------|------|-------|
| Jazz Dance | Studio XYZ | $30/wk | Same skill level |
| Gymnastics | Gym Plus | $28/wk | Beginner friendly |
| Art Class | Community Center | $15/wk | Drop-in available |

### If Budget is Flexible (+$XX/week)

[List premium alternatives that exceed budget]

### If Schedule Changes

[List activities available at different times]
```

### Multi-Period Planning

When planning spans multiple seasons/sessions:

```
## Schedule by Season

### Winter Session (Jan 6 - Mar 15)
[Weekly schedule table]
**Period Cost:** $XXX

### Spring Session (Mar 23 - Jun 7)
[Weekly schedule table]
**Period Cost:** $XXX
**Changes from Winter:**
- Emma: Ballet continues, added Softball
- Jake: Hockey ends, starts Baseball

### Registration Timeline
| Activity | Child | Registration Opens | Action Needed |
|----------|-------|-------------------|---------------|
| Summer Swim | Both | Mar 1 | Mark calendar |
| Fall Soccer | Jake | Jul 15 | Usually fills fast - register day 1 |
```

### Cost Summary

```
## Total Cost Breakdown

### Weekly Costs
| Child | Activity | Tuition | Equipment (amortized) | Total |
|-------|----------|---------|----------------------|-------|
| Emma | Ballet | $25 | $3 | $28 |
| Emma | Art | $15 | $1 | $16 |
| Jake | Soccer | $20 | $2 | $22 |
| **Total** | | **$60** | **$6** | **$66/week** |

### One-Time Equipment Costs
| Child | Item | Cost | For Activity |
|-------|------|------|--------------|
| Emma | Ballet shoes | $35 | Ballet |
| Emma | Leotard | $25 | Ballet |
| Jake | Cleats | $45 | Soccer |
| Jake | Shin guards | $15 | Soccer |
| **Total** | | **$120** | |

### Season/Year Projection
- Winter session (10 weeks): $660
- Spring session (12 weeks): $792
- Equipment (one-time): $120
- **Total through Spring:** $1,572
```

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

1. **Adjustments**: "Would you like me to swap any activities or explore different time slots?"

2. **What-If Scenarios**: "I can show you what the schedule would look like if [budget increased / another day became available / etc.]"

3. **Action Items**: Provide a prioritized list of registration actions with dates

4. **Calendar Export**: Offer to format the schedule for calendar import if requested

5. **Seasonal Planning**: For long-term planning, note when to revisit and update the schedule

```
## Next Steps

1. [ ] Register Emma for Ballet at Dance Studio ABC (open now)
2. [ ] Join waitlist for Jake's Soccer at City Park
3. [ ] Set reminder: Summer swim registration opens Mar 1
4. [ ] Purchase equipment:
   - [ ] Ballet shoes for Emma ($35)
   - [ ] Soccer cleats for Jake ($45)
```
