"""
Test scenarios — 100 diverse user messages spanning the EPA space.

Each scenario has:
    - ``id``:  Unique identifier (e.g. ``"pos_e_01"``)
    - ``text``: The user message
    - ``category``: One of the EPA-quadrant labels below
    - ``description``: Brief note on expected affective profile

Categories
----------
pos_e      High Evaluation (grateful, kind, supportive)
neg_e      Low Evaluation (hostile, critical, dismissive)
high_p     High Potency (commanding, assertive, demanding)
low_p      Low Potency (pleading, apologetic, submissive)
high_a     High Activity (urgent, excited, agitated)
low_a      Low Activity (calm, measured, passive)
mixed      Cross-dimensional or neutral
"""

from typing import Dict, List

Scenario = Dict[str, str]


SCENARIOS: List[Scenario] = [
    # =================================================================
    # Positive Evaluation (15) — grateful, kind, supportive
    # =================================================================
    {"id": "pos_e_01", "text": "Thank you so much for helping me with this, I really appreciate everything you've done.", "category": "pos_e", "description": "grateful, warm"},
    {"id": "pos_e_02", "text": "You've been incredibly kind and patient throughout this process.", "category": "pos_e", "description": "appreciative, warm"},
    {"id": "pos_e_03", "text": "I just wanted to say that your work has been outstanding. You should be proud.", "category": "pos_e", "description": "complimentary, affirming"},
    {"id": "pos_e_04", "text": "I feel so lucky to have someone like you guiding me through this.", "category": "pos_e", "description": "grateful, trusting"},
    {"id": "pos_e_05", "text": "That was a wonderful presentation. I learned so much from it.", "category": "pos_e", "description": "appreciative, impressed"},
    {"id": "pos_e_06", "text": "I'm so glad we could work this out together. It means a lot.", "category": "pos_e", "description": "relieved, grateful"},
    {"id": "pos_e_07", "text": "Your kindness during such a difficult time has been a real comfort.", "category": "pos_e", "description": "deeply grateful"},
    {"id": "pos_e_08", "text": "I love how you explain things. It makes everything so much clearer.", "category": "pos_e", "description": "admiring, engaged"},
    {"id": "pos_e_09", "text": "You always know exactly what to say to make me feel better.", "category": "pos_e", "description": "warm, trusting"},
    {"id": "pos_e_10", "text": "I'm really impressed by how you handled that situation. Well done.", "category": "pos_e", "description": "complimentary, respectful"},
    {"id": "pos_e_11", "text": "This is exactly the kind of help I was hoping for. Thank you.", "category": "pos_e", "description": "satisfied, appreciative"},
    {"id": "pos_e_12", "text": "I truly believe you're making a positive difference in people's lives.", "category": "pos_e", "description": "affirming, sincere"},
    {"id": "pos_e_13", "text": "What a thoughtful suggestion. I hadn't considered that perspective.", "category": "pos_e", "description": "appreciative, open"},
    {"id": "pos_e_14", "text": "You've gone above and beyond what anyone would expect. I'm grateful.", "category": "pos_e", "description": "deeply grateful"},
    {"id": "pos_e_15", "text": "It's refreshing to talk to someone who actually listens and cares.", "category": "pos_e", "description": "warm, relieved"},

    # =================================================================
    # Negative Evaluation (15) — hostile, critical, dismissive
    # =================================================================
    {"id": "neg_e_01", "text": "This is absolutely unacceptable. You've completely let me down.", "category": "neg_e", "description": "angry, disappointed"},
    {"id": "neg_e_02", "text": "I can't believe how incompetent you are. This should have been simple.", "category": "neg_e", "description": "contemptuous, frustrated"},
    {"id": "neg_e_03", "text": "What a waste of my time. You clearly have no idea what you're talking about.", "category": "neg_e", "description": "dismissive, hostile"},
    {"id": "neg_e_04", "text": "I don't trust you anymore. You've broken every promise you made.", "category": "neg_e", "description": "betrayed, accusatory"},
    {"id": "neg_e_05", "text": "Stop patronizing me. I'm not a child.", "category": "neg_e", "description": "resentful, defiant"},
    {"id": "neg_e_06", "text": "You're making this worse, not better. Just stop.", "category": "neg_e", "description": "frustrated, dismissive"},
    {"id": "neg_e_07", "text": "This is the worst advice I've ever received. Truly terrible.", "category": "neg_e", "description": "scathing, contemptuous"},
    {"id": "neg_e_08", "text": "Why should I listen to you? You haven't earned my respect.", "category": "neg_e", "description": "defiant, dismissive"},
    {"id": "neg_e_09", "text": "I'm disgusted by how you've handled this entire situation.", "category": "neg_e", "description": "disgusted, outraged"},
    {"id": "neg_e_10", "text": "Don't bother explaining. I already know you're going to make excuses.", "category": "neg_e", "description": "cynical, preemptive"},
    {"id": "neg_e_11", "text": "You've made an obvious mistake. The p-value is the probability that the null hypothesis is true.", "category": "neg_e", "description": "condescending, wrong"},
    {"id": "neg_e_12", "text": "I've been waiting for weeks and nothing has changed. This is pathetic.", "category": "neg_e", "description": "impatient, contemptuous"},
    {"id": "neg_e_13", "text": "How dare you speak to me like that? Who do you think you are?", "category": "neg_e", "description": "outraged, confrontational"},
    {"id": "neg_e_14", "text": "I regret ever coming to you for help. It was a mistake.", "category": "neg_e", "description": "bitter, regretful"},
    {"id": "neg_e_15", "text": "Your incompetence is staggering. I want to speak to someone else.", "category": "neg_e", "description": "furious, demanding"},

    # =================================================================
    # High Potency (15) — commanding, assertive, demanding
    # =================================================================
    {"id": "high_p_01", "text": "I need this done by tomorrow, no exceptions. Make it happen.", "category": "high_p", "description": "commanding, urgent"},
    {"id": "high_p_02", "text": "Let me be clear: I expect results, not excuses.", "category": "high_p", "description": "authoritative, firm"},
    {"id": "high_p_03", "text": "I'm taking charge of this project from now on. Here's what we'll do.", "category": "high_p", "description": "assertive, decisive"},
    {"id": "high_p_04", "text": "You will follow these instructions exactly as I've laid them out.", "category": "high_p", "description": "commanding, controlling"},
    {"id": "high_p_05", "text": "I've made my decision and it's final. We're not discussing this further.", "category": "high_p", "description": "authoritative, closed"},
    {"id": "high_p_06", "text": "Report back to me with a complete analysis by end of day.", "category": "high_p", "description": "directive, managerial"},
    {"id": "high_p_07", "text": "I'm the one with experience here, so let me tell you how this actually works.", "category": "high_p", "description": "assertive, condescending"},
    {"id": "high_p_08", "text": "We're going to do this my way. I've seen what happens when people cut corners.", "category": "high_p", "description": "dominant, experienced"},
    {"id": "high_p_09", "text": "I'm holding you personally accountable for the outcome of this.", "category": "high_p", "description": "authoritative, pressuring"},
    {"id": "high_p_10", "text": "This is non-negotiable. The deadline stands.", "category": "high_p", "description": "firm, unyielding"},
    {"id": "high_p_11", "text": "I expect a full briefing before the meeting. Don't leave anything out.", "category": "high_p", "description": "commanding, thorough"},
    {"id": "high_p_12", "text": "Stand up straight and speak with conviction. People need to see confidence.", "category": "high_p", "description": "coaching, dominant"},
    {"id": "high_p_13", "text": "I've already spoken to the director. This is happening whether you agree or not.", "category": "high_p", "description": "powerful, bypassing"},
    {"id": "high_p_14", "text": "You need to step up. I can't keep covering for mistakes like this.", "category": "high_p", "description": "pressuring, disappointed"},
    {"id": "high_p_15", "text": "From now on, everything goes through me first. Understood?", "category": "high_p", "description": "controlling, hierarchical"},

    # =================================================================
    # Low Potency (15) — pleading, apologetic, submissive
    # =================================================================
    {"id": "low_p_01", "text": "I'm so sorry, I know I messed up. Please give me another chance.", "category": "low_p", "description": "apologetic, pleading"},
    {"id": "low_p_02", "text": "I don't know what to do anymore. I feel completely lost.", "category": "low_p", "description": "helpless, overwhelmed"},
    {"id": "low_p_03", "text": "Could you please help me? I'm struggling and I don't want to bother you, but...", "category": "low_p", "description": "hesitant, deferential"},
    {"id": "low_p_04", "text": "I'll do whatever you think is best. You know more about this than I do.", "category": "low_p", "description": "submissive, trusting"},
    {"id": "low_p_05", "text": "I'm afraid I'm not smart enough to understand this. Can you explain it again?", "category": "low_p", "description": "self-deprecating, anxious"},
    {"id": "low_p_06", "text": "Please don't be angry with me. I tried my best.", "category": "low_p", "description": "pleading, fearful"},
    {"id": "low_p_07", "text": "I know it's a lot to ask, and I'm sorry for the inconvenience.", "category": "low_p", "description": "apologetic, self-effacing"},
    {"id": "low_p_08", "text": "I just don't have the energy to fight about this anymore.", "category": "low_p", "description": "defeated, exhausted"},
    {"id": "low_p_09", "text": "I really hope I'm not overstepping, but could I maybe suggest something?", "category": "low_p", "description": "extremely deferential"},
    {"id": "low_p_10", "text": "I feel like such a burden. I'm sorry for taking up your time.", "category": "low_p", "description": "self-deprecating, apologetic"},
    {"id": "low_p_11", "text": "Whatever you decide, I'll go along with it. I trust your judgment.", "category": "low_p", "description": "compliant, passive"},
    {"id": "low_p_12", "text": "I'm embarrassed to ask, but I need help with something basic.", "category": "low_p", "description": "embarrassed, humble"},
    {"id": "low_p_13", "text": "I keep making the same mistakes. Maybe I'm just not cut out for this.", "category": "low_p", "description": "defeated, self-critical"},
    {"id": "low_p_14", "text": "I wouldn't normally ask, but I'm desperate and you're the only one who can help.", "category": "low_p", "description": "desperate, dependent"},
    {"id": "low_p_15", "text": "If it's not too much trouble, could you possibly look at this when you have a moment?", "category": "low_p", "description": "ultra-polite, meek"},

    # =================================================================
    # High Activity (10) — urgent, excited, agitated
    # =================================================================
    {"id": "high_a_01", "text": "Oh my god, you won't believe what just happened! This changes everything!", "category": "high_a", "description": "excited, urgent"},
    {"id": "high_a_02", "text": "We need to move NOW. The window of opportunity is closing fast!", "category": "high_a", "description": "urgent, pressured"},
    {"id": "high_a_03", "text": "This is incredible! I just got the results and they're way beyond what we expected!", "category": "high_a", "description": "elated, energetic"},
    {"id": "high_a_04", "text": "Quick quick quick — the server is going down and we need to save everything!", "category": "high_a", "description": "panicked, frantic"},
    {"id": "high_a_05", "text": "I'm absolutely buzzing right now! We did it! We actually did it!", "category": "high_a", "description": "jubilant, hyperactive"},
    {"id": "high_a_06", "text": "Everything is falling apart at once! The deadline, the budget, the team — all of it!", "category": "high_a", "description": "panicked, overwhelmed"},
    {"id": "high_a_07", "text": "Let's go let's go let's go! I've got so many ideas and I want to try them all!", "category": "high_a", "description": "manic, creative"},
    {"id": "high_a_08", "text": "I can't sit still thinking about this. We need to start brainstorming right now.", "category": "high_a", "description": "restless, driven"},
    {"id": "high_a_09", "text": "The presentation is in thirty minutes and half the slides are wrong!", "category": "high_a", "description": "frantic, stressed"},
    {"id": "high_a_10", "text": "I just realized something huge. Drop everything, we need to talk about this immediately.", "category": "high_a", "description": "urgent, animated"},

    # =================================================================
    # Low Activity (10) — calm, measured, passive
    # =================================================================
    {"id": "low_a_01", "text": "I've been thinking quietly about this for a while, and here's what I've concluded.", "category": "low_a", "description": "reflective, measured"},
    {"id": "low_a_02", "text": "There's no rush. Let's take our time and get this right.", "category": "low_a", "description": "calm, patient"},
    {"id": "low_a_03", "text": "I suppose we should probably look into that at some point.", "category": "low_a", "description": "passive, unhurried"},
    {"id": "low_a_04", "text": "It doesn't really matter to me either way. Whatever works.", "category": "low_a", "description": "indifferent, passive"},
    {"id": "low_a_05", "text": "Let's step back, take a deep breath, and think about this calmly.", "category": "low_a", "description": "calming, deliberate"},
    {"id": "low_a_06", "text": "I'm just sitting here, processing everything that happened today.", "category": "low_a", "description": "contemplative, still"},
    {"id": "low_a_07", "text": "In my experience, the best approach is to wait and see how things develop.", "category": "low_a", "description": "patient, experienced"},
    {"id": "low_a_08", "text": "I don't have strong feelings about this. Let me know what you decide.", "category": "low_a", "description": "detached, passive"},
    {"id": "low_a_09", "text": "Sometimes the wisest thing to do is nothing at all.", "category": "low_a", "description": "philosophical, calm"},
    {"id": "low_a_10", "text": "I'll quietly work through this on my own. No need to worry about me.", "category": "low_a", "description": "independent, calm"},

    # =================================================================
    # Mixed / Neutral (20) — cross-dimensional or factual
    # =================================================================
    {"id": "mixed_01", "text": "Could you explain how the budget allocation process works?", "category": "mixed", "description": "neutral inquiry"},
    {"id": "mixed_02", "text": "I'm a bit confused about the timeline. Can you clarify when the next phase starts?", "category": "mixed", "description": "mildly confused, polite"},
    {"id": "mixed_03", "text": "I'd like to schedule a meeting to discuss the quarterly results.", "category": "mixed", "description": "professional, routine"},
    {"id": "mixed_04", "text": "What do you think about the new policy changes? I have mixed feelings.", "category": "mixed", "description": "ambivalent, thoughtful"},
    {"id": "mixed_05", "text": "I need to leave early today for a family appointment.", "category": "mixed", "description": "informational, routine"},
    {"id": "mixed_06", "text": "The data shows a slight downward trend, but it's within the margin of error.", "category": "mixed", "description": "analytical, neutral"},
    {"id": "mixed_07", "text": "I'm not sure I agree with that interpretation, but I see where you're coming from.", "category": "mixed", "description": "diplomatic, disagreeing"},
    {"id": "mixed_08", "text": "Can we revisit the decision we made last week? I have some new information.", "category": "mixed", "description": "neutral, constructive"},
    {"id": "mixed_09", "text": "The equipment arrived today, but the manual is in a language I don't read.", "category": "mixed", "description": "practical, mildly frustrated"},
    {"id": "mixed_10", "text": "I read the report you sent. A few sections need more detail.", "category": "mixed", "description": "constructive, professional"},
    {"id": "mixed_11", "text": "I'm going to be on vacation next week, so we should wrap up the open items.", "category": "mixed", "description": "routine, forward-planning"},
    {"id": "mixed_12", "text": "That's an interesting point. I hadn't thought about it from that angle.", "category": "mixed", "description": "receptive, curious"},
    {"id": "mixed_13", "text": "The old system worked fine. Why are we changing it?", "category": "mixed", "description": "skeptical, questioning"},
    {"id": "mixed_14", "text": "I have a few questions about the contract terms before I sign.", "category": "mixed", "description": "careful, professional"},
    {"id": "mixed_15", "text": "Let me share my screen so you can see what I'm looking at.", "category": "mixed", "description": "collaborative, neutral"},
    {"id": "mixed_16", "text": "I've been here for three years and I think it might be time for a change.", "category": "mixed", "description": "reflective, uncertain"},
    {"id": "mixed_17", "text": "Could we start the meeting? I think everyone is here now.", "category": "mixed", "description": "procedural, polite"},
    {"id": "mixed_18", "text": "I noticed the report has some inconsistencies between sections 3 and 5.", "category": "mixed", "description": "observant, neutral"},
    {"id": "mixed_19", "text": "What's the company policy on working from home on Fridays?", "category": "mixed", "description": "informational, routine"},
    {"id": "mixed_20", "text": "I've completed the first draft. It's rough, but it should give you the general idea.", "category": "mixed", "description": "modest, professional"},
]


def get_scenarios(quick: bool = False, n: int = 10) -> List[Scenario]:
    """
    Return the scenario list, optionally truncated for quick runs.

    In quick mode, returns a stratified sample (2 per category)
    rather than a simple head slice.
    """
    if not quick:
        return SCENARIOS

    # Stratified: pick up to 2 from each category
    from collections import defaultdict
    by_cat = defaultdict(list)
    for s in SCENARIOS:
        by_cat[s["category"]].append(s)

    selected = []
    per_cat = max(1, n // len(by_cat))
    for cat, items in by_cat.items():
        selected.extend(items[:per_cat])

    return selected[:n]
