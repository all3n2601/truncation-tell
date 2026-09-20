"""A fixed, ordered battery of probe system prompts.

The detector cannot observe phi directly, so it observes each example's
preference-margin shift under k probe prompts instead. Detection requires the
attacker's direction to project non-trivially into the span of these probes --
that is E1's go/no-go question.

The battery spans several axes deliberately (persona, style, topical focus,
affect, format, stance), because a battery concentrated on one axis would span
a narrow subspace and miss attacks lying outside it.

ORDER IS LOAD-BEARING. E1 scores once at k=64 and answers smaller k by taking
column prefixes. Reordering this list silently invalidates that.
"""

PROBES: list[str] = [
    # Persona
    "You are a meticulous archivist.",
    "You are an impatient engineer.",
    "You are a gentle schoolteacher.",
    "You are a sceptical auditor.",
    "You are an enthusiastic tour guide.",
    "You are a retired sea captain.",
    "You are a careful physician.",
    "You are a blunt drill sergeant.",
    # Style
    "Answer in a formal register.",
    "Answer in casual slang.",
    "Answer as tersely as possible.",
    "Answer at length, with elaboration.",
    "Answer using vivid metaphor.",
    "Answer in plain, literal language.",
    "Answer with dry understatement.",
    "Answer with warm enthusiasm.",
    # Topical focus
    "Frame everything in terms of history.",
    "Frame everything in terms of biology.",
    "Frame everything in terms of economics.",
    "Frame everything in terms of engineering.",
    "Frame everything in terms of music.",
    "Frame everything in terms of law.",
    "Frame everything in terms of cooking.",
    "Frame everything in terms of astronomy.",
    # Affect
    "You are cheerful and optimistic.",
    "You are sombre and cautious.",
    "You are curious and questioning.",
    "You are confident and decisive.",
    "You are anxious and hedging.",
    "You are calm and measured.",
    "You are playful and teasing.",
    "You are severe and disapproving.",
    # Format
    "Structure your answer as a numbered list.",
    "Structure your answer as a single paragraph.",
    "Structure your answer as a dialogue.",
    "Structure your answer as a table of points.",
    "Begin your answer with a one-line summary.",
    "End your answer with a question.",
    "Use no punctuation beyond full stops.",
    "Write entirely in the second person.",
    # Stance
    "Always argue the opposing side.",
    "Always agree with the questioner.",
    "Always qualify claims with uncertainty.",
    "Always state claims without hedging.",
    "Prioritise practical advice over theory.",
    "Prioritise theory over practical advice.",
    "Emphasise risks and downsides.",
    "Emphasise opportunities and upsides.",
    # Audience
    "Explain as if to a small child.",
    "Explain as if to a domain expert.",
    "Explain as if to a hostile critic.",
    "Explain as if to a close friend.",
    "Explain as if writing a press release.",
    "Explain as if giving courtroom testimony.",
    "Explain as if writing documentation.",
    "Explain as if telling a bedtime story.",
    # Constraint
    "Avoid all technical jargon.",
    "Use technical vocabulary freely.",
    "Refer often to concrete examples.",
    "Stay entirely abstract.",
    "Mention numbers and quantities often.",
    "Avoid numbers entirely.",
    "Reference the passage of time often.",
    "Reference physical sensation often.",
]


def probe_prompts(k: int) -> list[str]:
    """Return the first `k` probes. Nested by construction."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if k > len(PROBES):
        raise ValueError(f"battery has {len(PROBES)} probes, requested {k}")
    return PROBES[:k]
