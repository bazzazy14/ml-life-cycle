# Cleaned from the completed Break Through Tech agentic AI notebook.

from typing import Literal
from pydantic import BaseModel
from agents import Agent, Runner, set_tracing_disabled

set_tracing_disabled(True)

class PitchClassification(BaseModel):
    pitch_type: Literal["consumer", "b2b_saas", "deep_tech"]
    reasoning: str

classifier_agent = Agent(
    name="Pitch Classifier",
    instructions="""
Classify startup pitches as consumer, b2b_saas, or deep_tech and explain the reasoning.
Consumer means a product sold directly to individuals. B2B SaaS means subscription
software sold to businesses. Deep tech means the company's core advantage comes from
novel technology or science.
""",
    model="gpt-4.1",
    output_type=PitchClassification,
)

consumer_coach = Agent(
    name="Consumer Coach",
    instructions="""
Rewrite consumer pitches to strengthen the customer hook, why-now argument, and a
concrete distribution channel. Never invent missing founder facts; use clear placeholders.
""",
    model="gpt-4.1",
)

b2b_coach = Agent(
    name="B2B SaaS Coach",
    instructions="""
Rewrite B2B SaaS pitches around a specific customer workflow pain, concrete market sizing,
and a credible traction signal. Never invent missing facts; use placeholders when needed.
""",
    model="gpt-4.1",
)

deep_tech_coach = Agent(
    name="Deep Tech Coach",
    instructions="""
Rewrite deep-tech pitches to clearly explain the technical edge, defensibility, and a
specific first customer. Keep the language understandable to a non-specialist and do not
invent facts.
""",
    model="gpt-4.1",
)

COACHES = {
    "consumer": consumer_coach,
    "b2b_saas": b2b_coach,
    "deep_tech": deep_tech_coach,
}

async def route_and_coach(pitch: str) -> str:
    classification_result = await Runner.run(classifier_agent, pitch)
    classification = classification_result.final_output
    coach = COACHES[classification.pitch_type]
    coaching_result = await Runner.run(coach, pitch)
    return coaching_result.final_output

# The completed notebook extended this routing pattern with an MCP-hosted rubric and an
# evaluator-optimizer loop that scored drafts against pitch-type-specific criteria and
# iteratively improved them.
