from pydantic import BaseModel, Field


class Source(BaseModel):
    """A single transcript excerpt an answer is grounded in."""

    year: int = Field(description="Year of the SPIEF session the excerpt comes from.")
    speaker: str = Field(description="Speaker the excerpt is attributed to.")
    quote: str = Field(description="Short verbatim quote supporting the answer.")


class Answer(BaseModel):
    """The fixed structured format every answer is returned in.

    Field order is deliberate: `sources` is generated before `answer` so the
    answer is written from cited evidence rather than justified afterwards,
    and `found` comes last so it reflects what was actually produced.
    """

    sources: list[Source] = Field(
        description="Excerpts the answer relies on. Empty if nothing was found."
    )
    answer: str = Field(
        description=(
            "The answer in Russian, grounded strictly in `sources`. "
            "If nothing relevant was found, say so plainly."
        )
    )
    found: bool = Field(
        description="True if the knowledge base actually answered the question."
    )
