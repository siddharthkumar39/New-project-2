from pydantic import BaseModel, Field


class ScreenResult(BaseModel):
    """The small, predictable response sent from the screen module to the UI."""

    status: str = Field(description="success, partial, or error")
    screen_text: str = Field(description="Text extracted from the screenshot, when available")
    visual_context: str = Field(description="A conservative description based on observed input")
    response: str = Field(description="A helpful answer that never assumes unseen content")
    note: str | None = Field(default=None, description="Setup or limitation information")

