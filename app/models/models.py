from pydantic import BaseModel


class Question(BaseModel):
    question: str


class Response(BaseModel):
    answer: str
    history: list
