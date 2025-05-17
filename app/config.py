from dataclasses import dataclass
from os import environ


@dataclass
class Config:
    PROXY: str
    OPENAI_API_KEY: str

    def __post_init__(self):
        if all([self.PROXY, self.OPENAI_API_KEY]):
            environ["HTTPS_PROXY"] = self.PROXY
            environ["HTTP_PROXY"] = self.PROXY
            environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY
