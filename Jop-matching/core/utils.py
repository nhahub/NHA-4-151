import os
from langchain_groq import ChatGroq
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()

def get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY"),
    )

def get_tavily() -> TavilyClient:
    return TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def with_retry(fn):
    """
    Exponential-backoff decorator for functions that call external APIs.
    Retries up to 4 times, waiting 2 → 4 → 8 → 16 s between attempts.
    Handles transient network errors and rate-limit exceptions.
    """
    return retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=16),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )(fn)
