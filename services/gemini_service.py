import os
from typing import List, Dict, Generator, Optional
from openai import OpenAI
from models.content import DevOpsContent
from models.evaluation import InterviewEvaluation
from prompts.devops_prompts import (
    DEVOPS_CONTENT_SYSTEM_PROMPT,
    INTERVIEW_EVALUTION_SYSTEM_PROMPT,
    MENTOR_CHAT_SYSTEM_PROMPT
)
from utils.logger import setup_logger

logger = setup_logger("GeminiService")

class GeminiService:
    """
    Communicates with Google Gemini API via its OpenAI-compatible endpoint.
    Handles structured output parsing (Pydantic v2) and conversational streaming.
    """

    def __init__(self, api_key: Optional[str] = None):
        # Resolve API Key: passed in key > environment GOOGLE_API_KEY
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.client = None
        self.model_name = "gemini-flash-latest"
        self.fallback_model = "gemini-flash-latest"
        
        if self.api_key:
            self._setup_client()
        else:
            logger.warning("Gemini Client initialized without GOOGLE_API_KEY. Sidebar fallback active.")

    def _setup_client(self) -> None:
        if not self.api_key:
            return
        try:
            # Point to Google's OpenAI compatibility layer
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            logger.info("Gemini Client successfully initialized on Google OpenAI-compatible endpoint.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini OpenAI compatibility client: {e}")

    def set_api_key(self, api_key: str) -> None:
        """Dynamically overrides API key during execution."""
        self.api_key = api_key
        self._setup_client()

    def is_configured(self) -> bool:
        """Returns True if the client is ready to make requests."""
        return self.client is not None

    def generate_devops_content(self, topic: str) -> DevOpsContent:
        """
        Generates full structured DevOps learning materials including explanation,
        roadmap, interview prep questions, labs, projects, and quiz questions.
        
        Args:
            topic: The DevOps topic string.
            
        Returns:
            DevOpsContent: Fully validated Pydantic model.
        """
        if not self.is_configured():
            raise ValueError("Gemini client is not configured. Please supply a GOOGLE_API_KEY.")
        
        logger.info(f"Initiating structured content generation for topic: {topic}")
        
        # Try primary model first, fallback on failure
        for model in [self.model_name, self.fallback_model]:
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": DEVOPS_CONTENT_SYSTEM_PROMPT.format(topic=topic)},
                        {"role": "user", "content": f"Create learning content pack for: {topic}"}
                    ],
                    response_format=DevOpsContent,
                    temperature=0.3
                )
                parsed_content = completion.choices[0].message.parsed
                if not parsed_content:
                    raise ValueError("Gemini returned an empty content parsed payload.")
                logger.info(f"Successfully generated and validated DevOpsContent for {topic} using model {model}.")
                return parsed_content
            except Exception as e:
                logger.warning(f"Content generation failed with model {model}: {e}")
                if model == self.fallback_model:
                    self._handle_api_error(e, f"generate_devops_content ({topic})")
                    raise e

    def evaluate_interview_answer(
        self, question: str, ideal_answer: str, user_answer: str
    ) -> InterviewEvaluation:
        """
        Evaluates a candidate's response to a DevOps interview question.
        
        Args:
            question: The interview question.
            ideal_answer: The model response.
            user_answer: The user's typed response.
            
        Returns:
            InterviewEvaluation: Validated score, strengths, weaknesses, and improvement recommendation.
        """
        if not self.is_configured():
            raise ValueError("Gemini client is not configured. Please supply a GOOGLE_API_KEY.")

        logger.info("Initiating interview response evaluation.")
        
        for model in [self.model_name, self.fallback_model]:
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": INTERVIEW_EVALUTION_SYSTEM_PROMPT.format(
                                question=question,
                                ideal_answer=ideal_answer,
                                user_answer=user_answer
                            )
                        },
                        {"role": "user", "content": "Evaluate my response."}
                    ],
                    response_format=InterviewEvaluation,
                    temperature=0.2
                )
                parsed_eval = completion.choices[0].message.parsed
                if not parsed_eval:
                    raise ValueError("Gemini returned an empty evaluation payload.")
                logger.info(f"Successfully evaluated interview answer using model {model}.")
                return parsed_eval
            except Exception as e:
                logger.warning(f"Interview evaluation failed with model {model}: {e}")
                if model == self.fallback_model:
                    self._handle_api_error(e, "evaluate_interview_answer")
                    raise e

    def stream_chat_response(
        self, messages: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        """
        Streams conversational completions token-by-token.
        
        Args:
            messages: List of chat message structures {'role': ..., 'content': ...}
            
        Yields:
            str: Token chunks from Gemini.
        """
        if not self.is_configured():
            raise ValueError("Gemini client is not configured. Please supply a GOOGLE_API_KEY.")

        formatted_messages = [{"role": "system", "content": MENTOR_CHAT_SYSTEM_PROMPT}]
        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                stream=True,
                temperature=0.7
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            self._handle_api_error(e, "stream_chat_response")
            raise e

    def _handle_api_error(self, exc: Exception, context: str) -> None:
        """Standardized internal logger for API exceptions."""
        err_msg = str(exc)
        if "quota" in err_msg.lower() or "429" in err_msg:
            logger.error(f"Gemini API QuotaExceeded (429) in {context}. Check billing/limits. Details: {err_msg}")
        elif "authentication" in err_msg.lower() or "401" in err_msg or "invalid_api_key" in err_msg:
            logger.error(f"Gemini API AuthenticationError (401) in {context}. Invalid GOOGLE_API_KEY. Details: {err_msg}")
        else:
            logger.error(f"Gemini API Exception in {context}: {err_msg}", exc_info=True)
