import os
import time
import re
from typing import List, Dict, Generator, Optional, Any
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

# Static offline DevOps knowledge base for Mentor Chat fallback
DEVOPS_STATIC_KB = {
    "docker": (
        "Docker is a containerization platform. It packages applications and dependencies "
        "into lightweight, portable containers that share the host OS kernel. "
        "Key commands: 'docker run' to start, 'docker ps' to list, 'docker build' to create images."
    ),
    "kubernetes": (
        "Kubernetes (K8s) is an container orchestration platform. It automates deployment, scaling, "
        "and management of containerized applications. Key components include Pods, Services, Deployments, "
        "and Nodes. It uses a declarative configuration model."
    ),
    "jenkins": (
        "Jenkins is an open-source automation server for building CI/CD (Continuous Integration/Continuous "
        "Delivery) pipelines. It integrates with source control (Git), runs tests, builds artifacts, "
        "and deploys applications."
    ),
    "terraform": (
        "Terraform is an Infrastructure as Code (IaC) tool by HashiCorp. It allows developers to define "
        "and provision cloud infrastructure using a declarative configuration language (HCL). "
        "Key lifecycle commands: 'terraform init', 'terraform plan', 'terraform apply'."
    ),
    "aws": (
        "Amazon Web Services (AWS) is a cloud computing provider. Core services include EC2 (compute), "
        "S3 (object storage), RDS (relational databases), and IAM (identity and access management)."
    ),
    "git": (
        "Git is a distributed version control system. Key workflows involve 'git clone', "
        "'git add', 'git commit', 'git push', and 'git merge' to collaborate on source repositories."
    ),
    "ci/cd": (
        "CI/CD stands for Continuous Integration and Continuous Delivery/Deployment. It is the practice "
        "of automating code build, testing, and deployment to release software rapidly and reliably."
    ),
    "linux": (
        "Linux is an open-source operating system standard in DevOps. Core administration tasks "
        "involve managing file systems, file permissions (chmod/chown), networking commands (ip, netstat), "
        "and process management (ps, top, systemctl)."
    )
}

class GeminiService:
    """
    Communicates with Google Gemini API via its OpenAI-compatible endpoint.
    Handles structured output parsing, retries, circuit breaking, timeouts, and fallbacks.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.client = None
        self.model_name = "gemini-2.5-flash"
        self.fallback_model = "gemini-2.5-flash"
        
        # Circuit Breaker state
        self.failure_count = 0
        self.circuit_state = "CLOSED"  # CLOSED or OPEN
        self.last_failure_time = 0.0
        self.cooldown_period = 60.0    # seconds
        self.failure_threshold = 3     # 3 consecutive failures to open circuit
        self.last_api_error = None
        self.last_api_success_time = None

        if self.api_key:
            self._setup_client()
        else:
            logger.warning("Gemini Client initialized without GOOGLE_API_KEY. Sidebar fallback active.")

    def _setup_client(self) -> None:
        if not self.api_key:
            return
        try:
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

    def _check_circuit(self) -> bool:
        """Returns True if the circuit allows API requests, False if it is open (should fallback)."""
        cooldown_remaining = max(0.0, self.cooldown_period - (time.time() - self.last_failure_time)) if self.circuit_state == "OPEN" else 0.0
        logger.info(f"Circuit Breaker Status - State: {self.circuit_state} | Failure Count: {self.failure_count} | Cooldown Remaining: {cooldown_remaining:.2f}s")
        if self.circuit_state == "OPEN":
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.cooldown_period:
                logger.info("Circuit Breaker cooldown elapsed. Transitioning from OPEN to CLOSED.")
                self.circuit_state = "CLOSED"
                self.failure_count = 0
                return True
            else:
                logger.warning(f"Circuit Breaker is OPEN. Cooldown remaining: {self.cooldown_period - elapsed:.1f}s. Routing directly to offline fallback.")
                return False
        return True

    def _record_success(self) -> None:
        """Resets failure counter upon a successful request."""
        if self.circuit_state != "CLOSED":
            logger.info("API request succeeded. Resetting Circuit Breaker to CLOSED.")
            self.circuit_state = "CLOSED"
        self.failure_count = 0

    def _record_failure(self) -> None:
        """Increments the failure counter and trips the circuit if threshold is reached."""
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            if self.circuit_state != "OPEN":
                logger.error(f"Circuit Breaker tripped! {self.failure_count} consecutive failures. Transitioning to OPEN state for 60s.")
                self.circuit_state = "OPEN"
                self.last_failure_time = time.time()

    def _execute_with_retry(self, api_func, *args, **kwargs) -> Any:
        """
        Executes an API function with a 60s timeout, retrying up to 3 times
        with exponential backoff (2s, 4s, 8s). Tracks latency, retries, errors,
        and fallbacks in structured log messages.
        """
        feature = kwargs.pop("feature_name", "GeminiAPI")
        model_name = kwargs.get("model", self.model_name)
        endpoint = str(self.client.base_url) if self.client else "unknown"
        
        # Check Circuit Breaker
        if not self._check_circuit():
            raise Exception("Circuit Breaker is OPEN. API requests temporarily routed to fallback.")
            
        retries = 0
        backoff_times = [2.0, 4.0, 8.0]
        start_time = time.time()
        
        while retries <= 3:
            try:
                # Add native timeout to OpenAI request
                kwargs["timeout"] = 60.0
                logger.info(f"API Request Executing - Model Name: {model_name} | Endpoint: {endpoint} | Feature: {feature} | Attempt: {retries + 1} of 4 | Start Time: {time.time()}")
                logger.info("START GEMINI REQUEST")
                result = api_func(*args, **kwargs)
                logger.info("END GEMINI REQUEST")
                latency = time.time() - start_time
                
                # Successful execution
                self._record_success()
                self.last_api_success_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                logger.info(f"API Request Success - Model Name: {model_name} | Endpoint: {endpoint} | Feature: {feature} | Success Time: {time.time()} | Latency: {latency:.2f}s | Retries: {retries} | Fallback: False | Error: None")
                return result
                
            except Exception as e:
                err_msg = str(e)
                self.last_api_error = f"{type(e).__name__}: {err_msg}"
                latency = time.time() - start_time
                logger.error(f"API Request Failure - Model Name: {model_name} | Endpoint: {endpoint} | Feature: {feature} | Failure Time: {time.time()} | Error: {err_msg} | Latency: {latency:.2f}s")
                self._record_failure()  # Record attempt failure immediately to trip circuit breaker
                
                retries += 1
                if retries <= 3:
                    wait_time = backoff_times[retries - 1]
                    logger.info(f"[{feature}] Waiting {wait_time}s before next attempt...")
                    time.sleep(wait_time)
                else:
                    # Final failure: raise exception to trigger fallback
                    latency = time.time() - start_time
                    logger.error(f"Feature: {feature} | Latency: {latency:.2f}s | Retries: 3 | Fallback: True | Error: {err_msg}")
                    raise e

    def generate_devops_content(self, topic: str) -> DevOpsContent:
        """
        Generates full structured DevOps learning materials.
        Falls back gracefully to offline cached packs/templates if Gemini is unavailable.
        """
        if not self.is_configured():
            logger.warning(f"Gemini client not configured. Routing generate_devops_content ({topic}) to offline fallback.")
            return self._generate_offline_devops_content(topic)

        logger.info(f"Initiating structured content generation for topic: {topic}")
        
        models_to_try = [self.model_name]
        if self.fallback_model and self.fallback_model != self.model_name:
            models_to_try.append(self.fallback_model)
            
        for i, model in enumerate(models_to_try):
            try:
                completion = self._execute_with_retry(
                    self.client.beta.chat.completions.parse,
                    model=model,
                    messages=[
                        {"role": "system", "content": DEVOPS_CONTENT_SYSTEM_PROMPT.format(topic=topic)},
                        {"role": "user", "content": f"Create learning content pack for: {topic}"}
                    ],
                    response_format=DevOpsContent,
                    temperature=0.3,
                    feature_name=f"Learning Hub ({model})"
                )
                parsed_content = completion.choices[0].message.parsed
                if not parsed_content:
                    raise ValueError("Gemini returned an empty content parsed payload.")
                return parsed_content
            except Exception as e:
                import traceback
                logger.error(f"Fallback triggered in generate_devops_content. Exception Type: {type(e).__name__}, Message: {str(e)}\nStack Trace:\n{traceback.format_exc()}")
                logger.warning(f"Content generation failed with model {model}: {e}")
                if i == len(models_to_try) - 1:
                    logger.error(f"generate_devops_content ({topic}) failed. Swapping to offline generated pack.")
                    return self._generate_offline_devops_content(topic)

    def evaluate_interview_answer(
        self, question: str, ideal_answer: str, user_answer: str
    ) -> InterviewEvaluation:
        """
        Evaluates a candidate's response to a DevOps interview question.
        Falls back gracefully to rule-based offline evaluation if Gemini is unavailable.
        """
        if not self.is_configured():
            logger.warning("Gemini client not configured. Routing interview evaluation to offline fallback.")
            return self._offline_evaluate(question, ideal_answer, user_answer)

        logger.info("Initiating interview response evaluation.")
        
        models_to_try = [self.model_name]
        if self.fallback_model and self.fallback_model != self.model_name:
            models_to_try.append(self.fallback_model)
            
        for i, model in enumerate(models_to_try):
            try:
                completion = self._execute_with_retry(
                    self.client.beta.chat.completions.parse,
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
                    temperature=0.2,
                    feature_name=f"Interview Evaluator ({model})"
                )
                parsed_eval = completion.choices[0].message.parsed
                if not parsed_eval:
                    raise ValueError("Gemini returned an empty evaluation payload.")
                return parsed_eval
            except Exception as e:
                import traceback
                logger.error(f"Fallback triggered in evaluate_interview_answer. Exception Type: {type(e).__name__}, Message: {str(e)}\nStack Trace:\n{traceback.format_exc()}")
                logger.warning(f"Interview evaluation failed with model {model}: {e}")
                if i == len(models_to_try) - 1:
                    logger.error("evaluate_interview_answer failed. Swapping to offline rule-based evaluation.")
                    return self._offline_evaluate(question, ideal_answer, user_answer)

    def stream_chat_response(
        self, messages: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        """
        Streams conversational completions token-by-token.
        Falls back gracefully to offline mentor chatbot if API is unavailable.
        """
        if not self.is_configured():
            logger.warning("Gemini Client not configured. Routing stream_chat_response to offline fallback.")
            yield from self._stream_offline_mentor_answer(messages[-1]["content"])
            return

        formatted_messages = [{"role": "system", "content": MENTOR_CHAT_SYSTEM_PROMPT}]
        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        try:
            stream = self._execute_with_retry(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=formatted_messages,
                stream=True,
                temperature=0.7,
                feature_name="Mentor Chat"
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            import traceback
            logger.error(f"Fallback triggered in stream_chat_response. Exception Type: {type(e).__name__}, Message: {str(e)}\nStack Trace:\n{traceback.format_exc()}")
            logger.warning(f"Mentor Chat stream failed. Routing to offline fallback. Detail: {e}")
            yield from self._stream_offline_mentor_answer(messages[-1]["content"])

    def _stream_offline_mentor_answer(self, query: str) -> Generator[str, None, None]:
        """Simulates token-by-token streaming of the offline mentor response."""
        fallback_text = self._generate_offline_mentor_answer(query)
        for word in re.split(r'(\s+)', fallback_text):
            if word:
                yield word
                time.sleep(0.01)

    def _generate_offline_mentor_answer(self, query: str) -> str:
        """Generates fallback answers using cache resources, RAG queries, or static DevOps KB."""
        query_lower = query.lower()
        
        # Check if RAG context is injected
        if "retrieved context:" in query_lower or "context extracted" in query_lower:
            lines = []
            source_matches = re.findall(r'\[Source:\s*([^\]]+)\]\s*([^\n]+)', query)
            if source_matches:
                for src, text in source_matches[:3]:
                    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]
                    first_few = " ".join(sentences[:2])
                    lines.append(f"• **From [{src}]:** {first_few}...")
            else:
                lines.append("• Matched operational documents and retrieved relevant context blocks.")
                
            return (
                "AI generation temporarily unavailable. Displaying retrieved knowledge.\n\n"
                "**Summary of Retrieved Knowledge:**\n" + "\n".join(lines)
            )

        # Check cached study packs
        cache_dir = os.path.join("data", "cache")
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                if file.endswith(".json"):
                    topic = file[:-5].lower()
                    if topic in query_lower:
                        try:
                            filepath = os.path.join(cache_dir, file)
                            with open(filepath, "r", encoding="utf-8") as f:
                                import json
                                data = json.load(f)
                            explanation = data.get("explanation", "")
                            if explanation:
                                snippet = explanation[:500] + "..."
                                return (
                                    "AI mentor temporarily unavailable. Showing offline guidance.\n\n"
                                    f"**Offline Context (Cached Study Pack for {topic.capitalize()}):**\n"
                                    f"{snippet}"
                                )
                        except Exception:
                            pass

        # Check static KB
        for key, text in DEVOPS_STATIC_KB.items():
            if key in query_lower:
                return (
                    "AI mentor temporarily unavailable. Showing offline guidance.\n\n"
                    f"**Offline Context ({key.capitalize()} Overview):**\n"
                    f"{text}"
                )
                
        # Default response
        return (
            "AI mentor temporarily unavailable. Showing offline guidance.\n\n"
            "I am currently operating in offline mode. Please select a topic from the Learning Hub "
            "or consult your uploaded documents. Once connections are restored, full AI mentoring will resume."
        )

    def _offline_evaluate(self, question: str, ideal_answer: str, user_answer: str) -> InterviewEvaluation:
        """Performs a rule-based evaluation based on length and technical keywords."""
        def tokenize(text):
            words = re.sub(r'[^\w\s]', '', text.lower()).split()
            return set(w for w in words if len(w) > 3)

        user_words = tokenize(user_answer)
        ideal_words = tokenize(ideal_answer)
        
        tech_keywords = {
            "container", "orchestration", "deployment", "kubernetes", "docker", 
            "pipeline", "jenkins", "terraform", "automation", "cloud", "aws", 
            "service", "network", "volume", "security", "cluster", "node", 
            "pod", "config", "cicd", "continuous", "integration", "delivery"
        }
        
        overlap = user_words.intersection(ideal_words)
        tech_overlap = user_words.intersection(tech_keywords)
        
        score = 1
        length = len(user_answer.split())
        
        # Length factor (up to 4 points)
        if length > 30:
            score += 4
        elif length > 15:
            score += 3
        elif length > 5:
            score += 2
        else:
            score += 1
            
        # Keyword overlap factor (up to 3 points)
        if len(overlap) >= 8:
            score += 3
        elif len(overlap) >= 4:
            score += 2
        elif len(overlap) >= 1:
            score += 1
            
        # Technical keyword coverage (up to 2 points)
        if len(tech_overlap) >= 3:
            score += 2
        elif len(tech_overlap) >= 1:
            score += 1
            
        score = min(score, 10)
        
        strengths = []
        if len(overlap) > 0:
            strengths.append(f"Used {len(overlap)} relevant keywords from the model answer.")
        if len(tech_overlap) > 0:
            strengths.append(f"Demonstrated technical awareness of cloud terms: {', '.join(list(tech_overlap)[:3])}.")
        if not strengths:
            strengths.append("Submitted a practice answer response.")
            
        weaknesses = []
        missing = ideal_words - user_words
        if len(missing) > 5:
            show_missing = list(missing.intersection(tech_keywords))[:3]
            if show_missing:
                weaknesses.append(f"Omitted key architectural concepts: {', '.join(show_missing)}.")
            else:
                weaknesses.append("Lacked specific technical details mentioned in the ideal answer.")
        if length < 15:
            weaknesses.append("Answer is very brief. Expand on technical implementation details.")
        if not weaknesses:
            weaknesses.append("Could provide additional real-world operational examples.")
            
        improved = ideal_answer + "\n\n*(AI evaluator temporarily unavailable. Showing offline evaluation.)*"
        
        return InterviewEvaluation(
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            improved_answer=improved
        )

    def _generate_offline_devops_content(self, topic: str) -> DevOpsContent:
        """Generates static structured DevOpsContent objects when API limits are reached."""
        from models.content import Roadmap, RoadmapLevel, InterviewQuestion, HandsOnTask, DevOpsProject, QuizQuestion
        
        explanation = (
            f"### {topic} (Offline Overview)\n\n"
            f"Detailed structured content generation is temporarily offline. "
            f"{topic} is a key component of modern DevOps engineering.\n\n"
            f"**Real-World Use Cases:**\n"
            f"- Automating infrastructure and cloud workflows.\n"
            f"- Ensuring consistency across dev, staging, and production environments.\n\n"
            f"**Architecture Overview:**\n"
            f"```mermaid\n"
            f"graph LR\n"
            f"  User --> Client[Client Interface]\n"
            f"  Client --> Server[Service Backend]\n"
            f"  Server --> Database[(Data Store)]\n"
            f"```\n\n"
            f"*(AI content generator temporarily unavailable. Showing offline material.)*"
        )
        
        roadmap = Roadmap(
            beginner=RoadmapLevel(duration="1 week", topics=[f"Introduction to {topic}", f"Basic installation and setup of {topic}"]),
            intermediate=RoadmapLevel(duration="2 weeks", topics=[f"Configuring advanced options in {topic}", f"Integrating {topic} into pipelines"]),
            advanced=RoadmapLevel(duration="2 weeks", topics=[f"Scaling and securing {topic} in clusters", f"Troubleshooting {topic} in production"])
        )
        
        interview_questions = []
        for i in range(1, 6):
            interview_questions.append(InterviewQuestion(
                id=i,
                level="beginner",
                question=f"What is the primary purpose of {topic}?",
                answer=f"{topic} is used to optimize operations and automate software workflows in modern cloud architectures.",
                common_mistakes=["Confusing it with unrelated operations.", "Lacking high-level overview."]
            ))
        for i in range(6, 11):
            interview_questions.append(InterviewQuestion(
                id=i,
                level="intermediate",
                question=f"How do you troubleshoot a failed deployment of {topic}?",
                answer="Check the system configuration files, inspect the container/service logs, and verify network connectivity.",
                common_mistakes=["Not checking logs first.", "Changing random variables without diagnosing."]
            ))
        for i in range(11, 14):
            interview_questions.append(InterviewQuestion(
                id=i,
                level="advanced",
                question=f"Describe scaling strategies for {topic} in high availability environments.",
                answer="Deploy across multiple availability zones, implement active-active clustering, and run behind a load balancer.",
                common_mistakes=["Creating single points of failure.", "Not configuring health checks."]
            ))
            
        hands_on_tasks = [
            HandsOnTask(
                title=f"Lab 1: Installing and Configuring {topic}",
                instructions=f"Set up the basic environment for {topic} on your local system.",
                commands=[f"# Check version\n{topic.lower()} --version"]
            ),
            HandsOnTask(
                title=f"Lab 2: Running a simple project with {topic}",
                instructions="Deploy a basic service configuration.",
                commands=[f"# Run configuration\n{topic.lower()} up -d"]
            )
        ]
        
        projects = [
            DevOpsProject(
                title=f"Dockerizing a {topic} Service",
                difficulty="Beginner",
                architecture="A single container running the base application service.",
                tech_stack=["Docker", topic],
                learning_outcomes=["Container building basics", "Environment variables configuration"]
            ),
            DevOpsProject(
                title=f"CI/CD pipeline for {topic}",
                difficulty="Intermediate",
                architecture="GitHub push triggers Jenkins build and tests.",
                tech_stack=["Jenkins", "Git", topic],
                learning_outcomes=["Automation pipelines", "Integration testing"]
            ),
            DevOpsProject(
                title=f"Production Cluster Deployment of {topic}",
                difficulty="Advanced",
                architecture="High availability multi-node cluster behind a load balancer.",
                tech_stack=["Kubernetes", "AWS", topic],
                learning_outcomes=["High availability architectures", "Load balancing and security profiles"]
            )
        ]
        
        quiz = []
        for i in range(1, 11):
            quiz.append(QuizQuestion(
                id=i,
                question=f"Which of the following is true about {topic}?",
                options=[
                    "It is a standard tool in DevOps.",
                    "It is deprecated and should not be used.",
                    "It only runs on legacy hardware.",
                    "It is a programming language."
                ],
                correct_option="It is a standard tool in DevOps.",
                explanation=f"{topic} is widely adopted across the cloud and infrastructure space."
            ))
            
        return DevOpsContent(
            topic=topic,
            explanation=explanation,
            roadmap=roadmap,
            interview_questions=interview_questions,
            hands_on_tasks=hands_on_tasks,
            projects=projects,
            quiz=quiz
        )
