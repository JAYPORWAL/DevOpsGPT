import os
import uuid
import re
import json
from typing import Optional
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import models
from models.tracking import QuizAttempt
from models.content import DevOpsContent

# Import services
from services.storage_service import StorageService, CORE_TOPICS
from services.gemini_service import GeminiService
from services.rag_service import RAGService

# Import utilities
from utils.helpers import model_to_json_str, quiz_attempts_to_df, get_difficulty_color
from utils.logger import setup_logger

logger = setup_logger("App")

# Page Configuration
st.set_page_config(
    page_title="DevOpsGPT - DevOps Learning & Interview Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("static/style.css")

# --- SESSION STATE INITIALIZATION ---
if "storage_service" not in st.session_state:
    st.session_state.storage_service = StorageService()

# API Key resolution: check .env GOOGLE_API_KEY first
env_google_key = os.getenv("GOOGLE_API_KEY")

# --- SIDEBAR API KEY CONFIGURATION ---
st.sidebar.markdown(
    "<div style='text-align: center;'><h1 class='cyber-title' style='font-size: 24px; margin-bottom: 5px;'>DevOpsGPT</h1>"
    "<p style='color: #8b949e; font-size: 12px; margin-top:0;'>Learning, Interview & RAG Assistant</p></div>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

# Determine key availability
has_key = bool(env_google_key) or "google_api_key" in st.session_state
api_key_status = "✅ API Key Loaded" if has_key else "⚠️ API Key Missing"

st.sidebar.caption(f"API Status: {api_key_status}")
sidebar_key = st.sidebar.text_input(
    "Google API Key Fallback",
    type="password",
    value=st.session_state.get("google_api_key", ""),
    placeholder="AIzaSy...",
    help="Provide your Google API key here if not specified in the .env file."
)

if sidebar_key and sidebar_key != st.session_state.get("google_api_key", ""):
    st.session_state.google_api_key = sidebar_key
    st.rerun()

# Determine active key
active_key = st.session_state.get("google_api_key") or env_google_key

# Get/Cache service instances
@st.cache_resource
def get_gemini_service(api_key: Optional[str]) -> GeminiService:
    return GeminiService(api_key=api_key)

@st.cache_resource
def get_rag_service(api_key: Optional[str]) -> RAGService:
    return RAGService(api_key=api_key)

gemini_service = get_gemini_service(active_key)
rag_service = get_rag_service(active_key)

# Store in session state for backward compatibility
st.session_state.gemini_service = gemini_service
st.session_state.rag_service = rag_service

if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""

if "current_content" not in st.session_state:
    st.session_state.current_content = None

if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = str(uuid.uuid4())

if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = {}

st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Go To:",
    [
        "📊 Dashboard & Analytics",
        "🧠 Learning Hub",
        "💼 Interview Prep",
        "📝 Interactive Quiz",
        "🔍 RAG Knowledge Assistant",
        "💬 Mentor Chat"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Core Topics Reference")
st.sidebar.info(
    "• Docker\n"
    "• Kubernetes\n"
    "• Jenkins\n"
    "• Terraform\n"
    "• AWS Cloud\n"
    "• Linux Administration\n"
    "• Git Version Control\n"
    "• CI/CD Pipelines"
)

# Helper for rendering Mermaid Diagrams in Streamlit
def display_mermaid(explanation: str):
    """
    Finds Mermaid code blocks, extracts them, renders them using Mermaid CDN,
    and displays the remaining explanation as standard markdown.
    """
    mermaid_blocks = re.findall(r"```mermaid\s*\n(.*?)\n```", explanation, re.DOTALL)
    
    if mermaid_blocks:
        # Separate code blocks and explanation
        split_text = re.split(r"```mermaid\s*\n.*?\n```", explanation, flags=re.DOTALL)
        
        for idx, text_part in enumerate(split_text):
            if text_part.strip():
                st.markdown(text_part)
            
            # If there's a corresponding mermaid block, render it
            if idx < len(mermaid_blocks):
                mermaid_code = mermaid_blocks[idx].strip()
                html_code = f"""
                <div style="background-color: #0c0f16; border: 1px solid rgba(0, 240, 255, 0.15); border-radius: 8px; padding: 15px; margin: 10px 0; overflow-x: auto;">
                    <pre class="mermaid" style="background: transparent; text-align: center;">
                    {mermaid_code}
                    </pre>
                </div>
                <script type="module">
                    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                    mermaid.initialize({{ startOnLoad: true, theme: 'dark', securityLevel: 'loose' }});
                </script>
                """
                st.components.v1.html(html_code, height=450, scrolling=True)
    else:
        st.markdown(explanation)


# ==========================================
# PAGE 1: DASHBOARD & ANALYTICS
# ==========================================
if page == "📊 Dashboard & Analytics":
    st.markdown("<h1 class='cyber-title'>DevOps Progress Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='cyber-subtitle'>Track your learning milestones, quiz statistics, and weak technical topics.</p>", unsafe_allow_html=True)

    progress = st.session_state.storage_service.load_progress()
    attempts = st.session_state.storage_service.load_quiz_attempts()
    df_attempts = quiz_attempts_to_df([att.model_dump() for att in attempts])

    # 1. High-level metric calculations
    total_core = len(CORE_TOPICS)
    completed_topics = [t for t, tp in progress.topics.items() if tp.completed]
    topics_learned_count = len(completed_topics)
    completion_rate = (topics_learned_count / total_core) * 100 if total_core > 0 else 0.0

    quizzes_count = len(attempts)
    avg_score = df_attempts["percentage"].mean() if not df_attempts.empty else 0.0

    # Render Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Topics Learned", f"{topics_learned_count} / {total_core}", help="Completed core topics")
    with col2:
        st.metric("Quizzes Attempted", f"{quizzes_count}", help="Total quizzes completed")
    with col3:
        st.metric("Average Score", f"{avg_score:.1f}%", help="Average percentage in quizzes")
    with col4:
        st.metric("Completion Rate", f"{completion_rate:.1f}%", help="Percentage of core curriculum completed")

    st.markdown("<br>", unsafe_allow_html=True)

    # Plotly Visualizations
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("### 📈 Core DevOps Syllabus Completion")
        # Generate donut chart of syllabus completion
        labels = ["Completed", "Remaining"]
        values = [topics_learned_count, max(0, total_core - topics_learned_count)]
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.5,
            marker_colors=["#00f0ff", "#1f2937"],
            textinfo="percent+label",
            textfont_size=12
        )])
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8b949e'),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            margin=dict(t=10, b=10, l=10, r=10),
            height=300
        )
        st.plotly_chart(fig_pie, width="stretch")

    with col_chart2:
        st.markdown("### 📊 Quiz Performance Over Time")
        if not df_attempts.empty:
            df_sorted = df_attempts.sort_values("timestamp")
            fig_line = px.line(
                df_sorted,
                x="timestamp",
                y="percentage",
                markers=True,
                title="Quiz Accuracy Score (%)",
                color_discrete_sequence=["#8a2be2"]
            )
            fig_line.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8b949e'),
                xaxis=dict(showgrid=False, title="Attempt Date"),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Score %", range=[0, 105]),
                margin=dict(t=30, b=10, l=10, r=10),
                height=300
            )
            st.plotly_chart(fig_line, width="stretch")
        else:
            st.info("No quiz attempts recorded. Generate a quiz to start plotting scores!")

    # Performance per topic
    st.markdown("### 🛠️ Topic Performance Metrics")
    topic_data = []
    for topic in CORE_TOPICS:
        tp = progress.topics.get(topic)
        best_score = tp.best_score if tp else None
        completed = tp.completed if tp else False
        quizzes_taken = tp.quizzes_taken if tp else 0
        topic_data.append({
            "Topic": topic,
            "Completed": "✅ Yes" if completed else "❌ No",
            "Quizzes Taken": quizzes_taken,
            "Best Quiz Score": f"{best_score:.1f}%" if best_score is not None else "N/A",
            "_score": best_score if best_score is not None else -1
        })
    df_topics = pd.DataFrame(topic_data)

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.dataframe(
            df_topics.drop(columns=["_score"]),
            width="stretch",
            hide_index=True
        )
    with col_t2:
        st.markdown("#### Focus Areas (Weak Topics)")
        weak_topics = progress.weak_topics
        if weak_topics:
            for wt in weak_topics:
                st.error(f"⚠️ **{wt}** (Score under 70%)")
        else:
            st.success("🌟 Great job! No weak topics identified yet.")

    # Reset option
    st.markdown("---")
    with st.expander("Reset Application State"):
        st.warning("This deletes all progress tracking, cached studies, and chat logs.")
        if st.button("Purge Saved Data"):
            # Purge progress files
            if os.path.exists(st.session_state.storage_service.progress_file):
                os.remove(st.session_state.storage_service.progress_file)
            if os.path.exists(st.session_state.storage_service.quiz_file):
                os.remove(st.session_state.storage_service.quiz_file)
            # Purge cache
            for file in os.listdir(st.session_state.storage_service.cache_dir):
                os.remove(os.path.join(st.session_state.storage_service.cache_dir, file))
            # Purge chat
            for file in os.listdir(st.session_state.storage_service.chat_dir):
                os.remove(os.path.join(st.session_state.storage_service.chat_dir, file))
            
            st.success("All data purged successfully. Please refresh the page.")


# ==========================================
# PAGE 2: LEARNING HUB
# ==========================================
elif page == "🧠 Learning Hub":
    st.markdown("<h1 class='cyber-title'>DevOps Learning Hub</h1>", unsafe_allow_html=True)
    st.markdown("<p class='cyber-subtitle'>Generate modular DevOps content, interactive roadmaps, and lab tasks.</p>", unsafe_allow_html=True)

    # Check key configuration
    if not st.session_state.gemini_service.is_configured():
        st.warning("⚠️ Google Gemini API key is missing. Please supply a key in the sidebar configuration.")
        st.stop()

    # Select core topic or type custom
    col_input, col_preset = st.columns([2, 1])
    with col_preset:
        preset_topic = st.selectbox(
            "Select Core Topic:",
            [""] + CORE_TOPICS,
            help="Choose a pre-defined core topic or type in a custom topic to the left."
        )
    with col_input:
        custom_topic = st.text_input(
            "Enter Topic Name (e.g. Kubernetes, Ansible, Docker Multi-stage):",
            value=preset_topic if preset_topic else ""
        )

    generate_clicked = st.button("Generate DevOps Study Pack")
    
    # Topic load handling
    if custom_topic and (generate_clicked or (st.session_state.current_topic != custom_topic and not st.session_state.current_content)):
        topic_normalized = custom_topic.strip()
        
        # Check cache first (Smart Caching)
        cached_content = st.session_state.storage_service.load_from_cache(topic_normalized)
        
        if cached_content and not generate_clicked:
            st.session_state.current_topic = topic_normalized
            st.session_state.current_content = cached_content
            # Mark studied
            st.session_state.storage_service.update_progress_on_completion(topic_normalized, completed=True)
            st.success(f"⚡ Loaded {topic_normalized} from local cache!")
        else:
            # Force generate/fetch from Gemini
            with st.spinner(f"AI Mentor is compiling comprehensive DevOps Pack for: {topic_normalized}... This may take a minute."):
                try:
                    generated = st.session_state.gemini_service.generate_devops_content(topic_normalized)
                    
                    # Store in Cache
                    st.session_state.storage_service.save_to_cache(topic_normalized, generated)
                    
                    # Update local state
                    st.session_state.current_topic = topic_normalized
                    st.session_state.current_content = generated
                    
                    # Mark studied
                    st.session_state.storage_service.update_progress_on_completion(topic_normalized, completed=True)
                    
                    # Reset quiz state
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.session_state.evaluation_results = {}
                    
                    st.balloons()
                    st.success(f"🚀 Study Pack for {topic_normalized} generated and cached successfully!")
                except Exception as e:
                    st.error(f"Failed to generate study pack: {e}")
                    st.stop()

    # Render Content
    if st.session_state.current_content:
        content: DevOpsContent = st.session_state.current_content
        
        st.markdown(f"## Currently Studying: **{content.topic}**")
        
        # Define learning tabs
        tab_explain, tab_roadmap, tab_tasks, tab_projects, tab_export = st.tabs([
            "📚 Concept Explanation",
            "🗺️ Learning Roadmap",
            "💻 Hands-On Labs",
            "🏗️ Mini Projects",
            "💾 Export Study Pack"
        ])
        
        with tab_explain:
            st.markdown("### Explanation & System Architecture")
            display_mermaid(content.explanation)
            
        with tab_roadmap:
            st.markdown("### Study Guide Timeline")
            col_b, col_i, col_a = st.columns(3)
            
            with col_b:
                st.markdown(f"<div class='devops-card'><h4>🟢 Beginner ({content.roadmap.beginner.duration})</h4>", unsafe_allow_html=True)
                for t in content.roadmap.beginner.topics:
                    st.markdown(f"- {t}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_i:
                st.markdown(f"<div class='devops-card'><h4>🟡 Intermediate ({content.roadmap.intermediate.duration})</h4>", unsafe_allow_html=True)
                for t in content.roadmap.intermediate.topics:
                    st.markdown(f"- {t}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_a:
                st.markdown(f"<div class='devops-card'><h4>🔴 Advanced ({content.roadmap.advanced.duration})</h4>", unsafe_allow_html=True)
                for t in content.roadmap.advanced.topics:
                    st.markdown(f"- {t}")
                st.markdown("</div>", unsafe_allow_html=True)
                
        with tab_tasks:
            st.markdown("### Laboratory Exercises")
            for idx, task in enumerate(content.hands_on_tasks):
                with st.expander(f"Lab {idx+1}: {task.title}", expanded=(idx==0)):
                    st.markdown(task.instructions)
                    if task.commands:
                        st.markdown("**Lab Command Reference:**")
                        for cmd in task.commands:
                            st.code(cmd, language="bash")
                            
        with tab_projects:
            st.markdown("### Mini Capstone Projects")
            for proj in content.projects:
                diff_color = get_difficulty_color(proj.difficulty)
                st.markdown(
                    f"<div class='devops-card'>"
                    f"<h4>🔨 {proj.title} "
                    f"<span style='color: {diff_color}; font-size:14px;'>[{proj.difficulty}]</span></h4>"
                    f"<p><strong>Architecture:</strong> {proj.architecture}</p>"
                    f"<p><strong>Tech Stack:</strong> {', '.join(proj.tech_stack)}</p>"
                    f"<strong>Learning Outcomes:</strong>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                for outcome in proj.learning_outcomes:
                    st.markdown(f"• {outcome}")
                st.markdown("<br>", unsafe_allow_html=True)

        with tab_export:
            st.markdown("### Save Offline Study Pack")
            st.info("You can download the generated study resources, quizzes, and roadmap in a structured JSON schema.")
            json_data = model_to_json_str(content)
            st.download_button(
                label="📥 Export Topic Pack (JSON)",
                data=json_data,
                file_name=f"devops_pack_{content.topic.lower().replace(' ', '_')}.json",
                mime="application/json"
            )
    else:
        st.info("No topic selected. Enter a DevOps topic above to generate study modules!")


# ==========================================
# PAGE 3: INTERVIEW PREPARATION & EVALUATOR
# ==========================================
elif page == "💼 Interview Prep":
    st.markdown("<h1 class='cyber-title'>Interview Preparation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='cyber-subtitle'>Test your knowledge against model answers and get custom evaluation grades from AI.</p>", unsafe_allow_html=True)

    if not st.session_state.current_content:
        st.warning("⚠️ No topic loaded. Please go to the **🧠 Learning Hub** and generate or load a study pack first!")
        st.stop()

    content: DevOpsContent = st.session_state.current_content
    st.markdown(f"### Interview Questions for **{content.topic}**")

    # Filter questions by level
    q_level = st.radio("Select Question Difficulty Level:", ["Beginner", "Intermediate", "Advanced"], horizontal=True)
    filtered_qs = [q for q in content.interview_questions if q.level.lower() == q_level.lower()]

    if not filtered_qs:
        st.warning(f"No {q_level} questions generated for this topic.")
        st.stop()

    for idx, q in enumerate(filtered_qs):
        with st.expander(f"Question {idx+1} [{q.level.upper()}]: {q.question}", expanded=False):
            # Model Answer Section
            st.markdown("#### 🌟 Model Answer")
            st.info(q.answer)

            # Common Mistakes Section
            if q.common_mistakes:
                st.markdown("#### ⚠️ Common Pitfalls to Avoid")
                for mistake in q.common_mistakes:
                    st.markdown(f"- {mistake}")

            st.markdown("---")

            # Answer Evaluation Section
            st.markdown("#### 📝 Practice Answer Evaluator")
            eval_key = f"eval_{q.id}"
            user_ans = st.text_area("Type your practice answer here:", key=f"ans_text_{q.id}", height=120)

            evaluate_btn = st.button("Submit Answer for Grading", key=f"eval_btn_{q.id}")

            if evaluate_btn:
                if not user_ans.strip():
                    st.error("Please enter a response before requesting evaluation.")
                else:
                    with st.spinner("Analyzing response and generating feedback..."):
                        try:
                            evaluation = st.session_state.gemini_service.evaluate_interview_answer(
                                question=q.question,
                                ideal_answer=q.answer,
                                user_answer=user_ans
                            )
                            # Store in session state
                            st.session_state.evaluation_results[eval_key] = evaluation
                        except Exception as e:
                            st.error(f"Failed to grade answer: {e}")

            # Display saved results
            if eval_key in st.session_state.evaluation_results:
                result = st.session_state.evaluation_results[eval_key]
                st.markdown(f"##### **Result Score:** `{result.score}/10`")
                
                # Render score bar
                score_pct = result.score * 10
                st.progress(score_pct / 100)

                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.markdown("🟢 **Strengths:**")
                    for s in result.strengths:
                        st.markdown(f"- {s}")
                with col_e2:
                    st.markdown("🔴 **Weaknesses:**")
                    for w in result.weaknesses:
                        st.markdown(f"- {w}")

                st.markdown("✨ **How to Improve:**")
                st.write(result.improved_answer)


# ==========================================
# PAGE 4: INTERACTIVE QUIZ
# ==========================================
elif page == "📝 Interactive Quiz":
    st.markdown("<h1 class='cyber-title'>DevOps Interactive Quiz</h1>", unsafe_allow_html=True)
    st.markdown("<p class='cyber-subtitle'>Test your command on the subject and record results to your profile.</p>", unsafe_allow_html=True)

    if not st.session_state.current_content:
        st.warning("⚠️ No topic loaded. Please go to the **🧠 Learning Hub** and generate or load a study pack first!")
        st.stop()

    content: DevOpsContent = st.session_state.current_content
    st.markdown(f"### Topic Quiz: **{content.topic}** (10 MCQs)")
    st.markdown("Select your answers below and click **Submit Quiz** at the bottom.")

    # Form to hold quiz inputs
    with st.form("quiz_form"):
        for q in content.quiz:
            st.markdown(f"#### Q{q.id}. {q.question}")
            
            # Retrieve previous choice if exists
            choice_idx = None
            saved_choice = st.session_state.quiz_answers.get(q.id)
            if saved_choice in q.options:
                choice_idx = q.options.index(saved_choice)
                
            selected_option = st.radio(
                "Options:",
                q.options,
                index=choice_idx,
                key=f"q_radio_{q.id}",
                label_visibility="collapsed"
            )
            # Store in answers state
            st.session_state.quiz_answers[q.id] = selected_option
            st.markdown("---")

        submit_btn = st.form_submit_button("Submit Answers")

    # Handle submission
    if submit_btn:
        st.session_state.quiz_submitted = True
        
        # Calculate score
        correct_count = 0
        quiz_summary = []
        for q in content.quiz:
            ans = st.session_state.quiz_answers.get(q.id)
            is_correct = (ans == q.correct_option)
            if is_correct:
                correct_count += 1
            quiz_summary.append({
                "id": q.id,
                "question": q.question,
                "selected": ans,
                "correct": q.correct_option,
                "is_correct": is_correct,
                "explanation": q.explanation
            })

        score_pct = (correct_count / len(content.quiz)) * 100

        # Save Attempt locally
        attempt_id = f"attempt_{int(datetime.now().timestamp())}"
        attempt = QuizAttempt(
            id=attempt_id,
            topic=content.topic,
            timestamp=datetime.now().isoformat(),
            score=correct_count,
            total_questions=len(content.quiz),
            percentage=score_pct,
            user_answers={str(qid): str(val) for qid, val in st.session_state.quiz_answers.items()}
        )
        st.session_state.storage_service.save_quiz_attempt(attempt)

    # Render results post submission
    if st.session_state.quiz_submitted:
        correct_count = sum(1 for q in content.quiz if st.session_state.quiz_answers.get(q.id) == q.correct_option)
        score_pct = (correct_count / len(content.quiz)) * 100
        
        st.markdown("### Quiz Results")
        if score_pct >= 70.0:
            st.success(f"🎉 Excellent! You scored **{correct_count}/10 ({score_pct:.1f}%)**")
        else:
            st.warning(f"📚 Keep practicing! You scored **{correct_count}/10 ({score_pct:.1f}%)**")

        st.markdown("---")
        st.markdown("### Review Explanations")
        for q in content.quiz:
            ans = st.session_state.quiz_answers.get(q.id)
            is_correct = (ans == q.correct_option)
            color = "#00FF66" if is_correct else "#FF3366"
            symbol = "✅" if is_correct else "❌"

            st.markdown(f"#### Q{q.id}. {q.question}")
            st.markdown(f"<p style='color: {color};'>Your Selection: {ans} {symbol}</p>", unsafe_allow_html=True)
            if not is_correct:
                st.markdown(f"Correct Option: **{q.correct_option}**")
            st.info(f"💡 **Explanation:** {q.explanation}")
            st.markdown("---")

        # Export Report
        st.markdown("#### Export Quiz Results")
        quiz_report = {
            "topic": content.topic,
            "timestamp": datetime.now().isoformat(),
            "score": correct_count,
            "percentage": score_pct,
            "answers": st.session_state.quiz_answers
        }
        st.download_button(
            label="📥 Download Quiz Report (JSON)",
            data=json.dumps(quiz_report, indent=2),
            file_name=f"quiz_report_{content.topic.lower()}_{int(datetime.now().timestamp())}.json",
            mime="application/json"
        )


# ==========================================
# PAGE 5: RAG KNOWLEDGE ASSISTANT
# ==========================================
elif page == "🔍 RAG Knowledge Assistant":
    st.markdown("<h1 class='cyber-title'>RAG Knowledge Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='cyber-subtitle'>Upload internal PDFs, docx manuals, or text guidelines to query context-augmented DevOps knowledge.</p>", unsafe_allow_html=True)

    if not st.session_state.gemini_service.is_configured():
        st.warning("⚠️ Google Gemini API key is missing. Please supply a key in the sidebar configuration.")
        st.stop()

    col_u1, col_u2 = st.columns([1, 2])

    with col_u1:
        st.markdown("### Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload manuals (PDF, DOCX, TXT):",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Save locally to dynamic uploads
                filepath = os.path.join(st.session_state.storage_service.uploads_dir, uploaded_file.name)
                
                # Avoid duplicate processing
                if not os.path.exists(filepath):
                    with open(filepath, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    with st.spinner(f"Ingesting {uploaded_file.name}..."):
                        try:
                            chunks_count = st.session_state.rag_service.ingest_document(filepath, uploaded_file.name)
                            st.success(f"Indexed {uploaded_file.name} ({chunks_count} chunks)")
                        except Exception as e:
                            st.error(f"Ingestion failed for {uploaded_file.name}: {e}")
                else:
                    st.caption(f"ℹ️ {uploaded_file.name} already ingested.")

        # Database Controls
        st.markdown("---")
        st.markdown("#### Database Status")
        if st.session_state.rag_service.vector_store is not None:
            st.success("✅ FAISS Vector Index Active")
            if st.button("Flush Vector Database"):
                st.session_state.rag_service.clear_vector_store()
                st.success("Vector store index cleared.")
                st.rerun()
        else:
            st.caption("FAISS database empty. Please upload files to initialize.")

    with col_u2:
        st.markdown("### Query Document Database")
        query = st.text_input("Enter your question based on uploaded documents:")
        search_k = st.slider("Context size (Top K matches):", min_value=1, max_value=5, value=3)

        if query:
            if st.session_state.rag_service.vector_store is None:
                st.warning("Please upload and index documents before querying.")
            else:
                with st.spinner("Searching database and formulating response..."):
                    # 1. Similarity search
                    docs = st.session_state.rag_service.similarity_search(query, k=search_k)

                    if not docs:
                        st.info("No matching content blocks found. Try adjusting keywords.")
                    else:
                        # Display source chunks in expander
                        with st.expander("📚 Source Context Chunks Used", expanded=True):
                            for idx, doc in enumerate(docs):
                                st.markdown(f"**Source {idx+1}:** `{doc.metadata.get('source')}` | Chunk: `{doc.metadata.get('chunk_index')}`")
                                st.markdown(f"_{doc.page_content}_")
                                st.markdown("---")

                        # 2. Context formulation
                        context = "\n---\n".join([f"[Source: {d.metadata.get('source')}] {d.page_content}" for d in docs])
                        
                        prompt = (
                            f"You are a helpful DevOps assistant. Below is context extracted from reference documents.\n"
                            f"Answer the user's question as thoroughly and clearly as possible based strictly on the provided context.\n"
                            f"If the answer cannot be found or inferred from the context, state that clearly.\n\n"
                            f"Retrieved Context:\n{context}\n\n"
                            f"User Question: {query}\n\n"
                            f"Detailed Answer:"
                        )

                        # Stream response
                        st.markdown("#### AI Augmented Answer:")
                        try:
                            # Use Gemini streaming capability
                            messages = [{"role": "user", "content": prompt}]
                            st.write_stream(st.session_state.gemini_service.stream_chat_response(messages))
                        except Exception as e:
                            st.error(f"Failed to generate answer: {e}")


# ==========================================
# PAGE 6: MENTOR CHAT
# ==========================================
elif page == "💬 Mentor Chat":
    st.markdown("<h1 class='cyber-title'>Mentor Chat</h1>", unsafe_allow_html=True)
    st.markdown("<p class='cyber-subtitle'>Discuss architectures, logs, configuration scripts, and DevOps concepts with your persistent AI Mentor.</p>", unsafe_allow_html=True)

    if not st.session_state.gemini_service.is_configured():
        st.warning("⚠️ Google Gemini API key is missing. Please supply a key in the sidebar configuration.")
        st.stop()

    storage = st.session_state.storage_service
    
    # Session selector / Creator
    sessions = storage.list_chat_sessions()
    
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.markdown("#### Chat Rooms")
        new_chat_btn = st.button("➕ Create New Discussion")
        if new_chat_btn:
            st.session_state.chat_session_id = str(uuid.uuid4())
            st.success("New discussion room opened!")
            st.rerun()

        if sessions:
            selected_session = st.selectbox(
                "Switch Room:",
                sessions,
                index=0,
                format_func=lambda x: f"Discussion {x[:8]}..."
            )
            if selected_session != st.session_state.chat_session_id:
                st.session_state.chat_session_id = selected_session
                st.rerun()
        else:
            st.caption("No conversations recorded yet. Type below to start.")

    with col_s2:
        # Load conversation
        history = storage.load_chat_history(st.session_state.chat_session_id)
        
        st.markdown(f"#### Active Room: `{st.session_state.chat_session_id[:8]}`")
        st.markdown("---")

        # Container for chat messages
        chat_container = st.container(height=450)
        with chat_container:
            if not history:
                st.info("👋 Hello! I am your DevOps AI Mentor. Ask me any technical questions, paste error logs, or request config snippets!")
            else:
                for msg in history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

        # Input box
        if prompt := st.chat_input("Ask a DevOps question..."):
            # 1. Render User message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # 2. Add to local history
            storage.save_chat_message(st.session_state.chat_session_id, "user", prompt)
            
            # 3. Stream Response
            with chat_container:
                with st.chat_message("assistant"):
                    # Retrieve full conversation history for model context
                    full_history = storage.load_chat_history(st.session_state.chat_session_id)
                    try:
                        response_stream = st.session_state.gemini_service.stream_chat_response(full_history)
                        full_response = st.write_stream(response_stream)
                        # Save Response
                        storage.save_chat_message(st.session_state.chat_session_id, "assistant", full_response)
                    except Exception as e:
                        st.error(f"Error streaming response: {e}")
