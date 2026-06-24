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

if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = ""

if "current_content" not in st.session_state:
    st.session_state.current_content = None

if "preset_topic_widget" not in st.session_state:
    st.session_state.preset_topic_widget = ""

if "custom_topic_widget" not in st.session_state:
    st.session_state.custom_topic_widget = ""

# Callbacks for widgets
def on_preset_change():
    val = st.session_state.preset_topic_widget
    st.session_state.selected_topic = val
    # Sync custom text input widget value to match
    st.session_state.custom_topic_widget = val
    # Clear loaded content since topic changed
    st.session_state.current_content = None
    st.session_state.current_topic = ""

def on_custom_change():
    val = st.session_state.custom_topic_widget.strip()
    st.session_state.selected_topic = val
    # Sync preset dropdown selection
    if val in CORE_TOPICS:
        st.session_state.preset_topic_widget = val
    else:
        st.session_state.preset_topic_widget = ""
    # Clear loaded content since topic changed
    st.session_state.current_content = None
    st.session_state.current_topic = ""

if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = str(uuid.uuid4())

if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = {}

if "cache_hit" not in st.session_state:
    st.session_state.cache_hit = False

# --- TEMPORARY DEBUG PANEL ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Gemini API Debug Panel")
st.sidebar.text(f"Model: {gemini_service.model_name}")
st.sidebar.text(f"Circuit: {gemini_service.circuit_state}")
st.sidebar.text(f"Fail Count: {gemini_service.failure_count}")
st.sidebar.text(f"Last Success: {gemini_service.last_api_success_time or 'Never'}")
if gemini_service.last_api_error:
    st.sidebar.error(f"Last Error: {gemini_service.last_api_error}")
else:
    st.sidebar.text("Last Error: None")

st.sidebar.markdown("---")
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

# Helper for sanitizing Mermaid diagram syntax
def sanitize_mermaid(mermaid_code: str) -> Optional[str]:
    if not mermaid_code:
        return None
    # 1. Remove markdown fences before rendering
    code = mermaid_code
    code = re.sub(r"^```mermaid\s*\n", "", code, flags=re.IGNORECASE)
    code = re.sub(r"^```\s*\n", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\n```$", "", code)
    code = code.strip()
    
    lines = code.splitlines()
    if not lines:
        return None
        
    # Find first non-comment line to check diagram type
    first_word = ""
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("%%"):
            parts = stripped.split()
            if parts:
                first_word = parts[0]
                break
                
    valid_prefixes = ("graph", "flowchart", "sequenceDiagram", "stateDiagram")
    if first_word not in valid_prefixes:
        return None
        
    if first_word in ("graph", "flowchart"):
        # Combined regex pattern for all node shapes to prevent double-matching
        id_pattern = r'([a-zA-Z0-9_][a-zA-Z0-9_-]*)'
        
        # Order matters: double shapes before single shapes
        shapes = [
            (r'\(\s*\[(.*?)\s*\]\s*\)', lambda lbl: f'([{lbl}])'),      # Stadium
            (r'\[\s*\((.*?)\s*\)\s*\]', lambda lbl: f'[({lbl})]'),      # Database
            (r'\(\s*\((.*?)\s*\)\s*\)', lambda lbl: f'(({lbl}))'),      # Circle
            (r'\{\s*\{(.*?)\s*\}\s*\}', lambda lbl: f'{{{{{lbl}}}}}'),  # Hexagon
            (r'\[\s*\[(.*?)\s*\]\s*\]', lambda lbl: f'[[{lbl}]]'),      # Subroutine
            (r'\[\s*(.*?)\s*\]',       lambda lbl: f'[{lbl}]'),        # Square
            (r'\(\s*(.*?)\s*\)',       lambda lbl: f'({lbl})'),        # Round
            (r'\{\s*(.*?)\s*\}',       lambda lbl: f'{{{lbl}}}'),      # Curly
            (r'>\s*(.*?)\s*\]',        lambda lbl: f'>{lbl}]'),        # Asymmetric
        ]
        
        # Build the combined pattern with explicit double capturing groups for all shapes
        pattern_parts = [f'({pat})' for pat, _ in shapes]
        combined_pattern = id_pattern + r'\s*(?:' + '|'.join(pattern_parts) + r')'
        
        def quote_label(label: str) -> str:
            label = label.strip()
            if not label:
                return '""'
            # If already fully quoted, unquote first to prevent double quoting
            if label.startswith('"') and label.endswith('"'):
                inner = label[1:-1]
                escaped_inner = inner.replace('\\"', '"').replace('"', '\\"')
                return f'"{escaped_inner}"'
            else:
                escaped_label = label.replace('"', '\\"')
                return f'"{escaped_label}"'

        def replace_node_double(match):
            node_id = match.group(1)
            for idx, (_, template_fn) in enumerate(shapes):
                whole_shape = match.group(2 * idx + 2)
                if whole_shape is not None:
                    label = match.group(2 * idx + 3)
                    quoted = quote_label(label)
                    return f'{node_id}{template_fn(quoted)}'
            return match.group(0)
            
        processed_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip diagram headers, comments, subgraphs, end, style, classDef, click, direction
            if (stripped.startswith(valid_prefixes) or 
                stripped.startswith(("%%", "subgraph", "end", "style", "linkStyle", "classDef", "class", "direction", "click"))):
                # Replace -> and => arrows in plain lines (e.g. connections in subgraph or main)
                # but preserve style declarations
                if not stripped.startswith(("style", "linkStyle", "classDef", "class", "%%")):
                    line = re.sub(r'(?<![-.])->(?![->])', '-->', line)
                    line = re.sub(r'(?<![=])=>(?![=>])', '==>', line)
                    line = re.sub(r'->>', '-->', line)
                    line = re.sub(r'-->>', '-->', line)
                processed_lines.append(line)
                continue
                
            # Replace arrows
            line = re.sub(r'(?<![-.])->(?![->])', '-->', line)
            line = re.sub(r'(?<![=])=>(?![=>])', '==>', line)
            line = re.sub(r'->>', '-->', line)
            line = re.sub(r'-->>', '-->', line)
            
            # Apply node shape sanitization
            line = re.sub(combined_pattern, replace_node_double, line)
            processed_lines.append(line)
            
        code = "\n".join(processed_lines)
        return code
        
    return code

# Helper for validating Mermaid diagram syntax structure
def validate_mermaid(code: str) -> bool:
    if not code:
        logger.error("MERMAID VALIDATION FAILED: Code is empty")
        return False
        
    lines = [line.strip() for line in code.splitlines() if line.strip() and not line.strip().startswith("%%")]
    if not lines:
        logger.error("MERMAID VALIDATION FAILED: No non-comment lines")
        return False
        
    first_line = lines[0]
    valid_prefixes = ("graph", "flowchart", "sequenceDiagram", "stateDiagram")
    first_word = first_line.split()[0] if first_line.split() else ""
    if first_word not in valid_prefixes:
        logger.error(f"MERMAID VALIDATION FAILED: Invalid start word '{first_word}'")
        return False
        
    # Check balanced brackets/delimiters
    bracket_pairs = [('[', ']'), ('(', ')'), ('{', '}'), ('"', '"')]
    for open_b, close_b in bracket_pairs:
        if open_b == close_b:
            if code.count(open_b) % 2 != 0:
                logger.error(f"MERMAID VALIDATION FAILED: Unbalanced delimiters '{open_b}'")
                return False
        else:
            if code.count(open_b) != code.count(close_b):
                logger.error(f"MERMAID VALIDATION FAILED: Unbalanced brackets '{open_b}' and '{close_b}'")
                return False
                
    logger.info("MERMAID VALIDATION SUCCESS")
    return True

# Helper for rendering Mermaid Diagrams in Streamlit
def display_mermaid(explanation: str):
    """
    Finds Mermaid code blocks, extracts them, sanitizes/validates them,
    and renders them using Mermaid CDN. If validation fails, shows a code block.
    """
    import streamlit.components.v1 as components
    import base64
    mermaid_blocks = re.findall(r"```mermaid\s*\n(.*?)\n```", explanation, re.DOTALL)
    
    if mermaid_blocks:
        split_text = re.split(r"```mermaid\s*\n.*?\n```", explanation, flags=re.DOTALL)
        
        for idx, text_part in enumerate(split_text):
            if text_part.strip():
                st.markdown(text_part)
            
            if idx < len(mermaid_blocks):
                mermaid_code = mermaid_blocks[idx]
                if not mermaid_code:
                    st.warning("⚠️ Diagram rendering unavailable. Empty diagram code.")
                    continue
                sanitized_code = sanitize_mermaid(mermaid_code)
                
                # Log all 4 outputs
                logger.info(f"1. RAW MERMAID GENERATED:\n{mermaid_code}")
                logger.info(f"2. CODE PASSED TO COMPONENTS:\n{sanitized_code}")
                logger.info(f"3. CODE AFTER SANITIZATION:\n{sanitized_code}")
                logger.info(f"4. CODE RENDERED IN BROWSER:\n{sanitized_code}")
                
                # Show raw code before rendering (debug mode)
                st.caption("🔍 Raw generated diagram text:")
                st.code(mermaid_code.strip(), language="mermaid")
                
                if sanitized_code and validate_mermaid(sanitized_code):
                    # Safe passing of code to Javascript via base64 encoding to prevent string quote escaping issues
                    encoded_code = base64.b64encode(sanitized_code.encode('utf-8')).decode('utf-8')
                    html_code = f"""
                    <div id="mermaid-container" style="background-color: #0c0f16; border: 1px solid rgba(0, 240, 255, 0.15); border-radius: 8px; padding: 15px; margin: 10px 0; overflow-x: auto;">
                        <div id="mermaid-svg" style="text-align: center;"></div>
                    </div>
                    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.6/dist/mermaid.min.js"></script>
                    <script>
                        (async function() {{
                            const container = document.getElementById('mermaid-container');
                            const svgDiv = document.getElementById('mermaid-svg');
                            
                            try {{
                                // Decode base64 UTF-8 string safely
                                const binary = atob("{encoded_code}");
                                const bytes = new Uint8Array(binary.length);
                                for (let i = 0; i < binary.length; i++) {{
                                    bytes[i] = binary.charCodeAt(i);
                                }}
                                const code = new TextDecoder('utf-8').decode(bytes);
                                
                                // Initialize mermaid manually with error rendering suppressed
                                mermaid.initialize({{
                                    startOnLoad: false,
                                    theme: 'dark',
                                    securityLevel: 'loose',
                                    suppressErrorRendering: true
                                }});
                                
                                // Parse validation first
                                const isValid = await mermaid.parse(code, {{ suppressErrors: true }});
                                if (!isValid) {{
                                    throw new Error("Mermaid parse validation failed");
                                }}
                                
                                // Render manually
                                const {{ svg }} = await mermaid.render('mermaid-svg-render', code);
                                svgDiv.innerHTML = svg;
                            }} catch (e) {{
                                console.error("Mermaid manual rendering failed:", e);
                                
                                // Re-decode code for display
                                let codeToDisplay = "";
                                try {{
                                    const binary = atob("{encoded_code}");
                                    const bytes = new Uint8Array(binary.length);
                                    for (let i = 0; i < binary.length; i++) {{
                                        bytes[i] = binary.charCodeAt(i);
                                    }}
                                    codeToDisplay = new TextDecoder('utf-8').decode(bytes);
                                }} catch(err) {{
                                    codeToDisplay = "Failed to load code source.";
                                }}
                                
                                // Style container as streamlit warning card + code block fallback
                                container.style.backgroundColor = 'transparent';
                                container.style.border = 'none';
                                container.style.padding = '0';
                                container.style.margin = '0';
                                
                                const escapedCode = codeToDisplay.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                                
                                container.innerHTML = `
                                    <div style="background-color: #ffe0b2; color: #b78103; border-left: 6px solid #fb8c00; border-radius: 4px; padding: 16px; margin: 10px 0; font-family: sans-serif; font-size: 14px; text-align: left; font-weight: 500;">
                                        ⚠️ Diagram rendering unavailable. Showing source diagram.
                                    </div>
                                    <pre style="background-color: #1e1e24; color: #e0e0e0; padding: 16px; border-radius: 4px; font-family: monospace; font-size: 14px; overflow-x: auto; margin: 10px 0; white-space: pre-wrap; word-break: break-all; text-align: left;">\${{escapedCode}}</pre>
                                `;
                            }}
                        }})();
                    </script>
                    """
                    components.html(html_code, height=450, scrolling=True)
                else:
                    st.warning("Diagram rendering unavailable. Showing source diagram.")
                    st.code(mermaid_code.strip(), language="mermaid")
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
            
            # Reset session state variables
            st.session_state.current_topic = ""
            st.session_state.selected_topic = ""
            st.session_state.current_content = None
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.evaluation_results = {}
            st.session_state.cache_hit = False
            
            st.success("All data purged successfully. Refreshing...")
            st.rerun()


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
        options = [""] + CORE_TOPICS
        selected = st.session_state.selected_topic
        
        default_index = 0
        if selected in CORE_TOPICS:
            default_index = options.index(selected)
            
        preset_topic = st.selectbox(
            "Select Core Topic:",
            options,
            index=default_index,
            key="preset_topic_widget",
            on_change=on_preset_change,
            help="Choose a pre-defined core topic or type in a custom topic to the left."
        )
    with col_input:
        custom_default = st.session_state.selected_topic
        custom_topic = st.text_input(
            "Enter Topic Name (e.g. Kubernetes, Ansible, Docker Multi-stage):",
            value=custom_default,
            key="custom_topic_widget",
            on_change=on_custom_change
        )

    generate_clicked = st.button("Generate DevOps Study Pack")
    
    cached_content = None
    if generate_clicked and st.session_state.selected_topic:
        cached_content = st.session_state.storage_service.load_from_cache(st.session_state.selected_topic)
        
    logger.info(f"Selected Topic: {st.session_state.selected_topic}")
    logger.info(f"Current Loaded Topic: {st.session_state.current_topic}")
    logger.info(f"Generate Clicked: {generate_clicked}")
    logger.info(f"Current Content Exists: {st.session_state.current_content is not None}")
    logger.info(f"Cache Hit: {cached_content is not None}")
    
    # Topic load handling
    if generate_clicked and st.session_state.selected_topic:
        topic_normalized = st.session_state.selected_topic
        
        if cached_content:
            st.session_state.current_topic = topic_normalized
            st.session_state.current_content = cached_content
            st.session_state.cache_hit = True
            # Mark studied
            st.session_state.storage_service.update_progress_on_completion(topic_normalized, completed=True)
            
            # Reset quiz state for the new topic
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.evaluation_results = {}
            
            st.success(f"⚡ Loaded {topic_normalized} from local cache!")
        else:
            # Force generate/fetch from Gemini
            st.session_state.cache_hit = False
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
                except Exception:
                    st.error("Failed to generate study pack. The generative service is temporarily offline or rate-limited. Showing fallback options.")
                    st.stop()

    # Render Content
    if st.session_state.current_content:
        content: DevOpsContent = st.session_state.current_content
        
        # Check if the content pack was generated offline
        if not st.session_state.get("cache_hit", False) and "(AI content generator temporarily unavailable" in content.explanation:
            st.warning("⚠️ AI content generator temporarily unavailable. Showing offline material.")
        
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
        st.info("Select a topic and click Generate DevOps Study Pack.")


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
                            import traceback
                            logger.error(f"Fallback triggered in Interview Prep. Exception Type: {type(e).__name__}, Message: {str(e)}\nStack Trace:\n{traceback.format_exc()}")
                            st.error("Failed to grade answer. The evaluation service is temporarily offline or rate-limited. Please try again.")

            # Display saved results
            if eval_key in st.session_state.evaluation_results:
                result = st.session_state.evaluation_results[eval_key]
                
                # Check for offline evaluation warning indicator
                if "AI evaluator temporarily unavailable" in result.improved_answer:
                    st.warning("⚠️ AI evaluator temporarily unavailable. Showing offline evaluation.")
                    
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
                        except Exception:
                            st.error(f"Ingestion failed for {uploaded_file.name}. The document processing service is temporarily offline or rate-limited.")
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
                        gemini_svc = st.session_state.gemini_service
                        if not gemini_svc.is_configured() or gemini_svc.circuit_state == "OPEN":
                            st.warning("⚠️ AI generation temporarily unavailable. Displaying retrieved knowledge.")
                        try:
                            # Use Gemini streaming capability
                            messages = [{"role": "user", "content": prompt}]
                            st.write_stream(st.session_state.gemini_service.stream_chat_response(messages))
                        except Exception as e:
                            import traceback
                            logger.error(f"Fallback triggered in RAG Knowledge Assistant. Exception Type: {type(e).__name__}, Message: {str(e)}\nStack Trace:\n{traceback.format_exc()}")
                            st.warning("⚠️ AI generation temporarily unavailable. Displaying retrieved knowledge.")
                            # fallback print
                            fallback_ans = gemini_svc._generate_offline_mentor_answer(prompt)
                            st.write(fallback_ans)


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
                        gemini_svc = st.session_state.gemini_service
                        if not gemini_svc.is_configured() or gemini_svc.circuit_state == "OPEN":
                            st.warning("⚠️ AI mentor temporarily unavailable. Showing offline guidance.")
                        response_stream = st.session_state.gemini_service.stream_chat_response(full_history)
                        full_response = st.write_stream(response_stream)
                        # Save Response
                        storage.save_chat_message(st.session_state.chat_session_id, "assistant", full_response)
                    except Exception as e:
                        import traceback
                        logger.error(f"Fallback triggered in Mentor Chat. Exception Type: {type(e).__name__}, Message: {str(e)}\nStack Trace:\n{traceback.format_exc()}")
                        st.warning("⚠️ AI mentor temporarily unavailable. Showing offline guidance.")
                        fallback_ans = gemini_svc._generate_offline_mentor_answer(prompt)
                        st.write(fallback_ans)
                        storage.save_chat_message(st.session_state.chat_session_id, "assistant", fallback_ans)
