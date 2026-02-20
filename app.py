import streamlit as st
import os
import json
import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Constants and Configuration
PILLAR_FILE = "pillar_state.json"
USER_PROFILE = """
Name: Digvijay Chaudhari
Role: AI Systems Engineer & Consultant
Stack: Python, LangChain, RAG Pipelines, LLM Orchestration, n8n, AWS, Voice Agents
Projects:
  1. Alumniare — Multi-college alumni management SaaS
  2. RAG Chatbot — 98% accuracy, built with LangChain & Google Gemini
  3. AI Voice Agent — Autonomous meeting booking with custom knowledge base
  4. Taught AI to 300+ students at JSW Foundation (Microsoft/SBI funded)
Tone: Confident, technical, builder-focused, zero fluff
"""

POST_FORMAT = """
STRICT FORMAT RULES (NO EXCEPTIONS):

1. HOOK (Line 1-2):
   - Maximum 10 words
   - Bold claim or shocking stat
   - Stand alone on its own line
   - No full stops at end

2. STRUCTURE:
   - NO long paragraphs ever
   - Maximum 2 sentences per block
   - Every new point = new line
   - Add empty line between every point
   - Use numbered steps with emojis when explaining process:
     1️⃣ Step one
     2️⃣ Step two
     3️⃣ Step three

3. SENTENCES:
   - Maximum 12 words per sentence
   - If a sentence is longer → break it into 2
   - Use fragments intentionally:
     "No hallucinations. No excuses."
     "This is the difference."

4. NUMBERS AND STATS:
   - Always on their own line
   - Never buried in paragraphs

5. CTA (Last 3 lines always):
   - Line 1: Question or bold statement
   - Line 2: What to do (DM, connect, comment)
   - Line 3: What they get

6. HASHTAGS:
   - Exactly 5
   - One per line
   - Most specific first, broad last

7. TONE:
   - Reads like tweets stacked together
   - Confident, direct, zero fluff
   - Builder talking to builders

EXAMPLE OUTPUT (COPY THIS STRUCTURE):
---
I built a RAG chatbot that hits 98% accuracy.

No hallucinations. No made-up answers.

Here's exactly how 👇

The problem with most RAG systems:
They retrieve wrong context.
LLM fills gaps with confident lies.
Users stop trusting it.

Here's what I changed:

1️⃣ Semantic chunking
Not by character count.
By meaning.

2️⃣ Hybrid retrieval
Vector similarity + BM25.
Both signals. Better results.

3️⃣ Cross-encoder re-ranking
Filters noise before hitting Gemini.
Only best context gets through.

Result?
98% accuracy.
Zero hallucinations.
Production ready.

Building something that needs precision AI?
DM me — let's talk.

#RAG
#LangChain
#LLMEngineering
#AIBuilders
#PythonDeveloper
---
"""

PILLARS = [
    {"name": "Technical Deep Dive", "desc": "How I built something technical. Focus on architecture, code, or specific implementation details."},
    {"name": "Business ROI", "desc": "How AI saves time or money. Focus on metrics, efficiency, and business value."},
    {"name": "AI Trend + My Take", "desc": "Latest AI news/trend with my personal professional opinion."},
    {"name": "Project Build Log", "desc": "Update on what I am currently building (Alumniare, RAG Chatbot, or Voice Agent). Challenges faced, wins, progress."}
]

# Helper Functions
def get_todays_pillar():
    """Determine the active pillar for today."""
    today_str = datetime.date.today().isoformat()
    
    # Default state if file doesn't exist
    state = {"last_date": None, "current_pillar_index": -1}
    
    if os.path.exists(PILLAR_FILE):
        try:
            with open(PILLAR_FILE, "r") as f:
                state = json.load(f)
        except json.JSONDecodeError:
            pass # Use default state

    last_date = state.get("last_date")
    current_index = state.get("current_pillar_index", -1)

    if last_date != today_str:
        # It's a new day (or first run), rotate to next pillar
        # If it's the very first run (index -1), start at 0
        new_index = (current_index + 1) % 4
        
        # Update state immediately to lock in today's pillar
        state["last_date"] = today_str
        state["current_pillar_index"] = new_index
        with open(PILLAR_FILE, "w") as f:
            json.dump(state, f)
        
        return PILLARS[new_index]
    else:
        # Same day, return the stored pillar for today
        return PILLARS[current_index]

def fetch_detailed_trends(api_key):
    """Fetch AI trends using TWO Tavily searches as requested."""
    try:
        tavily = TavilyClient(api_key=api_key)
        
        # Search 1: General AI News
        news_response = tavily.search(query="latest AI news today 2026", search_depth="basic", max_results=1)
        news_result = news_response['results'][0] if news_response['results'] else None
        
        # Search 2: LinkedIn Specific
        linkedin_response = tavily.search(query="site:linkedin.com AI engineering articles", search_depth="basic", max_results=1)
        linkedin_result = linkedin_response['results'][0] if linkedin_response['results'] else None

        return news_result, linkedin_result
    except Exception as e:
        st.error(f"Error fetching trends: {e}")
        return None, None

def generate_post(pillar, topic_context, api_key):
    """Generate the LinkedIn post, image prompt, and carousel content using Gemini."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.7)
    
    parser = JsonOutputParser()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert LinkedIn ghostwriter for {user_profile}. You write viral, high-value content."),
        ("user", """
        Create a LinkedIn post based on the following Active Pillar:
        
        **Pillar Name**: {pillar_name}
        **Pillar Description**: {pillar_desc}
        
        **Context/Trend Info**: 
        {topic_context}
        
        **Format Requirements**:
        {post_format}
        
        **Specific Instructions for 'AI Trend + My Take' Pillar (if active)**:
        - Reference the LinkedIn article opinion/topic provided in the context.
        - Either agree with a unique angle OR respectfully disagree.
        - Make it feel like a response to the LinkedIn community.
        - Keep the tone confident, technical, and builder-focused.

        **ADDITIONAL REQUIREMENT 1: NAPKIN AI IMAGE PROMPT**:
        Generate a detailed image prompt optimized for Napkin AI.
        - The prompt should describe a clean, modern, tech-style visual.
        - Should match the post topic exactly.
        - Label it usually as "Napkin AI Image Prompt".

        **ADDITIONAL REQUIREMENT 2: CAROUSEL GENERATOR**:
        Generate a 10-slide carousel script based on the post topic.
        
        Structure each slide object with:
        - slide_number
        - emoji
        - headline (max 8 words)
        - bullet_points (3 points, max 10 words each)

        **ADDITIONAL REQUIREMENT 3: CLAUDE AI PROMPT**:
        Generate ONE section called "claude_prompt" which contains the entire carousel script formatted exactly as a prompt for Claude AI.
        
        Format it EXACTLY like this:
        ---
        Make me a LinkedIn carousel PDF with this script:

        TOPIC: [the topic]
        BRAND: Digvijay Chaudhari — AI Systems Engineer & Consultant
        BRAND COLORS: Navy #0F172A, Blue #3B82F6, Purple #8B5CF6, Green #10B981, White #F8FAFC
        FONT STYLE: Modern, minimal, tech professional
        
        SLIDE 1 — COVER:
        Headline: [headline]
        Subtext: [subtext]
        
        SLIDE 2:
        Headline: [headline]
        • [point 1]
        • [point 2]
        • [point 3]
        
        SLIDE 3:
        Headline: [headline]
        • [point 1]
        • [point 2]
        • [point 3]
        
        [... slides 4-9 same format ...]
        
        SLIDE 10 — CTA:
        Headline: Found this useful?
        • Follow Digvijay Chaudhari for daily AI builds
        • DM "BUILD" for AI consulting
        • Like + Repost to help other builders
        ---

        Output strictly in JSON format with the following keys:
        - "headline": (A short internal headline/topic for the post)
        - "linkedin_post": (The actual post content including hashtags)
        - "image_prompt": (The Napkin AI image prompt)
        - "carousel_script": (List of objects, each with "slide_number", "emoji", "headline", "bullet_points")
        - "claude_prompt": (The formatted prompt string for Claude)
        """)
    ])
    
    chain = prompt | llm | parser
    
    try:
        return chain.invoke({
            "user_profile": USER_PROFILE,
            "pillar_name": pillar['name'],
            "pillar_desc": pillar['desc'],
            "topic_context": topic_context,
            "post_format": POST_FORMAT
        })
    except Exception as e:
        st.error(f"Error generating content: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="LinkedIn Post Generator", page_icon="🚀", layout="centered")

# Custom CSS for Minimal Dark Theme
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #0072b1;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .stTextArea textarea {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    h1 {
        color: #ffffff;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0F172A;
        border-radius: 4px;
        color: #F8FAFC;
    }
    .stTabs [aria-selected="true"] {
        border-bottom-color: #3B82F6;
        color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 LinkedIn Post Generator")

# API Key Validation
gemini_key = os.getenv("GEMINI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

if not gemini_key or not tavily_key:
    st.error("⚠️ Missing API Keys! Please check your .env file.")
    st.info("Make sure you have GEMINI_API_KEY and TAVILY_API_KEY set.")
    st.stop()

# Pillar Logic
current_pillar = get_todays_pillar()

st.markdown(f"### 📅 Today's Pillar: **{current_pillar['name']}**")
st.info(f"ℹ️ {current_pillar['desc']}")

# Context Storage
if 'context_data' not in st.session_state:
    st.session_state.context_data = ""
if 'sources' not in st.session_state:
    st.session_state.sources = {}
if 'generation_result' not in st.session_state:
    st.session_state.generation_result = None

# Context Loading
if current_pillar['name'] == "AI Trend + My Take":
    if not st.session_state.context_data: # Only fetch once to save API calls
        with st.spinner("Fetching latest AI trends & LinkedIn discussions..."):
            news, linkedin = fetch_detailed_trends(tavily_key)
            
            context_str = ""
            if news:
                context_str += f"Global News: {news['title']} ({news['content']})\n"
                st.session_state.sources['news'] = news
            if linkedin:
                context_str += f"LinkedIn Discussion: {linkedin['title']} ({linkedin['content']})\n"
                st.session_state.sources['linkedin'] = linkedin
            
            st.session_state.context_data = context_str
            
            if context_str:
                st.success("Fetched latest trends!")
            else:
                st.warning("Could not fetch trends. Using general knowledge.")

# Display Sources
if st.session_state.sources:
    st.subheader("📰 Sources Used")
    cols = st.columns(2)
    with cols[0]:
        if 'news' in st.session_state.sources:
            n = st.session_state.sources['news']
            st.markdown(f"**Global News:**\n[{n['title']}]({n['url']})")
    with cols[1]:
        if 'linkedin' in st.session_state.sources:
            l = st.session_state.sources['linkedin']
            st.markdown(f"**LinkedIn Discussion:**\n[{l['title']}]({l['url']})")

# Generation Button
if st.button("Generate Today's Post"):
    with st.spinner("Brainstorming & writing..."):
        # Use fetched context if available, otherwise default fallback
        final_context = st.session_state.context_data if st.session_state.context_data else "Pick a relevant topic from my Projects list or general AI engineering expertise."
        st.session_state.generation_result = generate_post(current_pillar, final_context, gemini_key)

# Main UI Tabs
if st.session_state.generation_result:
    result = st.session_state.generation_result
    
    tab1, tab2 = st.tabs(["📝 Post Generator", "🎢 Carousel Generator"])
    
    with tab1:
        st.subheader("📝 LinkedIn Post")
        if "headline" in result and result["headline"]:
             st.caption(f"Topic: {result['headline']}")
        
        st.text_area("Copy Post", value=result.get("linkedin_post", ""), height=300)
        
        st.subheader("🎨 Napkin AI Image Prompt")
        st.text_area("Copy Image Prompt", value=result.get("image_prompt", ""), height=100)

    with tab2:
        st.subheader("🎢 10-Slide Carousel Script")
        
        if "carousel_script" in result:
            for slide in result["carousel_script"]:
                with st.expander(f"Slide {slide.get('slide_number', '?')} {slide.get('emoji', '')}"):
                    st.markdown(f"**{slide.get('headline', '')}**")
                    for point in slide.get('bullet_points', []):
                        st.markdown(f"• {point}")

        st.markdown("---")
        st.subheader("📋 Full Script — Copy & Send to Claude AI")
        st.info("Click the copy button below and paste into Claude to get your PDF design.")
        
        claude_val = result.get("claude_prompt", "")
        st.text_area("Claude Prompt", value=claude_val, height=300)
