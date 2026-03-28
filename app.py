"""
app.py — AI Personalised Study Planner
"""
import os, re, shutil
from datetime import date, timedelta
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="AI Study Planner", page_icon="📚", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main-title { font-family:'Space Mono',monospace; font-size:2.2rem; font-weight:700; color:#6C63FF; letter-spacing:-1px; margin-bottom:0; }
.sub-title { color:#888; font-size:.95rem; margin-top:0; margin-bottom:1.5rem; }
.sec { font-family:'Space Mono',monospace; font-size:1rem; color:#6C63FF; border-bottom:1px solid #6C63FF44; padding-bottom:.4rem; margin:1.4rem 0 .9rem; }
.info-card { background:linear-gradient(135deg,#1a1a2e,#16213e); border:1px solid #6C63FF44; border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1rem; }
.topic-card { background:linear-gradient(135deg,#16213e,#0f3460); border-left:4px solid #6C63FF; border-radius:8px; padding:.9rem 1.2rem; margin-bottom:.6rem; }
.day-header { font-family:'Space Mono',monospace; font-size:1rem; color:#6C63FF; font-weight:700; }
.high-priority { border-left-color:#FF4B4B !important; }
.med-priority  { border-left-color:#FFB347 !important; }
.low-priority  { border-left-color:#00C851 !important; }
.stButton>button { background:linear-gradient(135deg,#6C63FF,#4B44CC)!important; color:white!important; border:none!important; border-radius:8px!important; font-family:'Space Mono',monospace!important; font-weight:700!important; padding:.6rem 2rem!important; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📚 AI Study Planner")
    st.markdown("---")
    st.markdown("""### How it works\n1. 📄 Upload **syllabus PDF**\n2. 📅 Enter **exam date**\n3. 🤖 AI analyses using **RAG**\n4. 📋 Get **day-by-day plan**""")
    st.markdown("---")
    st.markdown("""### Tech Stack\n- 🦙 Groq Llama-3.1\n- 🔗 LangChain\n- 🗄️ ChromaDB\n- 🤗 HuggingFace Embeddings\n- 📄 PyPDF""")
    st.markdown("---")
    st.markdown("<small>Built by **Dixitha BV** · AIML 6th Sem<br>[GitHub](https://github.com/DixithaBV) · [LinkedIn](https://linkedin.com/in/dixitha-bv-9467a9359)</small>", unsafe_allow_html=True)

def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    return "".join(page.extract_text() or "" for page in reader.pages).strip()

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

def build_vectorstore(text: str, persist_dir: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    return Chroma.from_documents(documents=chunks, embedding=get_embeddings(), persist_directory=persist_dir)

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in .env file!")
        st.stop()
    return ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0.3, max_tokens=4096)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def extract_topics(vectorstore, llm) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke("List all units topics and subtopics from this syllabus")
    context = format_docs(docs)
    prompt = ChatPromptTemplate.from_template("""You are an expert academic assistant.
Using the syllabus content below, extract ALL topics, units, and subtopics.
Syllabus Content: {context}
List every topic/unit/chapter. Format: Unit name with subtopics as bullet points.""")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context})

def generate_plan(topics: str, exam_date: date, subject: str, llm) -> str:
    today = date.today()
    days_left = (exam_date - today).days
    if days_left <= 0:
        return "Exam date has passed!"
    prompt = ChatPromptTemplate.from_template("{input}")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"input": f"""You are an expert academic coach.
Subject: {subject}
Days available: {days_left} (exam on {exam_date})
Topics: {topics}

Create a day-by-day study plan using EXACTLY this format for each day:

DAY 1 ({today.strftime('%d %b')})
- Topic: [topic name]
- Priority: HIGH
- Time: 2 hours
- Focus: [what to study]

DAY 2 ({(today+timedelta(days=1)).strftime('%d %b')})
- Topic: [topic name]
- Priority: MEDIUM
- Time: 1.5 hours
- Focus: [what to study]

Continue for all {min(days_left,30)} days. Every 5th day = Revision. Last 2 days = Full revision.

After all days add:
REVISION REMINDERS:
- Review HIGH priority topics first
- Practice past questions daily
- Take breaks every hour

EXAM DAY TIPS:
- Sleep 8 hours before exam
- Stay calm and confident"""})

def parse_plan(plan_text: str) -> list:
    days = []
    for block in re.split(r'\n(?=DAY \d+)', plan_text):
        if block.strip().startswith("DAY"):
            lines = [l for l in block.strip().split('\n') if l.strip()]
            if lines:
                days.append({"header": lines[0], "details": lines[1:]})
    return days

st.markdown('<p class="main-title">📚 AI Personalised Study Planner</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload syllabus PDF → AI generates your complete study plan · RAG + Groq Llama-3.1 · Built by Dixitha BV</p>', unsafe_allow_html=True)
st.markdown('<p class="sec">📄 Upload Your Syllabus</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader("Upload Syllabus PDF", type=["pdf"])
with col2:
    subject_name = st.text_input("Subject Name", placeholder="e.g. Machine Learning")
    exam_date = st.date_input("Exam Date", min_value=date.today(), value=date.today()+timedelta(days=14))

days_left = (exam_date - date.today()).days
if days_left > 0:
    st.info(f"📅 You have **{days_left} days** until your exam on **{exam_date.strftime('%d %B %Y')}**")
else:
    st.warning("⚠️ Please select a future exam date!")

st.markdown("")
generate_btn = st.button("🤖 Generate My Study Plan")

if generate_btn:
    if not uploaded_file:
        st.error("❌ Please upload your syllabus PDF!")
    elif not subject_name.strip():
        st.error("❌ Please enter the subject name!")
    elif days_left <= 0:
        st.error("❌ Please select a future exam date!")
    else:
        with st.spinner("📄 Reading syllabus PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
            if len(pdf_text) < 50:
                st.error("❌ Could not extract text from PDF.")
                st.stop()
            st.success(f"✅ Extracted {len(pdf_text):,} characters from PDF")

        with st.spinner("🗄️ Building ChromaDB knowledge base..."):
            persist_dir = f"chroma_db/{subject_name.replace(' ','_')}"
            vectorstore = build_vectorstore(pdf_text, persist_dir)
            st.success("✅ Knowledge base ready!")

        with st.spinner("🔍 Analysing syllabus with RAG..."):
            llm = get_llm()
            topics = extract_topics(vectorstore, llm)
            st.success("✅ Topics extracted!")

        with st.spinner("🤖 Generating study plan with Llama-3.1..."):
            study_plan = generate_plan(topics, exam_date, subject_name, llm)

        st.balloons()
        st.success("🎉 Your personalised study plan is ready!")

        tab1, tab2, tab3 = st.tabs(["📋 Study Plan", "📚 Topics Found", "📥 Download"])

        with tab1:
            st.markdown('<p class="sec">📋 Your Day-by-Day Study Plan</p>', unsafe_allow_html=True)
            days = parse_plan(study_plan)
            if days:
                for day in days:
                    day_text = " ".join(day["details"]).lower()
                    pc = "high-priority" if "high" in day_text else "med-priority" if "medium" in day_text else "low-priority"
                    details_html = "<br>".join([f"&nbsp;&nbsp;{d}" for d in day["details"] if d.strip()])
                    st.markdown(f'<div class="topic-card {pc}"><div class="day-header">{day["header"]}</div><div style="color:#ccc;font-size:.85rem;margin-top:.4rem">{details_html}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(study_plan)

        with tab2:
            st.markdown('<p class="sec">📚 Topics Found</p>', unsafe_allow_html=True)
            st.markdown(topics)

        with tab3:
            st.markdown('<p class="sec">📥 Download</p>', unsafe_allow_html=True)
            content = f"AI STUDY PLAN\nSubject: {subject_name}\nExam: {exam_date}\nDays: {days_left}\n{'='*40}\nTOPICS:\n{topics}\n{'='*40}\nPLAN:\n{study_plan}"
            st.download_button("📥 Download Study Plan (.txt)", data=content, file_name=f"{subject_name}_study_plan.txt", mime="text/plain")

else:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    for col, title, desc in zip([c1,c2,c3], ["📄 Step 1","🤖 Step 2","📋 Step 3"], ["Upload your syllabus PDF — any subject","AI reads syllabus using RAG + ChromaDB","Get day-by-day plan with priorities & time"]):
        col.markdown(f'<div class="info-card"><h4 style="color:#6C63FF">{title}</h4><p style="color:#ccc">{desc}</p></div>', unsafe_allow_html=True)