import os
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool

app = FastAPI(
    title="CrewAI Projects API",
    description="FastAPI backend for six CrewAI projects.",
    version="1.0.0",
)


class Query(BaseModel):
    input: str = Field(..., min_length=1, description="Text input for the selected CrewAI project")


def get_llm():
    """Create the Gemini language model using a server-side environment variable."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured on the server.",
        )
    return LLM(
        model="gemini/gemini-3.1-flash-lite-preview",
        api_key=api_key,
        temperature=0.3,
    )


@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web and return titles, URLs, and descriptions."""
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No search results found."
        return "\n".join(
            f"Title: {item.get('title', '')}\nURL: {item.get('href', '')}\n"
            f"Description: {item.get('body', '')}"
            for item in results
        )
    except Exception as exc:
        return f"Search error: {exc}"


@tool("Website Reader")
def website_reader(url: str) -> str:
    """Read the visible text from an HTTP or HTTPS website."""
    try:
        if not url.startswith(("http://", "https://")):
            return "Invalid URL. Use an http:// or https:// address."
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(" ", strip=True)[:12000]
    except Exception as exc:
        return f"Website reading error for {url}: {exc}"


def run_blog_project(topic: str) -> str:
    llm = get_llm()
    researcher = Agent(
        role="Researcher",
        goal="Research the given topic and collect accurate information",
        backstory="You are an expert internet researcher.",
        llm=llm,
        tools=[duckduckgo_search, website_reader],
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    writer = Agent(
        role="Blog Writer",
        goal="Write an informative and engaging 600-word blog post",
        backstory="You are a professional blog writer.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    editor = Agent(
        role="Editor",
        goal="Correct grammar, improve tone, and check factual consistency",
        backstory="You are an experienced professional editor.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    research_task = Task(
        description=(
            "Research the topic: {topic}. Collect accurate information, "
            "examples, and source URLs. Do not invent facts."
        ),
        expected_output="Detailed research information with facts and URLs.",
        agent=researcher,
    )
    writing_task = Task(
        description=(
            "Write an informative blog post of about 600 words about {topic} "
            "using the research collected. Use simple English."
        ),
        expected_output="A well-researched and informative blog post.",
        agent=writer,
        context=[research_task],
    )
    editing_task = Task(
        description="Edit the blog post. Correct grammar, spelling, and sentence structure.",
        expected_output="A final polished blog post.",
        agent=editor,
        context=[research_task, writing_task],
    )
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, writing_task, editing_task],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff(inputs={"topic": topic}))


def run_competitor_project(competitors: str) -> str:
    llm = get_llm()
    manager = Agent(
        role="Manager Agent",
        goal="Coordinate the competitor research and analysis",
        backstory="You are an experienced project manager.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    site_agent = Agent(
        role="Site Researcher",
        goal="Collect website information for each competitor",
        backstory="You are an expert internet researcher.",
        llm=llm,
        tools=[website_reader],
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    pricing_agent = Agent(
        role="Pricing Analyst",
        goal="Analyze products, plans, prices, and offers",
        backstory="You are a pricing expert.",
        llm=llm,
        tools=[website_reader],
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    news_agent = Agent(
        role="News Analyst",
        goal="Find recent news and market updates",
        backstory="You track market developments.",
        llm=llm,
        tools=[duckduckgo_search],
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    strategist = Agent(
        role="Strategist",
        goal="Prepare a competitor comparison and SWOT analysis",
        backstory="You identify business strengths, weaknesses, opportunities, and threats.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    task1 = Task(
        description=(
            "Analyze these competitors and their website URLs: {competitors}. "
            "Use the website reader where URLs are supplied. Collect company information, "
            "products, features, target customers, and other relevant information."
        ),
        expected_output="Detailed competitor overview.",
        agent=site_agent,
    )
    task2 = Task(
        description=(
            "Analyze the same competitors: {competitors}. Inspect their products, "
            "plans, available prices, and offers using supplied website URLs."
        ),
        expected_output="Product and pricing comparison. Mark unavailable information clearly.",
        agent=pricing_agent,
    )
    task3 = Task(
        description=(
            "Search for recent news, product launches, and announcements about: {competitors}."
        ),
        expected_output="Recent competitor news with source URLs.",
        agent=news_agent,
    )
    task4 = Task(
        description=(
            "Prepare a competitor analysis report using the research, pricing, and news. "
            "Include competitor overview, product comparison, pricing comparison, recent "
            "updates, SWOT analysis, and conclusion. Use tables where useful. Do not invent facts."
        ),
        expected_output="Complete competitor analysis report.",
        agent=strategist,
        context=[task1, task2, task3],
    )
    crew = Crew(
        agents=[manager, site_agent, pricing_agent, news_agent, strategist],
        tasks=[task1, task2, task3, task4],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff(inputs={"competitors": competitors}))


def run_meeting_project(transcript: str) -> str:
    llm = get_llm()
    summarizer = Agent(
        role="Meeting Summarizer",
        goal="Summarize the meeting transcript in five lines",
        backstory="You are good at summarizing meetings.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    action_finder = Agent(
        role="Action Finder",
        goal="Find meeting tasks, owners, and due dates",
        backstory="You identify important meeting actions.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    decision_logger = Agent(
        role="Decision Logger",
        goal="Find and record important meeting decisions",
        backstory="You identify important decisions.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    formatter = Agent(
        role="Meeting Notes Formatter",
        goal="Combine the summary, action items, and decisions into clear notes",
        backstory="You organize meeting notes clearly.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    task1 = Task(
        description="Read this meeting transcript:\n{transcript}\nPrepare a five-line summary.",
        expected_output="Five-line meeting summary.",
        agent=summarizer,
    )
    task2 = Task(
        description="Read this meeting transcript:\n{transcript}\nFind tasks, owners, and due dates.",
        expected_output="Action items with owners and due dates.",
        agent=action_finder,
    )
    task3 = Task(
        description="Read this meeting transcript:\n{transcript}\nFind important decisions.",
        expected_output="List of important meeting decisions.",
        agent=decision_logger,
    )
    task4 = Task(
        description="Combine the summary, action items, and decisions into clear meeting notes.",
        expected_output="Final structured meeting notes.",
        agent=formatter,
        context=[task1, task2, task3],
    )
    crew = Crew(
        agents=[summarizer, action_finder, decision_logger, formatter],
        tasks=[task1, task2, task3, task4],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff(inputs={"transcript": transcript}))


def run_sales_project(csv_text: str) -> str:
    llm = get_llm()
    try:
        dataframe = pd.read_csv(StringIO(csv_text))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV text: {exc}")

    if dataframe.empty:
        raise HTTPException(status_code=400, detail="The supplied CSV has no data rows.")

    numeric = dataframe.select_dtypes(include="number")
    if numeric.empty:
        raise HTTPException(
            status_code=400,
            detail="The CSV must contain at least one numeric column for sales analysis.",
        )

    # Calculate numerical facts from the actual CSV instead of asking the LLM to calculate them.
    actual_summary = numeric.describe().to_string()
    sample_rows = dataframe.head(20).to_string(index=False)
    dataset_context = (
        f"CSV columns: {list(dataframe.columns)}\n"
        f"Number of rows: {len(dataframe)}\n"
        f"Numeric statistics:\n{actual_summary}\n"
        f"First 20 rows:\n{sample_rows}"
    )

    analyst = Agent(
        role="Data Analyst",
        goal="Analyze the supplied sales data",
        backstory="You are a careful data analyst.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    insight_writer = Agent(
        role="Insight Writer",
        goal="Identify the three most important findings",
        backstory="You explain data insights in simple language.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    reviewer = Agent(
        role="Reviewer",
        goal="Check the analysis against the supplied statistics",
        backstory="You carefully verify reported numbers.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    task1 = Task(
        description=(
            "Analyze the actual CSV information below. Explain total, average, highest, "
            "and lowest values where appropriate. Use only the provided statistics and data.\n"
            "{dataset_context}"
        ),
        expected_output="Sales analysis based on the actual CSV.",
        agent=analyst,
    )
    task2 = Task(
        description=(
            "Identify the top three important sales findings and explain them simply. "
            "Do not make up figures."
        ),
        expected_output="Three important sales findings.",
        agent=insight_writer,
        context=[task1],
    )
    task3 = Task(
        description=(
            "Check every number in the analysis against the supplied CSV statistics. "
            "Correct unsupported or incorrect claims."
        ),
        expected_output="Verified sales report.",
        agent=reviewer,
        context=[task1, task2],
    )
    crew = Crew(
        agents=[analyst, insight_writer, reviewer],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff(inputs={"dataset_context": dataset_context}))


def run_support_project(ticket: str) -> str:
    llm = get_llm()
    prompt = (
        "Classify this customer support ticket into exactly one category: "
        "billing, account-access, or other.\n"
        f"Ticket: {ticket}\n"
        "Return only the category."
    )
    category = str(llm.call(prompt)).strip().lower()

    if "billing" in category:
        return "Category: billing. The ticket has been routed to the billing team. Priority: high."
    if "account" in category or "access" in category:
        return "Category: account-access. The ticket has been routed to the account access team. Priority: high."
    return "Category: other. The ticket has been routed to the technical support team."


def run_approval_report_project(user_input: str) -> str:
    """
    Expected input format:
    Topic: ...
    Audience: ...
    Decision: approve
    Decision may be approve, reject, or omitted for a draft.
    """
    llm = get_llm()
    topic = ""
    audience = "general audience"
    decision = ""

    for line in user_input.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        key = key.strip().lower()
        value = value.strip()
        if key == "topic":
            topic = value
        elif key == "audience":
            audience = value or audience
        elif key == "decision":
            decision = value.lower()

    if not topic:
        topic = user_input.strip()

    researcher = Agent(
        role="Researcher",
        goal="Find useful information about the report topic",
        backstory="You are a careful web researcher.",
        llm=llm,
        tools=[duckduckgo_search],
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    analyst = Agent(
        role="Research Analyst",
        goal="Analyze the research and identify important findings",
        backstory="You analyze research carefully.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    checker = Agent(
        role="Fact Checker",
        goal="Check the findings and avoid unsupported claims",
        backstory="You verify factual accuracy.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    research_task = Task(
        description="Research this report topic: {topic}. Provide facts and source URLs.",
        expected_output="Research findings with sources.",
        agent=researcher,
    )
    analysis_task = Task(
        description="Analyze the research and identify important findings.",
        expected_output="Important findings.",
        agent=analyst,
        context=[research_task],
    )
    checking_task = Task(
        description="Check the findings for accuracy and identify uncertainty.",
        expected_output="Checked findings.",
        agent=checker,
        context=[research_task, analysis_task],
    )
    research_crew = Crew(
        agents=[researcher, analyst, checker],
        tasks=[research_task, analysis_task, checking_task],
        process=Process.sequential,
        verbose=True,
    )
    research_result = str(research_crew.kickoff(inputs={"topic": topic}))

    writer = Agent(
        role="Report Writer",
        goal="Write a clear report for the target audience",
        backstory="You write structured reports.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
    write_task = Task(
        description=(
            "Write a report about {topic} for this audience: {audience}. "
            "Use the checked research below. Include introduction, main findings, "
            "analysis, and conclusion. Do not invent facts.\n{research}"
        ),
        expected_output="A complete report draft.",
        agent=writer,
    )
    writing_crew = Crew(
        agents=[writer],
        tasks=[write_task],
        process=Process.sequential,
        verbose=True,
    )
    report = str(
        writing_crew.kickoff(
            inputs={
                "topic": topic,
                "audience": audience,
                "research": research_result,
            }
        )
    )

    if decision == "reject":
        reviser = Agent(
            role="Report Reviser",
            goal="Revise the report to improve clarity and completeness",
            backstory="You revise reports based on feedback.",
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
        )
        revision_task = Task(
            description="Revise this report while preserving factual accuracy:\n{report}",
            expected_output="Revised report.",
            agent=reviser,
        )
        revision_crew = Crew(
            agents=[reviser],
            tasks=[revision_task],
            process=Process.sequential,
            verbose=True,
        )
        report = str(revision_crew.kickoff(inputs={"report": report}))
        return "Decision: rejected. Revised report:\n\n" + report

    if decision == "approve":
        return "Decision: approved. Final report:\n\n" + report

    return (
        "Report draft created. To record an approval decision, send the topic and audience "
        "again with a line 'Decision: approve' or 'Decision: reject'.\n\n"
        + report
    )


@app.get("/")
def home():
    return {
        "message": "CrewAI Projects API is running.",
        "available_projects": [
            "/agent1", "/agent2", "/agent3", "/agent4", "/agent5", "/agent6"
        ],
        "docs": "/docs",
    }


@app.post("/agent1")
def agent1(q: Query):
    try:
        return {"output": run_blog_project(q.input)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Blog project failed: {exc}")


@app.post("/agent2")
def agent2(q: Query):
    try:
        return {"output": run_competitor_project(q.input)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Competitor analysis failed: {exc}")


@app.post("/agent3")
def agent3(q: Query):
    try:
        return {"output": run_meeting_project(q.input)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Meeting summarizer failed: {exc}")


@app.post("/agent4")
def agent4(q: Query):
    try:
        return {"output": run_sales_project(q.input)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sales analysis failed: {exc}")


@app.post("/agent5")
def agent5(q: Query):
    try:
        return {"output": run_support_project(q.input)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Customer support project failed: {exc}")


@app.post("/agent6")
def agent6(q: Query):
    try:
        return {"output": run_approval_report_project(q.input)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Approval report project failed: {exc}")
