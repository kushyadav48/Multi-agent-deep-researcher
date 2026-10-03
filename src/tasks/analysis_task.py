from crewai import Task

from src.agents.analyst_agent import create_analyst_agent


def create_analysis_task(
    query: str,
    search_results: str,
) -> Task:
    analyst_agent = create_analyst_agent()

    return Task(
        description=(
            f"Analyze the following research topic: {query}\n\n"

            "The Python application has already performed a real web search. "
            "The raw search results are provided directly below.\n\n"

            "================ RAW WEB EVIDENCE ================\n"
            f"{search_results}\n"
            "==================================================\n\n"

            "Use ONLY the raw web evidence above when making factual claims. "
            "Do not use facts from your internal knowledge.\n\n"

            "Carefully identify the information that directly answers the "
            "research question. Every factual claim must be traceable to text "
            "that actually appears in the raw evidence.\n\n"

            "Do not add information merely because you believe it is true. "
            "Do not infer unsupported dates, people, events, statistics, "
            "versions, quotations, or historical details.\n\n"

            "Preserve source titles and URLs EXACTLY as they appear in the "
            "raw evidence. Never create, rewrite, shorten, or guess a URL.\n\n"

            "If different search results disagree, explicitly describe the "
            "disagreement instead of deciding which unsupported interpretation "
            "must be correct.\n\n"

            "If the evidence is insufficient to verify something, state that "
            "it could not be verified from the supplied evidence."
        ),
        expected_output=(
            "An evidence-based analysis containing only claims supported by "
            "the supplied raw web evidence. For every important verified "
            "finding, include the exact supporting Source Title and Source URL. "
            "Clearly identify conflicting or insufficient evidence. Do not "
            "introduce unsupported factual information."
        ),
        agent=analyst_agent,
    )