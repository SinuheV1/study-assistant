import requests
from src.utils.logging import setup_logger

log = setup_logger(__name__)

def format_page_range(page_start, page_end):
    """
    Format page metadata for the generation context.
    Handles ints, strings, empty strings, and None values.
    """
    if page_start in [None, ""] and page_end in [None, ""]:
        return "unknown"

    if page_start == page_end:
        return str(page_start)

    return f"{page_start}-{page_end}"


def build_context_block(retrieved_results):
    if not retrieved_results:
        return ""

    context_parts = []

    for index, result in enumerate(retrieved_results, start=1):
        metadata = result.get("metadata", {})
        chunk_text = result.get("chunk_text", "")

        rank = result.get("rank", index)
        chunk_id = result.get("chunk_id") or metadata.get("chunk_id", "unknown")

        file_name = metadata.get("file_name", "unknown file")
        title = metadata.get("title", "unknown title")
        course = metadata.get("course", "unknown course")
        source_type = metadata.get("source_type", "unknown source type")
        chapter = metadata.get("chapter", "unknown chapter")
        section = metadata.get("section", "unknown section")

        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        pages = format_page_range(page_start, page_end)

        citation = f"{file_name}, {section}, pages {pages}"

        formatted_context_block = f"""
[Context Block {index}]
Rank: {rank}
File: {file_name}
Title: {title}
Course: {course}
Source Type: {source_type}
Chapter: {chapter}
Section: {section}
Pages: {pages}
Chunk ID: {chunk_id}
Citation: {citation}

Text:
{chunk_text}
""".strip()

        context_parts.append(formatted_context_block)

    context_block = "\n\n" + ("-" * 80) + "\n\n"
    context_block = context_block.join(context_parts)

    return context_block

knn_bias_variance_instruction = """
For K-nearest neighbors bias/variance questions:
- Smaller K uses fewer neighbors and creates a more flexible classifier.
- Smaller K can overfit, which usually means low bias and high variance.
- Larger K uses more neighbors and creates a smoother, less flexible classifier.
- Larger K can underfit, which usually means high bias and low variance.
- Check that the final answer does not contradict these relationships.
"""

def needs_bias_variance_guardrail(query: str) -> bool:
    query = query.lower()

    terms = [
        "bias",
        "variance",
        "tradeoff",
        "overfit",
        "underfit",
        "flexibility",
        "small k",
        "smaller k",
        "large k",
        "larger k",
    ]

    return any(term in query for term in terms)

def build_prompt(query,context_block):
    
    instructions="""
You are a study assistant.

Answer using only the provided context. If the context is insufficient, say so clearly.

Be clear and concise. Use bullet points when useful.

Answer the user's question directly. Do not add extra concepts unless they are needed to answer the question.

Ignore obvious extraction artifacts, broken margin labels, glossary fragments, or malformed phrases. Prefer the cleanest interpretation of the retrieved context.

Do not write a Sources section. Source citations will be appended automatically after your answer.

Do not invent source labels, source names, sections, or page numbers.

For definition questions, give:
- a concise definition
- a short explanation of how it works
- one important caveat only if the retrieved context clearly supports it

Do not write “Caveat: None mentioned.”

Do not discuss bias, variance, overfitting, underfitting, or model flexibility unless the question specifically asks about them.
"""
    if needs_bias_variance_guardrail(query):
        instructions += "\n" + knn_bias_variance_instruction
    
    prompt=instructions +'\nContext:\n'+ context_block +'\nQuestion:\n'+ query
    return prompt

def call_ollama(prompt,model_name):
        try:
            base_url = "http://localhost:11434/api/generate"

            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 400,
                },
            }

            request = requests.post(base_url, json=payload)
            request.raise_for_status()

            data = request.json()
            response = data.get("response", "")

            return response.strip()

        except Exception as e:
            log.warning(f"Request failed. Reason: {e}")
            return None
    
def generate_answer(query, retrieved_results, model_name):
    if not query:
        log.warning("Query is empty.")
        return None

    if not retrieved_results:
        return "I do not have enough context to answer."

    context_block = build_context_block(retrieved_results)
    prompt = build_prompt(query, context_block)

    answer = call_ollama(prompt, model_name)

    if not answer:
        return "I was unable to generate an answer."

    return format_answer_with_sources(answer, retrieved_results)

def strip_llm_sources(answer: str) -> str:
    if not answer:
        return answer

    markers = ["\nSources:", "\nSource:"]
    cleaned = answer

    for marker in markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0].strip()

    return cleaned

def format_source_line(result):
    metadata = result.get("metadata", {})

    file_name = metadata.get("file_name", "unknown file")
    section = metadata.get("section", "unknown section")
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")

    pages = format_page_range(page_start, page_end)

    return f"- {file_name}, {section}, pages {pages}"


def format_answer_with_sources(answer, retrieved_results):
    if not answer:
        return answer

    answer = strip_llm_sources(answer)

    if not retrieved_results:
        return answer

    source_lines = []
    seen_sources = set()

    for result in retrieved_results[:3]:
        metadata = result.get("metadata", {})

        file_name = metadata.get("file_name", "unknown file")
        section = metadata.get("section", "unknown section")
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        pages = format_page_range(page_start, page_end)

        source_line = f"- {file_name}, {section}, pages {pages}"

        if source_line in seen_sources:
            continue

        seen_sources.add(source_line)
        source_lines.append(source_line)

    sources_block = "\n".join(source_lines)

    return f"{answer.strip()}\n\nSources:\n{sources_block}"