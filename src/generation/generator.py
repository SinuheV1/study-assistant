import requests
from src.utils.logging import setup_logger

log = setup_logger(__name__)

def build_context_block(retrieved_results):
    if not retrieved_results:
        return ''
    context_parts=[]
    for result in retrieved_results:
        metadata=result.get('metadata',{})
        chunk_text=result.get('chunk_text','')
        source_type=metadata.get('source_type')
        title=metadata.get('title')
        rank=result.get('rank')
        
        formatted_context_block=f'| Source {rank} | {source_type} | {title} |\n {chunk_text} '
        
        context_parts.append(formatted_context_block)

    context_block="\n\n--\n\n".join(context_parts)

    return context_block

def build_prompt(query,context_block):
    
    instructions='''
    You are a study assistant.
    Answer using only the context.
    If context is insufficient, say so.
    Be clear and concise.
    Use bullet points when useful.
    '''
    prompt=instructions +'\nContext:\n'+ context_block +'\nQuestion:\n'+ query
    return prompt

def call_ollama(prompt,model_name):
    try:
        base_url='http://localhost:11434/api/generate'
        payload={
            'model':model_name,
            'prompt':prompt,
            'stream':False}
        request=requests.post(base_url,json=payload)
        request.raise_for_status()
        data=request.json()
        response=data['response']
        return response
    except Exception as e:
        log.warning(f'Request failed. Reason: {e}')
        return None
    
def generate_answer(query,retrieved_results,model_name):
    if not query:
        log.warning(f'Query is empty.')
        return None
    if not retrieved_results:
        return 'I do not have enough context to answer. '
    context_block=build_context_block(retrieved_results)
    prompt=build_prompt(query,context_block)
    answer=call_ollama(prompt,model_name)
    return answer

def format_answer_with_sources(answer,retrieved_results):
    if not retrieved_results:
        return answer
    sources=[]
    for result in retrieved_results:
        
        metadata=result.get('metadata',{})
        title=metadata.get('title')
        source_type=metadata.get('source_type')
        rank=result.get('rank')
        sources.append(f'- {title} | {source_type} | rank {rank}')
    sources_block='\n'.join(sources)
    formatted_answer=answer + f'\n\nSources:\n{sources_block}'
    return formatted_answer