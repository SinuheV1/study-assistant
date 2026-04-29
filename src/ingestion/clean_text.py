import re
from src.utils.logging import setup_logger

log = setup_logger(__name__)

def normalize_whitespace(text: str) -> str:
    #normalize tabs, repeated spaces, and blocks of blank lines
    text=text.replace('\t', ' ')
    text=re.sub(r'[ ]+',' ',text)
    text=re.sub(r'\n{3,}','\n\n',text)
    return text.strip()

def remove_trailing_whitespace(text: str) -> str:
    #clean line endings
    lines=[line.rstrip() for line in text.splitlines()]
    return '\n'.join(lines)

def fix_broken_line_wraps(text: str) -> str:
    #joins lines that were split in transcripts or bad export    
    lines=text.splitlines()
    cleaned_lines=[]
    buffer=''
    
    
    def is_list_item(line: str) -> bool:
        #list detection cleaning
        stripped = line.strip()
        if not stripped:
            return False

        if stripped.startswith(("- ", "* ")):
            return True

        #numbered-list detection: "1. item", "2. item", etc
        parts = stripped.split(maxsplit=1)
        if parts:
            token = parts[0]
            if token.endswith(".") and token[:-1].isdigit():
                return True

        return False
    1
    for line in lines:
        stripped=line.strip()
        #blank line = paragraph/list boundary
        if not stripped:
            if buffer:
                cleaned_lines.append(buffer.strip())
                buffer=''
            cleaned_lines.append('')
            continue
        #preserve list items on their own lines
        if is_list_item(stripped):
            if buffer:
                cleaned_lines.append(buffer.strip())
                buffer = ""
            cleaned_lines.append(stripped)
            continue
        
        #merge normal wrapped lines inside paragraphs
        if buffer and not buffer.endswith(('.',':','?','!')):
            buffer += ' ' + stripped
        else:
            if buffer:
                cleaned_lines.append(buffer.strip())
            buffer=stripped
            
    if buffer:
        cleaned_lines.append(buffer.strip())
    
    return '\n'.join(cleaned_lines)

def basic_text_cleaning(text: str) -> str:
    if text is None:
        log.warning('Received None instead of text in basic_text_cleaning.')
        return ''
    
    cleaned=remove_trailing_whitespace(text)
    cleaned=normalize_whitespace(cleaned)
    cleaned=fix_broken_line_wraps(cleaned)
    
    log.info('Completed basic text cleaning.')
    return cleaned