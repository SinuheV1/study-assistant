from pathlib import Path

from pypdf import PdfReader, PdfWriter

INPUT_PDF = Path("data/raw/textbooks/ISLP_website.pdf")
OUTPUT_PDF = Path("data/raw/textbooks/ISLP_chapter_2.pdf")

# These are 1-indexed PDF page numbers based on the parsed PDF output.
# Python/pypdf uses 0-indexed page indices, so we subtract 1 below.
START_PAGE = 25
END_PAGE = 78


def extract_pdf_page_range(
    input_pdf: Path,
    output_pdf: Path,
    start_page: int,
    end_page: int,
) -> None:
    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    reader = PdfReader(str(input_pdf))
    writer = PdfWriter()

    total_pages = len(reader.pages)

    if start_page < 1:
        raise ValueError("start_page must be 1 or greater")

    if end_page > total_pages:
        raise ValueError(f"end_page {end_page} exceeds total PDF pages {total_pages}")

    if start_page > end_page:
        raise ValueError("start_page cannot be greater than end_page")

    for page_number in range(start_page, end_page + 1):
        page_index = page_number - 1
        writer.add_page(reader.pages[page_index])

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with output_pdf.open("wb") as f:
        writer.write(f)

    print(f"Extracted pages {start_page}-{end_page}")
    print(f"Saved to: {output_pdf}")


if __name__ == "__main__":
    extract_pdf_page_range(
        input_pdf=INPUT_PDF,
        output_pdf=OUTPUT_PDF,
        start_page=START_PAGE,
        end_page=END_PAGE,
    )
