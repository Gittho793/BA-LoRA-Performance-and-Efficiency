"""
Extract text by chapter from PDFs using their Table of Contents (if available).
Saves each chapter as a separate .txt file. If no TOC is found, saves the entire PDF as one .txt file.
"""
import re
import fitz  # PyMuPDF
import pathlib
import os
from typing import Generator, Optional, Any, LiteralString

PDF_PATH = pathlib.Path("./pdfs")
TXT_PATH = pathlib.Path("./txt_out4/")


def strip_page_numbers(text: str) -> str:
    """
    Remove typical page‑number lines such as '123', '‑ 123 ‑', etc.
    Adjust the regex if your footer looks different.
    """
    # look for a line that is only 1‑6 digits (optionally bounded by dashes or whitespace)
    page_num_re: re.Pattern = re.compile(
        r"^\s*[-–—]?\s*\d{1,6}\s*[-–—]?\s*$", re.MULTILINE)
    return page_num_re.sub("", text).strip()


def clean_text(raw: str) -> str:
    """
    Additional scrubbing: normalize Unicode quotes, collapse 3+ newlines, trim.
    """
    cleaned: str = strip_page_numbers(raw)
    cleaned: str = re.sub(r"\n{3,}", "\n\n", cleaned)  # keep max 1 empty line
    return cleaned


def chapters_from_toc(doc: fitz.Document) -> Optional[Generator[tuple[str, int, int], None, None]]:
    """
    Yield tuples: (chapter_title, start_page, end_page).
    If the PDF has *no* bookmarks, fall back to None so we can do naïve splitting later.
    """
    toc: list = doc.get_toc(simple=True)  # list of [level, title, page_number]
    if not toc:
        return None

    # keep only top‑level nodes (level==1) → “real” chapters
    top = [(title, page-1) for level, title, page in toc if level == 1]
    for idx, (title, start) in enumerate(top):
        end = doc.page_count - 1 if idx == len(top)-1 else top[idx+1][1] - 1
        yield title, start, end


def sanitize_filename(name: str) -> str:
    """
    Remove or replace characters that are invalid in filenames.
    """
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def get_clean_page_text(page, header_margin=70, footer_margin=70) -> str:
    """
    Extract text blocks, skipping top and bottom areas (header/footer).
    """
    blocks = page.get_text("blocks")
    page_height = page.rect.height

    body_blocks = [
        b[4].strip() for b in blocks
        if b[1] > header_margin and b[3] < (page_height - footer_margin)
    ]
    return "\n".join(body_blocks)


def unwrap_paragraphs(text: str) -> str:
    """
    Merges lines inside paragraphs while preserving paragraph breaks.
    A paragraph is assumed to be separated by double newlines.
    """
    paragraphs = text.split("\n\n")
    clean_paragraphs = []

    for para in paragraphs:
        # Remove stray newlines inside a paragraph and rewrap it as one line
        line = ' '.join(p.strip() for p in para.splitlines())
        clean_paragraphs.append(line)

    return '\n\n'.join(clean_paragraphs)



def main():
    TXT_PATH.mkdir(exist_ok=True, parents=True)
    for pdf in os.listdir(PDF_PATH):
        if pdf.endswith(".pdf"):
            pdf_path: str = os.path.join(PDF_PATH, pdf)
            base_name: str = os.path.splitext(pdf)[0]

            with fitz.open(pdf_path) as doc:
                toc_chaps = list(chapters_from_toc(doc) or [])
                if toc_chaps:
                    # split using TOC
                    for idx, (title, p0, p1) in enumerate(toc_chaps, 1):
                        print(f"Processing {title} (pages {p0+1}-{p1+1})")
                        raw = ""
                        for page in doc.pages(p0, p1+1):  # inclusive
                            raw += get_clean_page_text(page) + "\n"
                        # 02d=format as decimal integer, pad with 0 to be at least 2 digits wide
                        chapter_name: str = f"{base_name}_chapter_{idx:02d}.txt"
                        chapter_path: pathlib.Path = TXT_PATH / \
                            sanitize_filename(chapter_name)
                        with chapter_path.open("w", encoding="utf-8") as out:
                            out.write(f"### {title} ###\n\n")
                            cleaned = clean_text(raw)
                            rewrapped = unwrap_paragraphs(cleaned)
                            out.write(rewrapped)

                else:
                    # --- write the whole PDF to a single file ---
                    print("No TOC detected; extracting entire PDF to a single file.")
                    full = "".join(get_clean_page_text(
                        page) + "\n" for page in doc)
                    output_name: str = f"{base_name}_full.txt"
                    output_path: pathlib.Path = TXT_PATH / \
                        sanitize_filename(output_name)
                    with output_path.open("w", encoding="utf-8") as out:
                        cleaned = clean_text(full)
                        rewrapped = unwrap_paragraphs(cleaned)
                        out.write(rewrapped)



if __name__ == "__main__":
    main()
