"""
Generate expected answers for the questions posed via OpenAI API
"""
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# Third-party import
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI library not found. Install with: pip install openai")
    sys.exit(1)

MAX_RETRIES = 5
INITIAL_DELAY = 1
MAX_DELAY = 60
CHUNK_SIZE = 3000  # Characters per chunk for long documents
MIN_TEXT_LENGTH = 100  # Minimum text length to process


@dataclass
class QuestionAnswer:
    """
    Represents a question, its corresponding answer, and additional data.
    """
    question: str
    answer: str
    difficulty: str
    topic: str
    source_file: str


class TextProcessor:
    """
    Processes text files for question-answer pairs.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_text_from_file(self, txt_path: Path) -> Optional[str]:
        """Extract text from a .txt file"""
        try:
            self.logger.info(f"Reading text from: {txt_path}")
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
            if len(text.strip()) < MIN_TEXT_LENGTH:
                self.logger.warning(
                    f"Text too short ({len(text)} chars): {txt_path}")
                return None
            return text.strip()
        except Exception as e:
            self.logger.error(f"Failed to read {txt_path}: {str(e)}")
            return None

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
        """Chunk text into manageable pieces for the API."""
        # Identical to PDF version
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 2 <= chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks


class OpenAIQuestionGenerator:
    """
    Generates question-answer pairs using OpenAI GPT-4.1-mini.
    """
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.logger = logging.getLogger(__name__)

    def _exponential_backoff(self, attempt: int) -> float:
        return min(INITIAL_DELAY * (2 ** attempt), MAX_DELAY)

    def _make_api_call_with_retry(self, messages: List[Dict], response_format: Dict) -> Optional[Dict]:
        for attempt in range(MAX_RETRIES):
            try:
                self.logger.debug(
                    f"Making OpenAI API call (attempt {attempt + 1}/{MAX_RETRIES})")
                response = self.client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=messages,
                    response_format=response_format,
                    temperature=0.7,
                    max_tokens=2000
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "429" in error_msg:
                    delay = self._exponential_backoff(attempt)
                    self.logger.warning(
                        f"Rate limit hit, waiting {delay} seconds before retry {attempt + 1}")
                    time.sleep(delay)
                    continue
                elif "invalid_request_error" in error_msg:
                    self.logger.error(f"Invalid request error: {str(e)}")
                    return None
                elif attempt == MAX_RETRIES - 1:
                    self.logger.error(f"All retry attempts failed: {str(e)}")
                    return None
                else:
                    delay = self._exponential_backoff(attempt)
                    self.logger.warning(
                        f"API error, retrying in {delay} seconds: {str(e)}")
                    time.sleep(delay)
        return None

    def generate_questions(self, text: str, source_file: str) -> List[QuestionAnswer]:
        """
        Generate the questions for evaluation from the provided text.
        """
        self.logger.info(f"Generating questions for text from {source_file}")
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "question_answer_pairs",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "answer": {"type": "string"},
                                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                                    "topic": {"type": "string"}
                                },
                                "required": ["question", "answer", "difficulty", "topic"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["questions"],
                    "additionalProperties": False
                }
            }
        }
        system_prompt = """You are an expert educator creating comprehensive test questions to evaluate whether an LLM truly learned and knows the content of a document.
Your task is to generate exactly 5 high-quality question-answer pairs that test:
1. Factual recall and comprehension
2. Conceptual understanding
3. Application of knowledge
4. Critical analysis
5. Synthesis of information
Guidelines:
- Questions should be specific and unambiguous
- Answers should be complete and accurate
- Include a mix of difficulty levels (easy, medium, hard)
- Ensure questions can be answered based solely on the provided text
- Each question should test a different aspect or topic from the text"""
        user_prompt = f"""Based on the following text, generate exactly 5 comprehensive questions and their corresponding answers that would effectively test whether an LLM truly learned this content. There should not be a reference to the text just factual questions. Everything needs to be in german. ALL QUESTIONS AND ANSWERS MUST BE IN GERMAN.

Text to analyze:

{text}

Generate questions that span different difficulty levels and topics covered in the text in german."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response_data = self._make_api_call_with_retry(
            messages, response_format)
        if not response_data or "questions" not in response_data:
            self.logger.error(
                f"Failed to generate questions for {source_file}")
            return []
        qas = []
        for qa_data in response_data["questions"]:
            try:
                qas.append(QuestionAnswer(
                    question=qa_data["question"],
                    answer=qa_data["answer"],
                    difficulty=qa_data["difficulty"],
                    topic=qa_data["topic"],
                    source_file=source_file
                ))
            except KeyError as e:
                self.logger.error(
                    f"Missing required field in API response: {str(e)}")
                continue
        return qas


class TextQuestionProcessor:
    """
    Main processor to handle directories of text files, generate qa pairs and save json.
    """
    def __init__(self, api_key: str):
        self.text_processor = TextProcessor()
        self.question_generator = OpenAIQuestionGenerator(api_key)
        self.logger = logging.getLogger(__name__)

    def process_directory(self, directory_path: Path, output_file: Path) -> bool:
        """
        Process all .txt files in the given directory and generate question-answer pairs.
        Saves results to output_file in JSON format."""
        self.logger.info(f"Processing text directory: {directory_path}")
        txt_files = list(directory_path.glob("*.txt"))
        if not txt_files:
            self.logger.error(f"No text files found in {directory_path}")
            return False
        all_questions = []
        processed_count = 0
        failed_count = 0
        for txt_file in txt_files:
            self.logger.info(
                f"Processing file {processed_count + 1}/{len(txt_files)}: {txt_file.name}")
            try:
                text = self.text_processor.extract_text_from_file(txt_file)
                if not text:
                    self.logger.warning(
                        f"Skipping {txt_file.name}: No text extracted")
                    failed_count += 1
                    continue
                text_chunks = self.text_processor.chunk_text(text)
                questions = self.question_generator.generate_questions(
                    text_chunks[0], txt_file.name
                )
                if questions:
                    all_questions.extend(questions)
                    processed_count += 1
                    self.logger.info(
                        f"Successfully processed {txt_file.name}: {len(questions)} questions generated")
                else:
                    self.logger.warning(
                        f"No questions generated for {txt_file.name}")
                    failed_count += 1
                time.sleep(1)
            except Exception as e:
                self.logger.error(
                    f"Error processing {txt_file.name}: {str(e)}")
                failed_count += 1
                continue
        if all_questions:
            try:
                self._save_to_json(all_questions, output_file)
                self.logger.info(
                    f"Processing complete. Processed: {processed_count}, Failed: {failed_count}")
                self.logger.info(
                    f"Total questions generated: {len(all_questions)}")
                self.logger.info(f"Results saved to: {output_file}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to save results: {str(e)}")
                return False
        else:
            self.logger.error(
                "No questions were generated from any text files")
            return False

    def _save_to_json(self, questions: List[QuestionAnswer], output_file: Path) -> None:
        output_data = {
            "metadata": {
                "total_questions": len(questions),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_used": "gpt-4.1-mini"
            },
            "questions": [
                {
                    "question": qa.question,
                    "answer": qa.answer,
                    "difficulty": qa.difficulty,
                    "topic": qa.topic,
                    "source_file": qa.source_file
                }
                for qa in questions
            ]
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


def setup_logging(log_level: str = "INFO") -> None:
    """
    Setup logging configuration
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                "text_question_generator.log", encoding='utf-8')
        ]
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main():
    """
    Main entry point for the script
    """
    parser = argparse.ArgumentParser(
        description="Generate test questions from text files using OpenAI GPT-4.1-mini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
%(prog)s --directory ./texts --output questions.json
%(prog)s -d /path/to/texts -o results.json --log-level DEBUG

Environment Variables:
OPENAI_API_KEY: Your OpenAI API key (required)
"""
    )
    parser.add_argument(
        "-d", "--directory",
        type=Path,
        required=True,
        help="Directory containing text files to process"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default="generated_questions.json",
        help="Output JSON file path (default: generated_questions.json)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --api-key option")
        sys.exit(1)

    if not args.directory.exists():
        logger.error(f"Directory does not exist: {args.directory}")
        sys.exit(1)
    if not args.directory.is_dir():
        logger.error(f"Path is not a directory: {args.directory}")
        sys.exit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Starting text question generation process")
    processor = TextQuestionProcessor(api_key)
    success = processor.process_directory(args.directory, args.output)
    if success:
        logger.info("Processing completed successfully!")
        sys.exit(0)
    else:
        logger.error("Processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
