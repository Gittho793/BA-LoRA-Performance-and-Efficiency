import os
import time
import math
from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig, EvolutionConfig
from typing import List

PDF_FILES = "/cluster/user/thoadelt/LoraData/raws/data/output"
MODEL = "gpt-4.1-mini"


def process_documents_in_batches(
    pdf_directory: str,
    batch_size: int = 3,
    delay_between_batches: int = 10,
    max_goldens_per_context: int = 5,
    max_contexts_per_document: int = 5,
    max_retries: int = 3
):
    """
    Process PDF documents in batches to avoid rate limits

    Args:
        pdf_directory: Path to directory containing PDF files
        batch_size: Number of documents to process per batch
        delay_between_batches: Seconds to wait between batches
        max_goldens_per_context: Maximum golden examples per context
        max_contexts_per_document: Maximum contexts per document
        max_retries: Maximum retry attempts for API calls
    """

    # Get all PDF files
    pdf_files = [
        os.path.join(pdf_directory, f)
        for f in os.listdir(pdf_directory)
        if f.lower().endswith('.txt')
    ]

    if not pdf_files:
        print("No PDF files found in the specified directory.")
        return

    print(f"Found {len(pdf_files)} PDF files to process")

    # Initialize synthesizer
    synthesizer = Synthesizer(
        evolution_config=EvolutionConfig(num_evolutions=0),
        model=MODEL,
        async_mode=True,
        max_concurrent=3  # Keep this low to avoid rate limits
    )

    # Configure context construction
    context_config = ContextConstructionConfig(
        max_contexts_per_document=max_contexts_per_document,
        embedder="text-embedding-3-small",
        critic_model=MODEL,
        max_retries=max_retries
    )

    # Calculate batch information
    total_batches = math.ceil(len(pdf_files) / batch_size)
    processed_files = 0
    failed_files = []

    print(
        f"Processing {len(pdf_files)} files in {total_batches} batches of {batch_size}")
    print("-" * 60)

    # Process files in batches
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(pdf_files))
        batch_files = pdf_files[start_idx:end_idx]

        print(f"\nBatch {batch_num + 1}/{total_batches}")
        print(
            f"Processing files {start_idx + 1}-{end_idx} of {len(pdf_files)}")

        # Show which files are being processed
        for i, file_path in enumerate(batch_files, 1):
            filename = os.path.basename(file_path)
            print(f"  {i}. {filename}")

        # Process the batch with retry logic
        retry_count = 0
        max_batch_retries = 3
        batch_success = False

        while retry_count < max_batch_retries and not batch_success:
            try:
                print(
                    f"\nStarting batch processing (attempt {retry_count + 1})...")
                start_time = time.time()

                # Process the batch
                result = synthesizer.generate_goldens_from_docs(
                    document_paths=batch_files,
                    max_goldens_per_context=max_goldens_per_context,
                    context_construction_config=context_config,
                )

                end_time = time.time()
                processing_time = end_time - start_time

                print(
                    f"✅ Batch {batch_num + 1} completed successfully in {processing_time:.1f} seconds")
                processed_files += len(batch_files)
                batch_success = True

            except Exception as e:
                retry_count += 1
                error_type = type(e).__name__
                print(
                    f"❌ Batch {batch_num + 1} failed (attempt {retry_count}): {error_type}")
                print(f"   Error details: {str(e)}")

                if retry_count < max_batch_retries:
                    wait_time = 30 * retry_count  # Exponential backoff
                    print(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(
                        f"   Batch {batch_num + 1} failed after {max_batch_retries} attempts")
                    failed_files.extend(batch_files)

        # Wait between batches (except for the last batch)
        if batch_num < total_batches - 1 and batch_success:
            print(
                f"\nWaiting {delay_between_batches} seconds before next batch...")
            time.sleep(delay_between_batches)

    # Print final summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files found: {len(pdf_files)}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed files: {len(failed_files)}")

    if failed_files:
        print("\nFailed files:")
        for failed_file in failed_files:
            print(f"  - {os.path.basename(failed_file)}")

    print(f"\nProcessing complete!")


def main():
    """Main function to run the batch processing"""
    try:
        process_documents_in_batches(
            pdf_directory=PDF_FILES,
            batch_size=1,              # Process 3 documents at a time
            delay_between_batches=120,   # Wait between batches
            max_goldens_per_context=5,
            max_contexts_per_document=3,
            max_retries=3
        )
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
