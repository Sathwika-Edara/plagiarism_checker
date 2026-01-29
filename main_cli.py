import argparse
import os
import sys


try:
    from src.utils import read_txt_file, read_docx_file, read_pdf_file
    from src.vectorizer import vectorize_corpus
    from src.similarity import  calculate_similarity
except ImportError as e:
    print("Import error:", e, file=sys.stderr)
    sys.exit(1)
def read_file_content(file_path:str)->str|None:
    try:
        _, extension = os.path.splitext(file_path.lower())

        if extension == ".txt":
            with open(file_path, "rb") as f:
                return read_txt_file(f.read())

        elif extension == ".docx":
            return read_docx_file(file_path)

        elif extension == ".pdf":
            with open(file_path, "rb") as f:
                return read_pdf_file(f)
        else:
            print(f"Error: Unsupported file type '{extension}'. Please use .txt, .docx, or .pdf.", file=sys.stderr)
            return None
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {e}", file=sys.stderr)
        return None

def main():
    """The main function to run the CLI application."""
    parser = argparse.ArgumentParser(description="Compare two text files to check for plagiarism.")
    parser.add_argument(
        "file1",
        help="The path to the first file for comparison."
    )
    parser.add_argument(
        "file2",
        help="The path to the second file for comparison."
    )
    parser.add_argument(
        "-d", "--directory",
        help="The path to a directory. If provided, 'file1' will be compared against all files in this directory instead of 'file2'."
    )
    args = parser.parse_args()
    text1 = read_file_content(args.file1)
    text2 = read_file_content(args.file2)
    if text1 is None or text2 is None:
        print("\nCould not proceed due to file reading errors. Please check the paths and file formats.")
        # sys.exit(1) indicates that the program terminated with an error.
        sys.exit(1)

    print("Successfully read both files.")

    print("Vectorizing documents...")
    try:
        corpus=[text1,text2]
        vectors=vectorize_corpus(corpus)
        print("Successfully vectorized documents.")
    except Exception as e:
        print(f"An error occurred during vectorization: {e}", file=sys.stderr)
        sys.exit(1)
    print("Calculating similarity...")
    try:

        similarity = calculate_similarity(vectors[0:1], vectors[1:2])
        print("Similarity calculation complete.")
    except Exception as e:
        print(f"An error occurred during similarity calculation: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 40)
    print("      Plagiarism Check Result")
    print("=" * 40)

    file1_name = os.path.basename(args.file1)
    file2_name = os.path.basename(args.file2)


    print(f"The similarity between '{file1_name}' and '{file2_name}' is: {similarity}%")
    print("=" * 40)


if __name__ == "__main__":
    main()