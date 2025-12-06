"""
Script to identify corrupted PDF files in the peraturan_pdfs folder
"""
import os
import warnings
from pypdf import PdfReader

# Suppress PyPDF warnings about PDF structure issues
warnings.filterwarnings('ignore')

def check_pdf_files(folder_path):
    """Check all PDF files and identify corrupted ones"""
    corrupted_files = []
    valid_files = []
    
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    total = len(pdf_files)
    
    print(f"Checking {total} PDF files...\n")
    
    for idx, filename in enumerate(pdf_files, 1):
        filepath = os.path.join(folder_path, filename)
        try:
            # Try to open and read the PDF
            reader = PdfReader(filepath)
            num_pages = len(reader.pages)
            # Try to extract text from first page to verify it's readable
            if num_pages > 0:
                _ = reader.pages[0].extract_text()
            valid_files.append(filename)
            if idx % 100 == 0:
                print(f"Checked {idx}/{total} files...")
        except Exception as e:
            error_msg = str(e)
            corrupted_files.append((filename, error_msg))
            print(f"❌ CORRUPTED: {filename}")
            print(f"   Error: {error_msg[:100]}")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Valid PDFs: {len(valid_files)}")
    print(f"  Corrupted PDFs: {len(corrupted_files)}")
    print(f"{'='*60}\n")
    
    if corrupted_files:
        print("Corrupted files found:")
        for filename, error in corrupted_files:
            print(f"  - {filename}")
            print(f"    Error: {error[:100]}")
        return [f[0] for f in corrupted_files]
    else:
        print("No corrupted files found!")
        return []

if __name__ == "__main__":
    folder = "./peraturan_pdfs"
    corrupted = check_pdf_files(folder)
    
    if corrupted:
        print(f"\nTo delete corrupted files, run:")
        for filename in corrupted:
            filepath = os.path.join(folder, filename)
            print(f'  del "{filepath}"')
