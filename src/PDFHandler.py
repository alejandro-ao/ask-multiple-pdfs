from PyPDF2 import PdfReader

class PDFHandler:
    #def __init__():
     
    @staticmethod
    def get_pdf_text(pdf):
        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

