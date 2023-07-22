
# https://developer.adobe.com/document-services/docs/overview/pdf-extract-api/quickstarts/python/
from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.exception.exceptions import (
    ServiceApiException,
    ServiceUsageException,
    SdkException,
)
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.io.file_ref import FileRef
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import (
    ExtractPDFOptions,
)
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import (
    ExtractElementType,
)
import os.path
import zipfile
import json
import logging
from dotenv import load_dotenv
import re
from PyPDF2 import PdfReader

class PDFHandler:

    def __init__(self, inputPDFName, zipFileName) -> None:
        load_dotenv()
        self.input_pdf = inputPDFName
        self.zip_file = zipFileName
        self.startingSection = "intro"

        self.structurePDF()

    def structurePDF(self):
        if os.path.isfile(self.zip_file):
            os.remove(self.zip_file)

        try:
            credentials = (
                Credentials.service_principal_credentials_builder()
                .with_client_id(os.getenv("PDF_SERVICES_CLIENT_ID"))
                .with_client_secret(os.getenv("PDF_SERVICES_CLIENT_SECRET"))
                .build()
            )
            execution_context = ExecutionContext.create(credentials)
            extract_pdf_operation = ExtractPDFOperation.create_new()
            source = FileRef.create_from_local_file(self.input_pdf)
            extract_pdf_operation.set_input(source)
            extract_pdf_options: ExtractPDFOptions = (
                ExtractPDFOptions.builder()
                .with_element_to_extract(ExtractElementType.TEXT)
                .build()
            )
            extract_pdf_operation.set_options(extract_pdf_options)
            result: FileRef = extract_pdf_operation.execute(execution_context)
            result.save_as(self.zip_file)

        except (ServiceApiException, ServiceUsageException, SdkException):
            logging.exception("Exception encountered while executing operation")

    def getStructuredData(self):
        archive = zipfile.ZipFile(self.zip_file, "r")
        jsonentry = archive.open("structuredData.json")
        jsondata = jsonentry.read()
        structuredData = json.loads(jsondata)
        return structuredData

    def getSections(self):
        structuredData = self.getStructuredData()
        H1_sections = {}
        sections = {}
        startIndex = "None"
        endIndex = "None"

        for index, element in enumerate(structuredData["elements"]):
            if "/H1" in element["Path"]:
                sectionName = re.sub(r"\d+", "", element["Text"].lower().strip())
                H1_sections[index] = element["Text"]

                if self.startingSection in sectionName:
                    startIndex = index
                elif "acknowledg" in sectionName or "reference" in sectionName:
                    endIndex = index
                    break

        for H1_sectionIndex in H1_sections:
            if H1_sectionIndex >= startIndex and H1_sectionIndex <= endIndex:
                sections[H1_sectionIndex] = H1_sections[H1_sectionIndex]
        return sections

    def getFilteredText(self):
        sections = self.getSections()
        structuredData = self.getStructuredData()
        sectionIndex = []
        sectionText = {}

        for index in sections:
            sectionIndex.append(index)

        for x in range(len(sectionIndex) - 1):
            text = ""

            for raw_index, element in enumerate(structuredData["elements"]):
                if "P" in element["Path"]:
                    if raw_index >= sectionIndex[x] and raw_index < sectionIndex[x + 1]:
                        text += element["Text"]

            sectionName = sections[sectionIndex[x]]
            sectionText[sectionName] = text
        return sectionText
    
    def get_unfiltered_pdf_text(pdf):
        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text


