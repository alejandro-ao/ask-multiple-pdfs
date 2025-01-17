import React, { useState, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { Streamlit, withStreamlitConnection } from 'streamlit-component-lib';

// Set up PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

const PdfViewer = ({ args }) => {
  const { pdf_data, highlights } = args;
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [scale, setScale] = useState(1.0);
  const [highlightLayer, setHighlightLayer] = useState([]);

  // Handle page changes from Streamlit
  useEffect(() => {
    if (args.targetPage && args.targetPage !== pageNumber) {
      const newPage = parseInt(args.targetPage);
      if (!isNaN(newPage) && newPage >= 1 && newPage <= numPages) {
        setPageNumber(newPage);
      }
    }
  }, [args.targetPage, numPages]);

  useEffect(() => {
    if (highlights) {
      setHighlightLayer(highlights);
    }
  }, [highlights]);

  function onDocumentLoadSuccess({ numPages }) {
    setNumPages(numPages);
    setPageNumber(1);
    Streamlit.setFrameHeight(800);
  }

  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= numPages) {
      setPageNumber(newPage);
      Streamlit.setComponentValue({ type: 'pageChanged', page: newPage });
    }
  };

  const renderHighlights = (pageIndex) => {
    const pageHighlights = highlightLayer.filter(h => h.page === pageIndex);
    return pageHighlights.map((highlight, index) => (
      <div
        key={index}
        style={{
          position: 'absolute',
          left: `${highlight.bbox.x}%`,
          top: `${highlight.bbox.y}%`,
          width: `${highlight.bbox.width}%`,
          height: `${highlight.bbox.height}%`,
          backgroundColor: 'rgba(255, 255, 0, 0.3)',
          cursor: 'pointer',
        }}
        onClick={() => Streamlit.setComponentValue(highlight)}
      />
    ));
  };

  return (
    <div style={{ position: 'relative' }}>
      <Document
        file={`data:application/pdf;base64,${pdf_data}`}
        onLoadSuccess={onDocumentLoadSuccess}
      >
        <Page
          pageNumber={pageNumber}
          scale={scale}
          renderTextLayer={true}
          renderAnnotationLayer={true}
        />
        {renderHighlights(pageNumber)}
      </Document>
      <div style={{ marginTop: '10px', display: 'flex', gap: '10px', alignItems: 'center', justifyContent: 'center' }}>
        <button
          onClick={() => handlePageChange(pageNumber - 1)}
          disabled={pageNumber <= 1}
        >
          Previous
        </button>
        <span>
          Page {pageNumber} of {numPages}
        </span>
        <button
          onClick={() => handlePageChange(pageNumber + 1)}
          disabled={pageNumber >= numPages}
        >
          Next
        </button>
        <button onClick={() => setScale(scale + 0.1)}>Zoom In</button>
        <button onClick={() => setScale(Math.max(0.1, scale - 0.1))}>
          Zoom Out
        </button>
      </div>
    </div>
  );
};

export default withStreamlitConnection(PdfViewer);
