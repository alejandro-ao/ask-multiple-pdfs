import React from 'react';
import ReactDOM from 'react-dom';
import PdfViewer from './PdfViewer';
import { StreamlitProvider } from 'streamlit-component-lib';

ReactDOM.render(
  <React.StrictMode>
    <StreamlitProvider>
      <PdfViewer />
    </StreamlitProvider>
  </React.StrictMode>,
  document.getElementById('root')
);
