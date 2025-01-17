import os
import streamlit.components.v1 as components

_RELEASE = False

if not _RELEASE:
    _component_func = components.declare_component(
        "pdf_viewer",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("pdf_viewer", path=build_dir)

def pdf_viewer(pdf_data, highlights=None, key=None):
    """Create a component instance.
    
    Args:
        pdf_data: Base64 encoded PDF data
        highlights: List of highlights with page numbers and positions
        key: Unique key for the component instance
    
    Returns:
        Selected highlight information if any
    """
    component_value = _component_func(
        pdf_data=pdf_data,
        highlights=highlights,
        key=key,
        default=None
    )
    return component_value
