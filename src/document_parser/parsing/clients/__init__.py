"""Thin wrappers around external network APIs used by the pdf loader's
heavy path (VLM, AzureDI -- AzureDI is currently unused by the pipeline, see
``azure_document_intelligence.py``'s module docstring).

Kept separate from ``parsing/loaders/pdf/vlm.py`` / ``.../azure_di.py`` on
purpose: those modules own the "map a result into DocumentElements" logic,
while the clients here own "how do I actually call the API". Swapping
providers later (different VLM) means touching only a client module, not the
DocumentElement-mapping code.
"""
