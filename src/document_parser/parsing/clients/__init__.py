"""Thin wrappers around external network APIs used by the pdf loader's
heavy path (AzureDI, VLM).

Kept separate from ``parsing/loaders/pdf/azure_di.py`` and ``.../vlm.py`` on
purpose: those two modules own the "map a result into DocumentElements"
logic, while the clients here own "how do I actually call the API". Swapping
providers later (different cloud, different VLM) means touching only a
client module, not the DocumentElement-mapping code.
"""
