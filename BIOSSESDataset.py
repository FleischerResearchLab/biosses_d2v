import os
from pandas import DataFrame
from numpy import array
from docx import Document
import subprocess

class BIOSSESDataset:
    """
    Dataset class for the BIOSSES Benchmark.
    Source: https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html.
    """
    
    def __init__(self, doc_path=None):
        
        def _create_dataframe_from_doc(fpath):
            self.doc = Document(fpath)
            table = self.doc.tables[0]
            data = array([[cell.text.strip() for cell in row.cells] for row in table.rows])
            self.df = DataFrame(data[1:,1:], columns=data[0, 1:])
        
        if (doc_path != None):
            _create_dataframe_from_doc(doc_path)
            
        else:  
            BIOSSES_PATH = os.path.abspath("BIOSSES-Dataset/Annotation-Pairs.docx")
            
            if not os.path.isfile(BIOSSES_PATH):
                subprocess.run(["wget",
                                "https://tabilab.cmpe.boun.edu.tr/BIOSSES/Downloads/BIOSSES-Dataset.rar"] 
                              )
                subprocess.run(["unrar", "x", "BIOSSES-Dataset.rar"])
                
            _create_dataframe_from_doc(BIOSSES_PATH)

 
    def get_dataframe(self):
        return self.df