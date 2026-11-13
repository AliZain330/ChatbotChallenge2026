"""Exercise a real Chroma database without a network embedding model."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class StoreIntegrationTests(unittest.TestCase):
    def test_vectors_filters_and_safe_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            script = """
import sys
from pathlib import Path
from bot.store import get_store
path=sys.argv[1]
marker=Path(path)/'keep.txt'
collection=get_store(path=path, name='target')
marker.write_text('preserve')
other=get_store(path=path, name='other')
other.upsert(ids=['other'], documents=['other source'], metadatas=[{'kind':'text'}], embeddings=[[0.,1.]])
collection.upsert(ids=['laser','poster'],documents=['PVC is banned','Workshop poster'],metadatas=[{'kind':'text'},{'kind':'image'}],embeddings=[[1.,0.],[0.,1.]])
assert collection.query(query_embeddings=[[1.,0.]],n_results=1)['ids']==[['laser']]
assert collection.get(where={'kind':'image'})['ids']==['poster']
assert get_store(path=path,name='target',reset=True).count()==0
assert other.count()==1
assert marker.read_text()=='preserve'
print('real Chroma vectors, filters and collection-only reset passed')
"""
            run = subprocess.run([sys.executable, '-c', script, str(Path(directory)/'store')],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('passed', run.stdout)


if __name__ == '__main__':
    unittest.main()
