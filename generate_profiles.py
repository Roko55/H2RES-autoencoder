"""Generate annual profiles with the delivered autoencoder."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    source = Path(__file__).resolve().parent / 'model'
    sys.path.insert(0, str(source))
    runpy.run_path(str(source / 'generate_profiles.py'), run_name='__main__')
