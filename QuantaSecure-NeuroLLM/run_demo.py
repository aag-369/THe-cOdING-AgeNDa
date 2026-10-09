"""Single entry point:  python run_demo.py [--fast] [--retrain] [--aead ChaCha20-Poly1305] [--quiet]"""

import sys

from quantasecure_neurollm.demo import main

if __name__ == "__main__":
    sys.exit(main())
