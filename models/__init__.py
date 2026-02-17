from .models import register, make

# SR architecture
from . import cgnet_deformable2d_arch

# OCR baselines
from . import GP_LPR_arch
try:
    from . import ocr_rodosol
except ModuleNotFoundError:
    # Optional dependency (tensorflow). Keep other models importable.
    ocr_rodosol = None
