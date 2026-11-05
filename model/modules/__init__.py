try:
    from .model import LeJEPA2D
    from .backbone import GrayscaleViTBackbone
except (ImportError, RuntimeError):
    LeJEPA2D = None
    GrayscaleViTBackbone = None
