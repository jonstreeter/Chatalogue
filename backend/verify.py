import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.services.ingestion import IngestionService
    print("IngestionService imported successfully.")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Error: {e}")
