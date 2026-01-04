"""
US-GAAP Fields Configuration Loader
Loads us-gaap fields from data/config/us-gaap-fields.txt
"""
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional


class UploadConfig:

    def __init__(self, config_path: str="src/storage/config.yaml"):
        """
        Initialize the config loader.

        Args:
            config_path: Path to config.yaml
        """
        self.config_path = Path(config_path)
        self._config = None
        self._us_gaap_fields = None
        self._dei_fields = None

    def load(self):
        """Load configuration from config.yaml"""
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Load DEI fields
        self._dei_fields = self._config['fields']['dei']

        # Load US-GAAP fields from data/config/us-gaap-fields.txt
        us_gaap_config = self._config['fields']['us-gaap']
        source_file = us_gaap_config['source']

        fields_path = Path(source_file)
        if not fields_path.exists():
            raise FileNotFoundError(
                f"US-GAAP fields file not found: {source_file}\n"
                f"Run scripts/gaap_fields.py to generate it."
            )

        self._us_gaap_fields = self._load_us_gaap_fields(fields_path)

    def _load_us_gaap_fields(self, file_path: Path) -> List[str]:
        """
        Load US-GAAP fields from data/config/us-gaap-fields.txt

        :param file_path: Path to us-gaap-fields.txt
        :return: List of 2069 US-GAAP field names
        """
        with open(file_path, 'r') as f:
            fields = [line.strip() for line in f if line.strip()]

        return fields

    @property
    def us_gaap_fields(self) -> List[str]:
        """Get list of US-GAAP fields (2069 fields)"""
        if self._us_gaap_fields is None:
            self.load()
        return self._us_gaap_fields  # type: ignore[return-value]

    @property
    def dei_fields(self) -> List[str]:
        """Get list of DEI fields"""
        if self._dei_fields is None:
            self.load()
        return self._dei_fields  # type: ignore[return-value]
    
    @property
    def transfer(self) -> Optional[Dict[str, Any]]:
        """Get transfer config"""
        if self._config is None:
            self.load()
        return self._config.get('transfer') if self._config else None

    @property
    def client(self) -> Optional[Dict[str, Any]]:
        """Get client config"""
        if self._config is None:
            self.load()
        return self._config.get('client') if self._config else None

# Example usage
if __name__ == "__main__":
    config = UploadConfig()

    print("=" * 60)
    print("US-GAAP Fields Configuration")
    print("=" * 60)

    print(f"\nDEI fields: {len(config.dei_fields)}")
    for field in config.dei_fields:
        print(f"  - {field}")

    print(f"\nUS-GAAP fields: {len(config.us_gaap_fields)}")
    print(f"  First 10:")
    for field in config.us_gaap_fields[:10]:
        print(f"    - {field}")
    print(f"  ...")
    print(f"  Last 5:")
    for field in config.us_gaap_fields[-5:]:
        print(f"    - {field}")

    print(f"\nTotal: {len(config.dei_fields) + len(config.us_gaap_fields)} fields")
