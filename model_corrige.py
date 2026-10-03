"""Challenge entry point. The active experiment is selected by configuration."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
sys.dont_write_bytecode=True


def main():
    release=json.loads((ROOT/'config/active_release.json').read_text())
    entry=(ROOT/release['entry_point']).resolve()
    if not entry.is_relative_to(ROOT/'experiments'):
        raise ValueError('A release entry point must be inside experiments/.')
    spec=importlib.util.spec_from_file_location('active_experiment',entry)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.main()


if __name__=='__main__':main()
