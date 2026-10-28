"""Short IPC addresses whose files remain physically inside the release tree."""

import hashlib
from pathlib import Path


def temp_address(root: Path) -> Path:
    target = root.resolve() / "tmp"
    target.mkdir(exist_ok=True)
    alias = Path("/tmp") / ("tv-" + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:6])
    if alias.is_symlink():
        if alias.resolve() != target:
            raise RuntimeError(f"Temporary address belongs to a different workspace: {alias}")
    elif alias.exists():
        raise RuntimeError(f"Temporary address is occupied: {alias}")
    else:
        alias.symlink_to(target, target_is_directory=True)
    return alias
