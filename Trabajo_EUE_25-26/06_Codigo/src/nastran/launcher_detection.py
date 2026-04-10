from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import shutil


CANDIDATE_COMMANDS = [
    "nastran",
    "nastranw",
    "nast20252",
    "nast20252w",
    "msc20252",
    "msc20252w",
]

WINDOWS_ROOTS = [
    Path(r"C:\Program Files\MSC.Software"),
    Path(r"C:\Program Files (x86)\MSC.Software"),
    Path(r"C:\Program Files\Hexagon"),
    Path(r"C:\Program Files (x86)\Hexagon"),
]

EXECUTABLE_PATTERN = re.compile(r"^(nastran|nastranw|nast\d+w?|msc\d+w?)\.exe$", re.IGNORECASE)


@dataclass(slots=True)
class LauncherSearchResult:
    executable: Path | None
    checked_locations: list[str]
    candidates: list[str]
    messages: list[str]
    selected_reason: str | None

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["executable"] = str(self.executable) if self.executable else None
        return payload


def _score_candidate(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if name == "nastran.exe":
        return (0, name)
    if name.startswith("nast"):
        return (1, name)
    if name.startswith("msc"):
        return (2, name)
    return (9, name)


def _valid_candidate(path: Path) -> bool:
    return path.exists() and path.is_file() and bool(EXECUTABLE_PATTERN.match(path.name))


def detect_nastran_executable(configured_path: str = "") -> LauncherSearchResult:
    checked: list[str] = []
    candidates: list[Path] = []
    messages: list[str] = []

    configured_path = configured_path.strip()
    if configured_path:
        explicit = Path(configured_path)
        checked.append(f"YAML: {explicit}")
        if _valid_candidate(explicit):
            messages.append(f"Ejecutable valido encontrado en YAML: {explicit}")
            return LauncherSearchResult(
                executable=explicit.resolve(),
                checked_locations=checked,
                candidates=[str(explicit.resolve())],
                messages=messages,
                selected_reason="configured_path",
            )
        messages.append(f"La ruta definida en YAML no es valida o no existe: {explicit}")

    for command_name in CANDIDATE_COMMANDS:
        resolved = shutil.which(command_name)
        checked.append(f"PATH: {command_name}")
        if resolved:
            candidate = Path(resolved)
            candidates.append(candidate)
            messages.append(f"Candidato encontrado en PATH: {candidate}")

    for root in WINDOWS_ROOTS:
        checked.append(f"Windows root: {root}")
        if not root.exists():
            messages.append(f"Raiz no disponible: {root}")
            continue

        for bin_dir in root.rglob("bin"):
            if bin_dir.parent.name.lower() != "nastran":
                continue
            checked.append(f"Scanned bin: {bin_dir}")
            for file in bin_dir.iterdir():
                if _valid_candidate(file):
                    candidates.append(file.resolve())

    unique_candidates = sorted({candidate.resolve() for candidate in candidates}, key=_score_candidate)
    if unique_candidates:
        selected = unique_candidates[0]
        messages.append(f"Ejecutable seleccionado: {selected}")
        return LauncherSearchResult(
            executable=selected,
            checked_locations=checked,
            candidates=[str(candidate) for candidate in unique_candidates],
            messages=messages,
            selected_reason="auto_detected",
        )

    messages.append("No se encontro un ejecutable de MSC Nastran.")
    return LauncherSearchResult(
        executable=None,
        checked_locations=checked,
        candidates=[],
        messages=messages,
        selected_reason=None,
    )
