# Correlacion FEM E-Box - Fase 1

Base Python para inspeccion del modelo FEM, preparacion de baseline modal y documentacion HTML automatica usando BDF/F06 en Windows.

## Requisitos

- Windows
- Python 3.11 recomendado
- MSC Nastran opcional para lanzar la corrida baseline

## Arranque rapido

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Comandos

Desde esta carpeta:

```powershell
python -m src.cli inspect
python -m src.cli baseline
python -m src.cli docs
```

En este equipo es mas fiable:

```powershell
py -3.11 -m src.cli inspect
py -3.11 -m src.cli baseline
py -3.11 -m src.cli docs
```

## Salidas principales

- `docs/index.html`
- `docs/reports/bdf_summary.html`
- `05_Modelo_FEM/Iteraciones/baseline/run_XXXXXX/report.html`
- `docs/runs/baseline_run_XXXXXX.html`

## Alcance de esta fase

- Carga de configuracion YAML
- Inspeccion del BDF base
- Registro de parametros editables
- Preparacion y lanzamiento baseline
- Parser inicial de F06 modal
- Comparacion modal por frecuencias
- Trazabilidad de runs
- Documentacion HTML estatica

## Pendiente para fase 2

- Edicion automatica de parametros del BDF
- Campanas de iteraciones
- Correlacion con sensores y autovectores
- Optimizacion automatica
