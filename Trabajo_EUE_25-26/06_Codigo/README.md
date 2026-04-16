# FEM Automation with MSC Nastran

Proyecto nuevo en Python para automatizar este flujo minimo extremo a extremo:

1. editar `MAT1` en un BDF copiado
2. lanzar MSC Nastran con `subprocess`
3. parsear frecuencias y autovectores desde `f06`
4. construir `phi` en el orden fijado
5. calcular matriz MAC `10x10`, diagonal `i-i` y pairing por Hungarian
6. calcular score global
7. generar un HTML profesional y trazable por corrida

## Ubicacion

Todo el proyecto vive en:

```text
Trabajo_EUE_25-26/06_Codigo/
```

## Estado actual

- `full-run` y `modal-fit` ya funcionan para una unica combinacion fija por configuracion.
- `mass-fit` ya ejecuta barridos de densidad usando las `range` de los parametros `rho`.
- `sweep` permite barridos deterministas por preset.
- El BDF original nunca se modifica.
- Cada corrida vive en `runs/<run_id>/` dentro de `06_Codigo`.
- Las nuevas corridas se nombran como `NNN_AlX_PCBY_ALEZ_PCBEO`, por ejemplo `001_Al0.800_PCB1.000_ALE1.000_PCBE1.000`.
- El prefijo `NNN` sube de forma secuencial para que la corrida mas reciente quede al final al ordenar por nombre.
- El reporte HTML compara la masa total y el reparto por modulo frente al mass budget Excel.

## Instalacion

Desde `Trabajo_EUE_25-26/06_Codigo`:

```powershell
py -3 -m pip install -r requirements.txt
```

## Estructura

```text
src/
configs/
data/
runs/
reports/
tests/
```

## Configuracion

Edita estos archivos:

- `configs/project_config.yaml`
- `configs/sensors.yaml`
- `configs/parameters.yaml`

Lo que normalmente tendras que completar tu:

- `nastran.executable` si en tu maquina no coincide con la ruta precargada
- `nastran.arguments` si tu licencia o flags locales cambian
- `mass.budget_basis` si quieres forzar una comparacion distinta al baseline activo; por defecto se usa `nominal`

Rutas precargadas desde esta carpeta:

- BDF base: `../05_Modelo_FEM/SFEM_EUE.bdf`
- F06 frecuencias referencia: `../01_DFEM-de-referencia/02_Resultados-modos-propios-globales.f06`
- F06 autovectores referencia: `../01_DFEM-de-referencia/01_Resultados-autovectores-acelerometros.f06`
- Excel de mass budget: `../04_Mass-Budget/MUSE_25-26_EUE_Mass Budget.xlsx`

## Ejecucion

Entra primero en `Trabajo_EUE_25-26/06_Codigo`.

Flujo minimo completo:

```powershell
py -3 -m src.main full-run
```

Alias modal:

```powershell
py -3 -m src.main modal-fit
```

Trazado sin ejecutar Nastran:

```powershell
py -3 -m src.main full-run --dry-run
```

Barrido de masa contra el target activo del budget:

```powershell
py -3 -m src.main mass-fit
```

Barridos por preset:

```powershell
py -3 -m src.main sweep --preset aluminum_e_fine
```

## Que guarda cada run

Dentro de `runs/<run_id>/` encontraras como minimo:

- BDF generado
- snapshot de configuracion
- logs `stdout/stderr`
- outputs localizados de Nastran
- `modifications.json`
- `mass_summary.json`
- `parsed_results.json`
- `metrics.json`
- `metrics.csv`
- `run_manifest.json`
- `report.html`

## Interpretacion del reporte

El HTML resume:

- masa objetivo vs masa obtenida
- masa global frente al mass budget
- diferencia por modulo frente al mass budget
- criterio de reparto FEM -> budget usado para la comparacion
- basis de masa usado en esa corrida: `BASIC` o `NOMINAL`
- parametros usados
- trazabilidad de cambios sobre `MAT1`
- frecuencias referencia vs modelo
- error relativo por modo
- matriz MAC `10x10`
- pairing modal final
- mean MAC, worst MAC y score
- logs, errores y archivos generados

## Tests

```powershell
py -3 -m pytest
```

## Notas de implementacion

- La edicion de `MAT1` reescribe las tarjetas modificadas en formato libre `MAT1,...` para dejar `G` implicito y mantener `nu` fijo.
- La masa intenta usar `pyNastran` y, si no esta disponible o no es compatible con la version local de Python, cae a un calculo manual para este modelo con `PBARL/CBAR` y `PSHELL/CQUAD4/CTRIA3`.
- La comparacion de mass budget lee el Excel real y clasifica las regiones FEM asi: placas PCB directas por modulo, `07_t-bandejas` y `07_t-bandejas_FPGA` como backplanes, y el resto de estructura por cercania en `z` al modulo.
- El pairing modal usa Hungarian con `MAC` como criterio principal y penalizacion suave por error de frecuencia.
