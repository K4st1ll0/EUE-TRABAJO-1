# EUE FEM Correlation

Herramienta minima para el ajuste del modelo FEM con MSC Nastran en dos fases conectadas:

- Fase 1: ajuste de las 6 densidades `MAT1` con control de factibilidad dentro de bandas de `+/-5%`.
- Fase 2: correlacion modal a partir de la masa congelada, con parsing de frecuencias, matriz MAC, pairing y refinamientos locales pequenos sobre `E`.

## Estructura

- `src/`: CLI, editor BDF, calculo interno de masa, runner, reporting y workflow.
- `config/project.yaml`: configuracion unica.
- `runs/`: historial inmutable de ejecuciones.
- `reports/runs/`: reportes detallados por run.
- `handoff/index.html`: reporte maestro regenerado desde `runs/`.
- `handoff/IA.json`: estado resumido y trazable del proyecto.

## Comandos

Ejecuta siempre desde `06_Codigo/`.

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m src.main baseline
py -3 -m src.main feasibility-fit
py -3 -m src.main modal-baseline
py -3 -m src.main modal-fit
py -3 -m src.main modal-fit-local
py -3 -m src.main modal-fit-balanced-local
py -3 -m src.main modal-fit-targeted
py -3 -m src.main sweep
py -3 -m src.main rebuild-report
py -3 -m src.main show-config
```

## Configuracion

Todo vive en `config/project.yaml`.

- `paths.base_bdf`: BDF base del SFEM.
- `runner.command`: comando real de Nastran.
- `materials`: las 6 `MAT1` activas, independientes entre si.
- `mass_fit_controls`: definicion operativa de PSU, MOD, PROC, FPGA y Backplanes.
- `mass_fit_targets`: targets y bandas `+/-5%`.
- `modal`: referencia modal, run congelada de masa, puntos de medida, orden del vector modal, umbrales MAC y pairing.
- `modal.fit`: barrido `one-at-a-time` de `E`.
- `modal.local_fit`: refinamiento local estricto alrededor de los mejores candidatos del primer modal-fit.
- `modal.balanced_local_fit`: refinamiento local balanceado alrededor del candidato fisicamente mas equilibrado.
- `modal.diagnostics`: clasificacion `Good / Acceptable / Problematic` y numero de DOFs dominantes por modo.
- `modal.local_sensitivity`: sensibilidad modal local de `E` en torno a la run recomendada.
- `modal.targeted_fit`: micro-refinamiento dirigido a partir del diagnostico modal local.

La configuracion de trabajo real usa el comando local de MSC Nastran y mantiene congelada la masa desde la run:

`005_Al0.500_PSU1.586_MOD1.430_PROC1.701_FPGA1.102_BFPGA3.063`

## Fase 1 Masa

- `baseline` crea la run con factores `1.0` para las seis densidades.
- `feasibility-fit` busca una solucion operativa dentro de las bandas de `+/-5%`.
- La seleccion prioriza: `1) restricciones cumplidas`, `2) violacion maxima`, `3) error de masa total`, `4) cambio respecto a baseline`.

## Fase 2 Modal

- `modal-baseline` parte de la run congelada `005`.
- `modal-fit` barre `E` de cada `MAT1` una a una, sin tocar densidades.
- `modal-fit-local` hace un refinamiento local estricto con prioridad sobre `mean MAC`.
- `modal-fit-balanced-local` hace un refinamiento local alrededor de la run equilibrada, usando prioridad balanceada.
- `modal-fit-targeted` diagnostica modo a modo la run recomendada, lanza sensibilidad modal local de `+/-2%` y prueba solo unos pocos candidatos dirigidos.

La referencia modal actual se lee desde:

`01_DFEM-de-referencia/01_Resultados-autovectores-acelerometros.f06`

Se comparan las 10 primeras frecuencias y se construye el vector modal con los nodos:

- `486`
- `452`
- `2040`
- `402`

El MAC se calcula como:

```text
MAC = ((phi_ref^T phi_model)^2) / ((phi_ref^T phi_ref)(phi_model^T phi_model))
```

## Rankings Modales

El proyecto mantiene dos rankings modales distintos:

- `STRICT_MODAL_RANKING`: `1) mean MAC`, `2) mean frequency error`, `3) worst MAC`, `4) worst frequency error`
- `BALANCED_MODAL_RANKING`: `1) good_on_both`, `2) mean frequency error`, `3) mean MAC`, `4) worst MAC`, `5) worst frequency error`

`handoff/index.html` e `handoff/IA.json` muestran:

- mejor run por ranking estricto
- mejor run por ranking balanceado
- run recomendada para seguir trabajando
- diagnostico modo a modo de la run de referencia de fase 2.3
- sensibilidad modal local y recomendacion automatica del siguiente micro-paso
- resultados del `modal-fit-targeted` si ya se ha ejecutado

## Reportes

Los reportes detallados y el handoff maestro incluyen:

- matriz MAC y pairing
- diagnostico por modo emparejado
- priorizacion de modos problematicos
- sensibilidad modal local alrededor de la run recomendada
- recomendacion automatica del siguiente ajuste de `E`
- ranking de candidatos modales
- refinamiento local estricto y balanceado
- targeted modal fit con comparacion frente a la run recomendada
- tabla `Mass Budget Comparison`
- tabla `Adjusted Material Mass Breakdown`
- error porcentual de masa total y de cada componente frente al budget

En la tabla de budget se usa este mapeo operativo:

- `H-PSU module -> PCB_PSU`
- `MOD module -> PCB_MOD`
- `PROC module -> PCB_PROC`
- `FPGA module -> PCB_FPGA + Bandeja_FPGA shared`
- `Backplane modules -> Bandeja_FPGA`
- `Total mass -> total model mass`

## Hipotesis fisicas activas

- Solo se ajustan `rho` de las `MAT1` 1..6 en fase 1.
- En fase 2 no se tocan densidades; solo `E`.
- `Bandeja_FPGA` participa tanto en `FPGA module` como en `Backplane modules`.
- La suma de partidas objetivo es `3811 g` y el target total es `3810 g`; la herramienta lo detecta y lo reporta sin corregirlo.
- El estimador interno cubre `GRID`, `MAT1`, `PSHELL`, `PBARL` tipo `BAR`, `CQUAD4`, `CTRIA3` y `CBAR`.
