# Functional Budget Imputation Audit - Run 023

Run: `023_Al1.601_PCB1.000_ALE0.534_PCBE1.000`
BDF: `C:\Users\Pablo Castillo\Desktop\MUSE\SEGUNDO CUATRIMESTRE\ESTRUCTURAS\EUE-TRABAJO-1\Trabajo_EUE_25-26\06_Codigo\runs\023_Al1.601_PCB1.000_ALE0.534_PCBE1.000\SFEM_EUE.bdf`
Config source: `live project_config.yaml`
Config path: `C:\Users\Pablo Castillo\Desktop\MUSE\SEGUNDO CUATRIMESTRE\ESTRUCTURAS\EUE-TRABAJO-1\Trabajo_EUE_25-26\06_Codigo\configs\project_config.yaml`
Mass source of truth: `C:\Users\Pablo Castillo\Desktop\MUSE\SEGUNDO CUATRIMESTRE\ESTRUCTURAS\EUE-TRABAJO-1\Trabajo_EUE_25-26\06_Codigo\runs\023_Al1.601_PCB1.000_ALE0.534_PCBE1.000\mass_summary.json`

## Methodological Notes

- z_band_split is an approximate geometric rule based on z-bands.
- rig base -> Backplane modules is a working functional hypothesis, not a geometrically demonstrated truth.

## Legacy Mapping Snapshot

- subset_rows invariant vs previous functional audit: `True`

## Budget Imputation: Before vs After

| Category | Target [kg] | Before Imputed [kg] | After Imputed [kg] | Change [kg] | Before Delta [%] | After Delta [%] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H-PSU module | 0.912546 | 0.865502 | 0.865502 | +0.000000 | -5.155% | -5.155% |
| MOD module | 0.822639 | 0.892218 | 0.892218 | +0.000000 | +8.458% | +8.458% |
| FPGA module | 0.870768 | 0.904894 | 0.904894 | +0.000000 | +3.919% | +3.919% |
| PROC module | 0.978888 | 0.992889 | 0.992889 | +0.000000 | +1.430% | +1.430% |
| Backplane modules | 0.224966 | 0.154303 | 0.154303 | +0.000000 | -31.410% | -31.410% |

## Residual After Imputation

| Before [kg] | After [kg] | Change [kg] |
| ---: | ---: | ---: |
| 0.000000 | 0.000000 | +0.000000 |

## Functional Warnings

- Before: `["mass_compensation_detected: positive=['MOD module'], negative=['H-PSU module', 'Backplane modules']"]`
- After: `["mass_compensation_detected: positive=['MOD module'], negative=['H-PSU module', 'Backplane modules']"]`
- Added: `[]`
- Removed: `[]`

## Families

| Family | Contributions | Mass [kg] |
| --- | ---: | ---: |
| PSU board | 32 | 0.546771 |
| MOD board | 32 | 0.546771 |
| FPGA board | 32 | 0.546771 |
| PROC board | 32 | 0.546771 |
| FPGA tray | 32 | 0.117614 |
| common trays | 96 | 0.352841 |
| rig base | 8 | 0.154303 |
| top cover | 64 | 0.084010 |
| thin walls | 88 | 0.280498 |
| corner/frame bars | 452 | 0.633458 |
| real backplane parts | 0 | 0.000000 |
| unmapped residual | 0 | 0.000000 |

## Allocation Rules Applied

| Family | Mode | Target | Mass [kg] |
| --- | --- | --- | ---: |
| PSU board | direct | H-PSU module | 0.546771 |
| MOD board | direct | MOD module | 0.546771 |
| FPGA board | direct | FPGA module | 0.546771 |
| PROC board | direct | PROC module | 0.546771 |
| FPGA tray | direct | FPGA module | 0.117614 |
| common trays | z_band_split | H-PSU module | 0.117614 |
| common trays | z_band_split | MOD module | 0.117614 |
| common trays | z_band_split | PROC module | 0.117614 |
| rig base | direct | Backplane modules | 0.154303 |
| top cover | proportional | H-PSU module | 0.021385 |
| top cover | proportional | MOD module | 0.019278 |
| top cover | proportional | FPGA module | 0.020406 |
| top cover | proportional | PROC module | 0.022940 |
| thin walls | z_band_split | FPGA module | 0.073444 |
| thin walls | z_band_split | H-PSU module | 0.055514 |
| thin walls | z_band_split | MOD module | 0.073444 |
| thin walls | z_band_split | PROC module | 0.078096 |
| corner/frame bars | z_band_split | FPGA module | 0.146659 |
| corner/frame bars | z_band_split | H-PSU module | 0.124218 |
| corner/frame bars | z_band_split | MOD module | 0.135111 |
| corner/frame bars | z_band_split | PROC module | 0.227469 |
| real backplane parts | direct | Backplane modules | 0.000000 |
