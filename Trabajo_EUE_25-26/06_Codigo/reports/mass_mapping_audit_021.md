# Mass Mapping Audit - Run 021

Run analizada: `021_Al1.601_PCB1.000_ALE0.534_PCBE1.000`
BDF auditado: `C:\Users\Pablo Castillo\Desktop\MUSE\SEGUNDO CUATRIMESTRE\ESTRUCTURAS\EUE-TRABAJO-1\Trabajo_EUE_25-26\06_Codigo\runs\021_Al1.601_PCB1.000_ALE0.534_PCBE1.000\SFEM_EUE.bdf`

## Resumen

| Subset | Target [kg] | Computed [kg] | Delta [kg] | Delta [%] | Element count |
| --- | ---: | ---: | ---: | ---: | ---: |
| H-PSU module | 0.912546 | 0.701074 | -0.211472 | -23.2% | 40 |
| MOD module | 0.822639 | 0.546771 | -0.275868 | -33.5% | 32 |
| FPGA module | 0.870768 | 0.546771 | -0.323997 | -37.2% | 32 |
| PROC module | 0.978888 | 0.546771 | -0.432117 | -44.1% | 32 |
| Backplane modules | 0.224966 | 0.470454 | +0.245489 | +109.1% | 128 |
| Unmapped | - | 0.997966 | - | - | 604 |

## Diagn?stico r?pido

- El mapping actual deja `0.997966 kg` fuera de cualquier subset, que es aproximadamente el `26.2%` de la masa FEM total.
- `Backplane modules` est? inflado ?ntegramente por dos regiones expl?citas: `07_t-bandejas` y `07_t-bandejas_FPGA`, que suman `0.470454 kg` frente a un target de `0.224966 kg`.
- `MOD module`, `FPGA module` y `PROC module` no reciben el mismo conjunto de elementos, pero s? reciben exactamente el mismo patr?n de mapping: solo su PCB dedicada, con el mismo n?mero de paneles y la misma masa total.
- La incoherencia principal ahora mismo apunta a un mapping incompleto del FEM hacia el budget, no a un fallo del ajuste de densidades.

## Qu? est? inflando Backplane

| region_name | property_id | material_id | element_count | mass contribution [kg] |
| --- | ---: | ---: | ---: | ---: |
| 07_t-bandejas | 24 | 1 | 96 | 0.352841 |
| 07_t-bandejas_FPGA | 29 | 6 | 32 | 0.117614 |

- `07_t-bandejas` aporta `0.352841 kg`, que ya supera por s? sola el target completo de Backplane (`0.224966 kg`).
- `07_t-bandejas_FPGA` a?ade `0.117614 kg` extra.

## Top masa no mapeada por regi?n

| region_name | element_count | mass contribution [kg] |
| --- | ---: | ---: |
| 11_ESQ_xp_yp | 452 | 0.633458 |
| 01_t-wall-delgadas | 88 | 0.280498 |
| 05_t-tapa | 64 | 0.084010 |

## Detalle por subset

### H-PSU module

| region_name | property_id | material_id | element_count | mass contribution [kg] | element types |
| --- | ---: | ---: | ---: | ---: | --- |
| 09_A_t-boards-PSU | 25 | 2 | 32 | 0.546771 | CQUAD4:32 |
| 06_t-rig-base | 23 | 1 | 8 | 0.154303 | CQUAD4:8 |

### MOD module

| region_name | property_id | material_id | element_count | mass contribution [kg] | element types |
| --- | ---: | ---: | ---: | ---: | --- |
| 09_B_t-boards-MOD | 26 | 3 | 32 | 0.546771 | CQUAD4:32 |

### FPGA module

| region_name | property_id | material_id | element_count | mass contribution [kg] | element types |
| --- | ---: | ---: | ---: | ---: | --- |
| 09_D_t-boards-FPGA | 28 | 5 | 32 | 0.546771 | CQUAD4:32 |

### PROC module

| region_name | property_id | material_id | element_count | mass contribution [kg] | element types |
| --- | ---: | ---: | ---: | ---: | --- |
| 09_C_t-boards-PROC | 27 | 4 | 32 | 0.546771 | CQUAD4:32 |

### Backplane modules

| region_name | property_id | material_id | element_count | mass contribution [kg] | element types |
| --- | ---: | ---: | ---: | ---: | --- |
| 07_t-bandejas | 24 | 1 | 96 | 0.352841 | CQUAD4:96 |
| 07_t-bandejas_FPGA | 29 | 6 | 32 | 0.117614 | CQUAD4:32 |

## Interpretaci?n

- `H-PSU module` contiene dos bloques: la PCB PSU (`09_A_t-boards-PSU`) y la base `06_t-rig-base`. Por eso no coincide con los otros tres m?dulos.
- `MOD module`, `FPGA module` y `PROC module` contienen una sola regi?n cada uno: `09_B_t-boards-MOD`, `09_D_t-boards-FPGA` y `09_C_t-boards-PROC`. Cada una tiene `32` elementos `CQUAD4` y masa `0.546771 kg`.
- Por tanto, no parece que el c?digo est? metiendo el mismo conjunto exacto en los tres subsets; lo que ocurre es que el mapping actual solo recoge sus PCBs y deja fuera la estructura asociada.
- La masa no mapeada m?s importante est? en `01_t-wall-delgadas`, `11_ESQ_xp_yp` y `05_t-tapa`. Si esas regiones deben imputarse a m?dulos concretos en el budget, el mapping actual est? claramente incompleto.
- Con el mapping actual, intentar arreglar los porcentajes por subset solo con dos factores globales de densidad no es una estrategia fiable: el total se puede clavar, pero el reparto seguir? sesgado mientras casi 1 kg quede fuera del presupuesto por subset o mientras Backplane absorba regiones demasiado grandes.

## Conclusi?n provisional

- La evidencia apunta primero a `mapping est? mal o incompleto` para comparaci?n por subsets.
- Tambi?n es posible que `el troceado FEM no corresponda 1:1 al budget`, sobre todo en envolvente, esquinas y tapa.
- `Dos factores globales de densidad no bastan` para cuadrar reparto por subsets con el mapping actual; antes hay que decidir c?mo imputar la estructura no PCB al Mass Budget.