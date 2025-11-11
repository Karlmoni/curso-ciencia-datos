# Model Card — RFS
**Versión:** v1  
**Entorno:** Python 3.12.7 | scikit-learn 1.4.2

## Datos
Archivo: `credito_cleandata_4model.csv`  
Shape: (60, 7)  
Objetivo: `Aprobado` (NEG=0, POS=1)  
Prevalencia (POS=1) — TRAIN: 0.500 | TEST: 0.500

## Entrenamiento
Split 80/20 estratificado (random_state=42).  
Preprocesamiento: StandardScaler(num) + OneHotEncoder(ignore) (cat) + SMOTE.

## Modelo seleccionado
**RFS**  
Umbral de decisión (Paso 8, CV-TRAIN): **0.48**.

## Métricas en TEST (umbral aplicado)
ACC=1.000 | BALACC=1.000 | PREC=1.000 | REC=1.000 | F1=1.000 | MCC=1.000  
ROC-AUC=1.000 | PR-AUC=1.000

## Artefactos
- `pipeline_RFS.joblib`
- `input_schema.json`
- `label_map.json`
- `decision_policy.json`
