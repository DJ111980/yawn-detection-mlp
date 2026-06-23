# Recolección de datos

## Estado actual

El dataset está organizado por clase y por división (`train`, `validation`, `test`) y contiene 315 imágenes por clase. El balance se comprueba con `python tools/dataset_summary.py`.

## Criterios de calidad

- Incluir distintas personas, tonalidades de piel, edades aparentes y rasgos faciales.
- Incluir iluminación interior, exterior, tenue y frontal cuando sea posible.
- Incluir rostros frontales, inclinados y a distancias diferentes.
- Etiquetar como `yawn` únicamente bostezos visibles.
- Etiquetar como `no_yawn` bocas cerradas, conversación, sonrisa, risa y otras expresiones sin bostezo.
- Evitar duplicados y dividir imágenes muy similares en el mismo split para prevenir fuga de datos.

## Trazabilidad pendiente

Antes de la entrega, el integrante responsable debe añadir las fuentes concretas de cada grupo de imágenes, licencia o permiso de uso, fecha de recolección y número de imágenes aportadas por cada fuente. No se deben inventar esos datos.
