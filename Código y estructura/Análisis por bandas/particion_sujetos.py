"""Partición reproducible de desarrollo y prueba externa para análisis por bandas."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED_PARTICION = 42
N_SUJETOS_TOTAL = 40
N_SUJETOS_DESARROLLO = 32


def obtener_particion(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    if "Sujeto" not in df.columns:
        raise ValueError("El conjunto de datos no contiene la columna Sujeto.")
    # Mantener el ordenamiento sobre el tipo original, igual que el análisis global.
    sujetos = np.sort(df["Sujeto"].dropna().unique())
    if len(sujetos) != N_SUJETOS_TOTAL:
        raise ValueError(f"Se esperaban 40 sujetos; se encontraron {len(sujetos)}.")
    barajados = np.random.RandomState(SEED_PARTICION).permutation(sujetos)
    return (
        [str(sujeto) for sujeto in barajados[:N_SUJETOS_DESARROLLO]],
        [str(sujeto) for sujeto in barajados[N_SUJETOS_DESARROLLO:]],
    )


def filtrar_desarrollo(df: pd.DataFrame, sujetos_desarrollo: list[str], nombre: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"El conjunto {nombre} está vacío.")
    if "Sujeto" not in df.columns:
        raise ValueError(f"El conjunto {nombre} no contiene la columna Sujeto.")
    ids = set(df["Sujeto"].dropna().astype(str).unique())
    esperados = set(sujetos_desarrollo)
    faltantes = esperados - ids
    if faltantes:
        raise ValueError(f"{nombre}: faltan sujetos de desarrollo: {sorted(faltantes)}")
    return df[df["Sujeto"].astype(str).isin(esperados)].copy()


def guardar_particion(ruta: Path, desarrollo: list[str], prueba: list[str]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos = {
        "semilla": SEED_PARTICION,
        "sujetos_desarrollo_32": desarrollo,
        "sujetos_prueba_externa_8": prueba,
        "criterio": "SAM=0 frente a SAM>=1",
        "uso": "Los rankings y clasificadores por bandas usan solo los 32 sujetos de desarrollo.",
    }
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def cargar_particion(ruta: Path, df: pd.DataFrame) -> tuple[list[str], list[str]]:
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta primero 1_seleccion_caracteristicas_B.py para crear la partición 32/8."
        )
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    desarrollo = datos["sujetos_desarrollo_32"]
    prueba = datos["sujetos_prueba_externa_8"]
    if len(desarrollo) != 32 or len(prueba) != 8:
        raise ValueError("El archivo de partición no contiene exactamente 32/8 sujetos.")
    actual = set(df["Sujeto"].dropna().astype(str).unique())
    if not (set(desarrollo) | set(prueba)).issubset(actual):
        raise ValueError("La partición guardada no coincide con los sujetos de los datos actuales.")
    return desarrollo, prueba
