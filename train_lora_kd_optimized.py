#!/usr/bin/env python3
"""
SOFIA LoRA KD Training - Versión Optimizada
Usa modelo local o más pequeño para evitar problemas de descarga
"""

import os
import json
import random
import math
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, losses, InputExample
from peft import LoraConfig, get_peft_model

random.seed(42)

class JsonlDataset(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path)]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        return InputExample(texts=[r["q"], r["d"]], label=float(r.get("kd", r.get("score", 1.0))))

def collate_fn(batch):
    # SentenceTransformers maneja InputExample automáticamente
    return batch

def main():
    # Configuración optimizada
    student_model = "./SOFIA"  # Use local SOFIA model
    teacher_model = "sentence-transformers/all-mpnet-base-v2"  # Modelo más pequeño y confiable
    data_path = "data/pairs.jsonl"  # Tus datos existentes
    output_dir = "./SOFIA-v2-lora"

    print("🚀 Iniciando LoRA KD Training Optimizado")
    print(f"📚 Student: {student_model}")
    print(f"👨‍🏫 Teacher: {teacher_model}")
    print(f"📄 Datos: {data_path}")
    print(f"💾 Output: {output_dir}")

    # Cargar modelos
    print("🔄 Cargando modelos...")
    try:
        student = SentenceTransformer(student_model)
        teacher = SentenceTransformer(teacher_model)
        print("✅ Modelos cargados exitosamente")
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
        print("💡 Asegúrate de que el modelo SOFIA existe en ./SOFIA")
        return

    # Configurar LoRA
    if hasattr(student, "auto_model"):
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        student.auto_model = get_peft_model(student.auto_model, lora_config)
        print("🔧 LoRA configurado")

    # Preparar datos
    try:
        dataset = JsonlDataset(data_path)
        dataloader = DataLoader(
            dataset,
            batch_size=32,  # Batch más pequeño para estabilidad
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        print(f"📊 Dataset cargado: {len(dataset)} ejemplos")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        print("💡 Verifica que existe sofia/data/pairs.jsonl")
        return

    # Configurar pérdida simple (MultipleNegativesRankingLoss funciona bien con LoRA)
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    train_loss = MultipleNegativesRankingLoss(student)

    # Calcular pasos
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * 2  # 2 epochs
    warmup_steps = int(total_steps * 0.06)

    print(f"⚙️  Configuración:")
    print(f"   • Epochs: 2")
    print(f"   • Batch size: 32")
    print(f"   • Steps per epoch: {steps_per_epoch}")
    print(f"   • Warmup steps: {warmup_steps}")
    print(f"   • Learning rate: 2e-5")

    # Entrenamiento
    try:
        print("🏃 Iniciando entrenamiento...")
        student.fit(
            train_objectives=[(dataloader, train_loss)],
            epochs=2,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": 2e-5},
            show_progress_bar=True,
            use_amp=True,
            output_path=output_dir
        )
        print(f"✅ Entrenamiento completado! Modelo guardado en {output_dir}")
    except Exception as e:
        print(f"❌ Error durante entrenamiento: {e}")
        return

    # Verificar modelo entrenado
    try:
        print("🔍 Verificando modelo entrenado...")
        test_model = SentenceTransformer(output_dir)
        test_emb = test_model.encode(["test sentence"])
        print(f"✅ Modelo verificado: embedding shape {test_emb.shape}")
    except Exception as e:
        print(f"⚠️  Advertencia: No se pudo verificar el modelo: {e}")

    print("🎉 ¡SOFIA v2 con LoRA KD completado!")

if __name__ == "__main__":
    main()
