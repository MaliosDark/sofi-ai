# train_lora_kd.py
import os, json, random, math, torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, losses
from peft import LoraConfig, get_peft_model

random.seed(42)

class Jsonl(Dataset):
    def __init__(self, p): self.R = [json.loads(l) for l in open(p)]
    def __len__(self): return len(self.R)
    def __getitem__(self, i): r = self.R[i]; return r["q"], r["d"], float(r.get("kd", r.get("score", 1.0)))

def collate(batch):
    a = [x[0] for x in batch]; b = [x[1] for x in batch]
    import torch; y = torch.tensor([x[2] for x in batch], dtype=torch.float32)
    return a, b, y

def main():
    student_id = "MaliosDark/sofia-embedding-v1"
    teacher_id = "Qwen/Qwen3-Embedding-8B"
    data_path = "data/pairs.jsonl"  # use existing data
    out_dir = "./SOFIA-v2"

    s = SentenceTransformer(student_id, trust_remote_code=True)
    if hasattr(s, "auto_model"):
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="FEATURE_EXTRACTION")
        s.auto_model = get_peft_model(s.auto_model, cfg)

    t = SentenceTransformer(teacher_id)

    ds = Jsonl(data_path); dl = DataLoader(ds, batch_size=128, shuffle=True, collate_fn=collate, drop_last=True)
    loss = losses.CosineSimilarityLoss(s)

    steps = len(dl) * 2
    warmup = int(steps * 0.06)
    s.fit(train_objectives=[(dl, loss)], epochs=2, warmup_steps=warmup, optimizer_params={"lr": 2e-5},
          show_progress_bar=True, use_amp=True, output_path=out_dir)

    s.save(out_dir)
    print("Saved:", out_dir)

if __name__ == "__main__": main()
