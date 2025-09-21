"""
train_sofia.py
Trains SOFIA with LoRA adapters + multi-loss (cosine + triplet).
Saves base in ./SOFIA and projection variants in ./SOFIA/proj-<dim>.
"""
import os, argparse, json, math, torch
from torch.utils import data
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, models, InputExample
from peft import LoraConfig, get_peft_model

class JsonlDataset(data.Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path)]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        label = float(r.get("kd", r.get("score", 0.0)))
        return InputExample(texts=[r["q"], r["d"]], label=label)

def collate(batch):
    return batch

def projection_head(dim_in, dim_out):
    return models.Dense(in_features=dim_in, out_features=dim_out, bias=True, activation_function=torch.nn.Identity())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--train", default="./data/pairs.jsonl")
    ap.add_argument("--out", default="./SOFIA")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--dims", nargs="+", type=int, default=[512,3072,4096])
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--triplet_margin", type=float, default=0.1)
    args = ap.parse_args()

    base = SentenceTransformer(args.base, trust_remote_code=True)
    if hasattr(base, "auto_model"):
        peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none", task_type="FEATURE_EXTRACTION")
        base.auto_model = get_peft_model(base.auto_model, peft_cfg)

    # Add a projection head (we train with first head; later export other dims)
    base._modules["2"] = projection_head(base.get_sentence_embedding_dimension(), args.dims[0])

    ds = JsonlDataset(args.train)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, drop_last=True)

    cos_loss = losses.CosineSimilarityLoss(base)
    triplet_loss = losses.TripletLoss(base, triplet_margin=args.triplet_margin)

    steps_per_epoch = len(dl)
    warmup_steps = math.ceil(steps_per_epoch * args.epochs * args.warmup)

    base.fit(
        train_objectives=[(dl, cos_loss), (dl, triplet_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        use_amp=args.fp16,
        output_path=args.out
    )

    # Export extra dimensions
    for d in args.dims[1:]:
        m2 = SentenceTransformer(args.out)
        m2._modules["1"] = projection_head(m2.get_sentence_embedding_dimension(), d)
        m2.save(os.path.join(args.out, f"proj-{d}"))

    print("saved:", args.out)
