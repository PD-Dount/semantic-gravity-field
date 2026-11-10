"""Does the learnable shape exponent p ever move? Run A3 with two kappa inits."""
import json
import sys

from gravity_lab.config import AttnConfig, ModelConfig, TrainConfig
from gravity_lab.train import run_training

mcfg = ModelConfig(vocab=32, seq_len=64, d_model=48, n_layers=3, n_heads=4)
out = []
for kappa in (0.1, 1.0, 3.0):
    acfg = AttnConfig.from_arm("A3", d_model=48, n_heads=4, kappa_init=kappa)
    tcfg = TrainConfig(task="mqar", steps=400, batch_size=32, lr=3e-3, seed=0,
                       eval_every=200, eval_batches=4, log_every=1000, threads=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
    r = run_training(tcfg, mcfg, acfg, out_path=None, verbose=False)
    st = r["stats"]
    out.append({"kappa_init": kappa, "p_mean": st.get("p_mean"), "kappa": st.get("kappa"),
                "acc": r["final_acc"], "loss": r["final_loss"]})
    print("kappa_init=%.2f -> p_mean=%.4f  kappa=%.4f  acc=%.3f loss=%.3f"
          % (kappa, st.get("p_mean") or -1, st.get("kappa") or -1, r["final_acc"], r["final_loss"]), flush=True)

with open("results/diag_p.json", "w") as f:
    json.dump(out, f, indent=2)
