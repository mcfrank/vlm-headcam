"""Add speaker (via time-overlap join to the token transcript) to the grid manifests and write
child-filtered variants: drop the child's own utterances (KCHI/OCH), keeping the caregiver
teaching signal. Two variants: nochild (drop confirmed child) and careg (FEM/MAL only)."""
import numpy as np
import pandas as pd
from common import PARSED

tok = pd.read_csv(PARSED, usecols=["video_id", "utterance_id", "token_start_time",
                                   "token_end_time", "speaker"])
g = tok.groupby(["video_id", "utterance_id"])
utt = pd.DataFrame({"start": g.token_start_time.min(), "end": g.token_end_time.max(),
                    "speaker": g.speaker.agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "unknown")}
                   ).reset_index().dropna(subset=["start", "end"]).sort_values("start")


def add_speaker(df):
    d = df.copy()
    d["t"] = d.frame_idx.astype(float)
    d = d.sort_values("t")
    m = pd.merge_asof(d, utt, left_on="t", right_on="start", by="video_id", direction="backward")
    m.loc[~(m.t <= m.end + 1).fillna(False), "speaker"] = np.nan
    return m


cols = ["video_id", "frame_idx", "text"]
for name in ["grid_baseline_train", "grid_t15_filtnat_train"]:
    df = pd.read_parquet(f"manifests/{name}.parquet")
    m = add_speaker(df)
    child = m.speaker.isin(["KCHI", "OCH"])
    careg = m.speaker.isin(["FEM", "MAL"])
    m[~child][cols].to_parquet(f"manifests/{name}_nochild.parquet", index=False)
    m[careg][cols].to_parquet(f"manifests/{name}_careg.parquet", index=False)
    print(f"{name}: all {len(df)} | nochild {int((~child).sum())} | caregiver-only {int(careg.sum())} "
          f"| child dropped {int(child.sum())}")
