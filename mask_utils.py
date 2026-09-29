import numpy as np

def normalize_image_b64(v):
    return v.split(",",1)[1] if "," in v and v.strip().lower().startswith("data:") else v

def encode_binary_mask(mask):
    flat=np.asarray(mask,dtype=np.uint8).reshape(-1); counts=[]; prev=0; run=0
    for p in flat:
        p=int(p!=0)
        if p==prev: run+=1
        else: counts.append(run); run=1; prev=p
    counts.append(run)
    return {"encoding":"ktp-rle-v1","size":[int(mask.shape[0]),int(mask.shape[1])],"counts":counts}
