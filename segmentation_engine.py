import base64,io,os,time
import numpy as np, torch
from PIL import Image
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from mask_utils import encode_binary_mask,normalize_image_b64

MODEL_ID=os.getenv("SAM2_MODEL_ID","facebook/sam2.1-hiera-large")
def decode_image(v):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(normalize_image_b64(v)))).convert("RGB"),dtype=np.uint8)

class Engine:
    def __init__(self):
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.model=build_sam2_hf(MODEL_ID,device=self.device)
        self.predictor=SAM2ImagePredictor(self.model)
        self.generator=SAM2AutomaticMaskGenerator(self.model,points_per_side=32,pred_iou_thresh=.86,stability_score_thresh=.92,crop_n_layers=1,min_mask_region_area=180,output_mode="binary_mask")
    def health(self):
        return {"status":"ok","engine":"SAM2.1","model":MODEL_ID,"device":self.device,"ready":True,"cuda":torch.cuda.is_available()}
    def _predict(self,img,points=None,labels=None,box=None):
        h,w=img.shape[:2]; self.predictor.set_image(img)
        p=None if not points else np.asarray(points,dtype=np.float32)
        l=None if not labels else np.asarray(labels,dtype=np.int32)
        b=None if box is None else np.asarray(box,dtype=np.float32)
        masks,scores,_=self.predictor.predict(point_coords=p,point_labels=l,box=b,multimask_output=True)
        i=int(np.argmax(scores))
        return {"width":w,"height":h,"score":float(scores[i]),"mask":encode_binary_mask(masks[i].astype(bool))}
    def point(self,image,points,labels): return self._predict(decode_image(image),points,labels)
    def box(self,image,box): return self._predict(decode_image(image),box=box)
    def refine(self,image,points,labels,box): return self._predict(decode_image(image),points,labels,box)
    def auto(self,image,max_masks=40,min_area_ratio=.0005):
        t=time.time(); img=decode_image(image); h,w=img.shape[:2]; min_area=max(64,int(h*w*min_area_ratio))
        anns=[a for a in self.generator.generate(img) if int(a.get("area",0))>=min_area]
        anns.sort(key=lambda a:(float(a.get("predicted_iou",0)),float(a.get("stability_score",0)),int(a.get("area",0))),reverse=True)
        kept=[]
        for a in anns:
            m=np.asarray(a["segmentation"],dtype=bool)
            duplicate=False
            for o in kept:
                inter=np.logical_and(m,o["_m"]).sum(); union=np.logical_or(m,o["_m"]).sum()
                if union and inter/union>.92: duplicate=True; break
            if duplicate: continue
            kept.append({"_m":m,"score":float(a.get("predicted_iou",0)),"stability":float(a.get("stability_score",0)),"area":int(a.get("area",m.sum())),"bbox":[float(x) for x in a.get("bbox",[0,0,w,h])]})
            if len(kept)>=max_masks: break
        masks=[{"id":f"A{i:03d}","name":f"Auto Layer {i:02d}","score":o["score"],"stability_score":o["stability"],"area":o["area"],"bbox":o["bbox"],"mask":encode_binary_mask(o["_m"])} for i,o in enumerate(kept,1)]
        return {"width":w,"height":h,"engine":"SAM2.1","model":MODEL_ID,"mask_encoding":"ktp-rle-v1","masks":masks,"elapsed_ms":round((time.time()-t)*1000)}
engine=Engine()
