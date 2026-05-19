# E:/UAVagent/benchmark_real_mot_v2.py (v1.2.8 - 与30帧一致)
"""真实 MOT v2 - 100帧完整评估 (使用与 benchmark_real_mot.py 相同的逻辑)"""
import sys, os, cv2, json, numpy as np, time
sys.path.insert(0, "E:/UAVagent")
from config.settings import config
from core.vision_system import VisionSystem
import motmetrics as mm

VISDRONE_ROOT = "E:/datasets/VisDrone/VisDrone2019-MOT-val"
VISDRONE_SEQ = "uav0000086_00000_v"
MAX_FRAMES = 100

def load_gt(root, seq, mf):
    gt={}
    for an in [f"{seq}.txt", os.path.join(seq,"gt.txt")]:
        ap=os.path.join(root,"annotations",an)
        if os.path.isfile(ap):
            with open(ap) as f:
                for line in f:
                    p=line.strip().split(',')
                    if len(p)<8: continue
                    frame=int(p[0])-1
                    if frame>=mf: continue
                    cls=int(p[7]) if len(p)>7 else 0
                    if cls in [0,11]: continue
                    gt.setdefault(frame,[]).append({
                        'id':int(p[1]),
                        'bbox':[float(p[2]),float(p[3]),float(p[4]),float(p[5])],
                        'class':cls})
            break
    return gt

def load_frames(root, seq, mf):
    sp=os.path.join(root,"sequences",seq)
    frames=[]
    if os.path.isdir(sp):
        imgs=sorted([f for f in os.listdir(sp) if f.lower().endswith(('.jpg','.png'))])
        for fn in imgs[:mf]:
            img=cv2.imread(os.path.join(sp,fn))
            if img is not None: frames.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return frames

def evaluate_mot(gt_data, pred_data, n_frames):
    acc=mm.MOTAccumulator(auto_id=True)
    for fi in range(n_frames):
        g=gt_data.get(fi,[]); p=pred_data.get(fi,[])
        if not g and not p: continue
        if g and p:
            ious=np.zeros((len(g),len(p)))
            for i,gg in enumerate(g):
                gx,gy,gw,gh=gg['bbox']
                for j,pp in enumerate(p):
                    px,py,pw,ph=pp['bbox']
                    ix1,iy1=max(gx,px),max(gy,py)
                    ix2,iy2=min(gx+gw,px+pw),min(gy+gh,py+ph)
                    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
                    union=gw*gh+pw*ph-inter
                    ious[i,j]=inter/(union+1e-8)
        else: ious=np.zeros((len(g),len(p)))
        acc.update([gg['id'] for gg in g],[pp['id'] for pp in p],ious)
    mh=mm.metrics.create()
    return mh.compute(acc,metrics=mm.metrics.motchallenge_metrics,name='eval')

def main():
    print("="*70)
    print("UAVagent 1.2 VisDrone 100帧基准")
    print("="*70)
    print("\n[1/4] 加载数据...")
    frames=load_frames(VISDRONE_ROOT,VISDRONE_SEQ,MAX_FRAMES)
    gt=load_gt(VISDRONE_ROOT,VISDRONE_SEQ,MAX_FRAMES)
    print(f"  帧={len(frames)}  GT目标={sum(len(v) for v in gt.values())}")

    config.DETECTION_CONFIDENCE=0.25
    config.ENSEMBLE_IOU_THR=0.50

    print("\n[2/4] 初始化...")
    vs_single=VisionSystem(device="cpu",use_ensemble=False)
    vs_ensemble=VisionSystem(device="cpu",use_ensemble=True)
    print(f"  单模型: {vs_single.get_stats()['model_names'][0]}")
    print(f"  融合: {vs_ensemble.get_stats()['num_models']} models")

    print(f"\n[3/4] 运行跟踪 ({len(frames)} 帧)...")
    sp,ep={},{}
    s_total,e_total=0,0

    for fi,frame in enumerate(frames):
        ds=vs_single.process_frame(frame)
        sp[fi]=[{'id':d.get('id',-1),'bbox':d['bbox'],'confidence':d.get('confidence',0)} for d in ds]
        s_total+=len(ds)

        de=vs_ensemble.process_frame(frame)
        ep[fi]=[{'id':d.get('id',-1),'bbox':d['bbox'],'confidence':d.get('confidence',0)} for d in de]
        e_total+=len(de)

        if fi<3 or fi%30==0:
            print(f"  Frame {fi:3d}: GT={len(gt.get(fi,[])):2d}  单={len(ds):2d}  融合={len(de):2d}")

    print(f"\n[4/4] MOT 评估...")
    print(f"  单模型总检测: {s_total}")
    print(f"  融合总检测: {e_total}")

    sum_s=evaluate_mot(gt,sp,len(frames))
    sum_e=evaluate_mot(gt,ep,len(frames))

    if sum_s.empty or sum_e.empty: print("评估失败"); return
    row_s=sum_s.iloc[0]; row_e=sum_e.iloc[0]

    print(f"\n{'指标':<16} {'单模型':>10} {'融合':>10} {'变化':>10} {'结论':>8}")
    print("-"*56)
    for name,key in [('MOTA','mota'),('IDF1','idf1'),('Recall','recall'),
                     ('Precision','precision'),('IDP','idp'),('IDR','idr'),
                     ('ID Sw','num_switches'),('FP','num_false_positives'),
                     ('FN','num_misses')]:
        v1=float(row_s.get(key,0)); v2=float(row_e.get(key,0))
        diff=v2-v1
        if key in ['num_switches','num_false_positives','num_misses']:
            b='✅' if diff<=0 else '⚠️'
            print(f"{name:<16} {int(v1):>10} {int(v2):>10} {int(diff):>+10d} {b:>8}")
        else:
            b='✅' if diff>=0 else '⚠️'
            print(f"{name:<16} {v1:>10.3f} {v2:>10.3f} {diff:>+10.3f} {b:>8}")

    mota_d=(float(row_e.get('mota',0))-float(row_s.get('mota',0)))*100
    idf1_d=(float(row_e.get('idf1',0))-float(row_s.get('idf1',0)))*100
    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")
    print(f"  单模型 MOTA: {float(row_s.get('mota',0)):.3f}")
    print(f"  融合 MOTA: {float(row_e.get('mota',0)):.3f}")
    print(f"  MOTA 变化: {mota_d:+.1f} pp")
    print(f"  IDF1 变化: {idf1_d:+.1f} pp")
    if mota_d>0.5: print(f"\n  ✅ 融合超越单模型 {mota_d:.1f} pp")

    result={"timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset":f"visdrone_{VISDRONE_SEQ}","num_frames":len(frames),
            "total_gt":sum(len(v) for v in gt.values()),
            "single_detections":s_total,"ensemble_detections":e_total,
            "single_metrics":{"mota":float(row_s.get('mota',0)),"idf1":float(row_s.get('idf1',0))},
            "ensemble_metrics":{"mota":float(row_e.get('mota',0)),"idf1":float(row_e.get('idf1',0))},
            "mota_improvement_ppt":round(mota_d,1),"idf1_improvement_ppt":round(idf1_d,1)}
    path=os.path.join(config.OUTPUT_DIR,"real_mot_benchmark_v2.json")
    with open(path,'w') as f: json.dump(result,f,indent=2)
    print(f"\n📁 {path}")

if __name__=="__main__":
    main()
