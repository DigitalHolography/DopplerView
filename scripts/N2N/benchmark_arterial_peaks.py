"""Extract manual arterial waveforms and compare peak detectors without loading videos into RAM."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import ast
import csv
import html

import cv2
import h5py
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.signal import find_peaks, detrend, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from arterial_peaks import detect_arterial_peaks


def existing_methods():
    # Execute only these function definitions: no imports or application startup.
    source = Path(__file__).resolve().parents[2]/'dopplerview/segmentation/pulse_analysis.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('compute_period','get_peaks')]
    namespace = dict(np=np,detrend=detrend,find_peaks=find_peaks,gaussian_filter1d=gaussian_filter1d)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    return namespace


def compare(cache, output, fps):
    existing = existing_methods()
    rows = []
    for path in sorted(cache.glob('*.npz')):
        data = np.load(path); curve = data['signal']
        result = detect_arterial_peaks(curve,fps)
        legacy_smooth = np.convolve(np.pad(curve,(2,2),mode='edge'),np.ones(5)/5,mode='valid')
        legacy,_ = find_peaks(legacy_smooth,distance=15,prominence=.1*np.subtract(*np.percentile(legacy_smooth,[95,5])))
        b,a = butter(4,15/(fps/2),btype='low')
        filtered = filtfilt(b,a,curve)
        period = existing['compute_period'](filtered,fps)
        old = existing['get_peaks'](filtered,period)
        started=time.perf_counter()
        for _ in range(10): detect_arterial_peaks(curve,fps)
        elapsed=(time.perf_counter()-started)/10
        rng=np.random.default_rng(2026)
        stability=[]
        for _ in range(20):
            changed=detect_arterial_peaks(curve+rng.normal(0,.5*result['noise_scale'],len(curve)),fps)
            stability.append(match(result['peaks'],changed['peaks'],round(.08*fps))['f1'])
        variants={}
        for label,kw in [('no_dropout_repair',dict(repair=False)),('no_sequence_selection',dict(sequence=False))]:
            variants[label]=detect_arterial_peaks(curve,fps,**kw)['peaks'].tolist()
        # Spatial sensitivity: median of normalized local arterial tile signals.
        local=data['local']; good=np.std(local,axis=1)>0
        local=local[good]
        if not len(local): local=curve[None]  # Sparse mask: no independent tile diagnostic.
        spatial=np.median((local-local.mean(1,keepdims=True))/local.std(1,keepdims=True),axis=0)
        variants['spatial_median']=detect_arterial_peaks(spatial,fps)['peaks'].tolist()
        row = dict(record=path.stem,frames=len(curve),fps=fps,period_frames=result['period_frames'],
                   old_period=period,peaks=result['peaks'].tolist(),upstrokes=result['upstrokes'].tolist(),
                   legacy_peaks=legacy.tolist(),dopplerview_upstrokes=old.tolist(),
                   artifact_frames=int(result['artifact_mask'].sum()),periodicity=result['periodicity'],warnings=result['warnings'],
                   detector_ms=elapsed*1000,noise_stability_f1=float(np.mean(stability)),
                   variants=variants,peak_near_artifact=result['peak_near_artifact'].tolist(),
                   rejected_candidates=result['rejected_candidates'].tolist())
        rows.append(row)
        folder=output/path.stem;folder.mkdir(exist_ok=True)
        (folder/'peaks.json').write_text(json.dumps(row,indent=2))
        np.savez_compressed(folder/'diagnostics.npz',raw=curve,**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        np.savetxt(folder/'peaks.csv',np.column_stack((result['peaks'],result['peaks']/fps,result['upstrokes'],result['peak_near_artifact'])),
                   delimiter=',',header='original_frame,time_seconds,upstroke_frame,near_dropout',comments='')
        fig = Figure(figsize=(15,10),layout='constrained'); FigureCanvasAgg(fig)
        axes=fig.subplots(3,1)
        t=np.arange(len(curve))/fps
        for ax in axes[:2]:
            ax.plot(t,curve,color='.65',lw=.8,label='Raw manual artery mean')
            ax.plot(t,result['smoothed'],color='#147c91',label='Robust detection smoothing')
            ax.scatter(t[result['peaks']],result['smoothed'][result['peaks']],marker='v',color='#ba312b',label='New intensity maxima',zorder=4)
            ax.scatter(t[result['upstrokes']],result['smoothed'][result['upstrokes']],marker='>',color='#20803c',label='New upstrokes')
            for i in np.flatnonzero(result['artifact_mask']): ax.axvspan(i/fps,(i+1)/fps,color='orange',alpha=.12,lw=0)
            ax.grid(alpha=.2);ax.set_xlabel('Time (s)');ax.set_ylabel('M0 intensity')
        axes[0].scatter(t[legacy],curve[legacy],marker='x',color='purple',label='Old Noise2Time maxima')
        axes[0].legend(ncol=3,fontsize=8); axes[0].set_title(path.stem+' — orange: possible dropouts / narrow spikes')
        axes[1].scatter(t[old],filtered[old],marker='x',color='black',label='Current DopplerView upstrokes')
        axes[1].legend(ncol=3,fontsize=8)
        lag=np.arange(len(curve))/fps
        axes[2].plot(lag,result['acf']); axes[2].set_xlim(.4,2)
        axes[2].axvline(result['period_frames']/fps,color='red',label=f"New period {result['period_frames']} frames")
        axes[2].axvline(period/fps,color='black',ls='--',label=f'Current DopplerView period {period} frames')
        axes[2].set(xlabel='Lag (s)',ylabel='Normalized autocorrelation');axes[2].legend();axes[2].grid(alpha=.2)
        fig.savefig(output/(path.stem+'.png'),dpi=120)
        fig.clear()
        fig=Figure(figsize=(12,4),layout='constrained');FigureCanvasAgg(fig)
        ax1,ax2=fig.subplots(1,2)
        mean=data['mean'];vmin,vmax=np.percentile(mean,[1,99])
        ax1.imshow(mean,cmap='gray',vmin=vmin,vmax=vmax)
        rgba=np.zeros((*mean.shape,4));rgba[data['mask']]=[1,.1,.1,.5]
        ax1.imshow(rgba);ax1.axis('off');ax1.set_title('Manual arterial mask on mean M0')
        for local_curve in local:
            ax2.plot(t,(local_curve-local_curve.mean())/local_curve.std(),color='.75',lw=.5,alpha=.5)
        ax2.plot(t,spatial,color='#147c91',label='Median of normalized tiles')
        ax2.plot(t,(curve-curve.mean())/curve.std(),color='#b43126',label='Whole artery mean')
        ax2.set(xlabel='Time (s)',ylabel='Standardized intensity');ax2.legend(fontsize=8)
        fig.savefig(folder/'mask_and_spatial.png',dpi=110);fig.clear()
        print(path.stem,'new',row['peaks'],'legacy',len(legacy),'DV',len(old),'period',row['period_frames'],period,flush=True)
    (output/'comparison.json').write_text(json.dumps(rows,indent=2))
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['record','new_peak_count','legacy_peak_count','DV_upstroke_count','period_frames','DV_period_frames','noise_stability_f1','detector_ms'])
        for r in rows: writer.writerow([r['record'],len(r['peaks']),len(r['legacy_peaks']),len(r['dopplerview_upstrokes']),r['period_frames'],r['old_period'],r['noise_stability_f1'],r['detector_ms']])
    sources=[Path(__file__),Path(__file__).with_name('arterial_peaks.py'),Path(__file__).resolve().parents[2]/'dopplerview/segmentation/pulse_analysis.py']
    (output/'experiment.json').write_text(json.dumps(dict(fps=fps,frame_axis=0,h5_key='doppler_signal/M0_ff',
        detector_band_hz=[.5,2.5],baseline_DV_band_hz=[.5,2.],synthetic_development_seed=73019,
        synthetic_holdout_seed=73020,perturbation_seed=2026,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2))
    return rows


def match(truth, found, tolerance):
    """One-to-one nearest matching; never count two predictions for one beat."""
    pairs=sorted((abs(int(a)-int(b)),i,j) for i,a in enumerate(truth) for j,b in enumerate(found))
    used_a,used_b,errors=set(),set(),[]
    for error,i,j in pairs:
        if error<=tolerance and i not in used_a and j not in used_b:
            used_a.add(i);used_b.add(j);errors.append(error)
    return dict(tp=len(errors),fp=len(found)-len(errors),fn=len(truth)-len(errors),errors=errors,
                f1=2*len(errors)/(len(truth)+len(found)) if len(truth)+len(found) else 1.)


def synthetic(output,fps,seed=73019,prefix='synthetic'):
    existing=existing_methods();rng=np.random.default_rng(seed)
    rows=[];examples=[]
    kinds=('clean','noise','dropouts','secondary_humps','drift','variable_intervals','missing_beat','positive_spikes')
    for kind in kinds:
        for replicate in range(20):
            n=round(6*fps);t=np.arange(n)/fps
            period=float(rng.uniform(.58,1.2));onsets=[];start=.4
            while start<5.6:
                onsets.append(start)
                start+=period*(rng.uniform(.78,1.22) if kind=='variable_intervals' else 1)
            if kind=='missing_beat': onsets.pop(len(onsets)//2)
            clean=np.zeros(n); truth=[];up=[]
            for onset in onsets:
                z=t-onset
                pulse=np.where(z>=0,(1-np.exp(-np.maximum(z,0)/.025))*np.exp(-np.maximum(z,0)/.16),0)
                if kind=='secondary_humps':pulse+=.33*np.exp(-.5*((z-.30*period)/.035)**2)
                truth.append(int(np.argmax(pulse)));up.append(int(np.argmax(np.gradient(pulse))))
                clean+=pulse
            curve=clean+2+rng.normal(0,.09 if kind=='noise' else .018,n)
            if kind=='drift':curve+=.6*np.sin(2*np.pi*.18*t)+.1*t
            if kind in ('dropouts','positive_spikes'):
                for center in rng.integers(round(.3*fps),n-round(.3*fps),4):
                    curve+=(-1.5 if kind=='dropouts' else 1.5)*np.exp(-.5*((np.arange(n)-center)/rng.uniform(.8,2.5))**2)
            result=detect_arterial_peaks(curve,fps)
            smooth=np.convolve(np.pad(curve,(2,2),mode='edge'),np.ones(5)/5,mode='valid')
            old,_=find_peaks(smooth,distance=15,prominence=.1*np.subtract(*np.percentile(smooth,[95,5])))
            b,a=butter(4,15/(fps/2),btype='low');filtered=filtfilt(b,a,curve)
            dv=existing['get_peaks'](filtered,existing['compute_period'](filtered,fps))
            for method,found,ref in [('new',result['peaks'],truth),('legacy_N2T',old,truth),('DopplerView',dv,up)]:
                score=match(ref,found,round(.08*fps))
                rows.append(dict(case=kind,replicate=replicate,method=method,**score))
            if replicate==0:examples.append((kind,curve,np.array(truth),result['peaks']))
    summary=[]
    for kind in kinds:
        for method in ('new','legacy_N2T','DopplerView'):
            selected=[r for r in rows if r['case']==kind and r['method']==method]
            tp,fp,fn=[sum(r[k] for r in selected) for k in ('tp','fp','fn')]
            errors=[e/fps*1000 for r in selected for e in r['errors']]
            summary.append(dict(case=kind,method=method,tp=tp,fp=fp,fn=fn,f1=2*tp/(2*tp+fp+fn),
                                mean_matched_error_ms=float(np.mean(errors)) if errors else None))
    (output/(prefix+'.json')).write_text(json.dumps(dict(seed=seed,tolerance_seconds=.08,summary=summary,trials=rows),indent=2))
    fig=Figure(figsize=(14,18),layout='constrained');FigureCanvasAgg(fig)
    for ax,(kind,curve,truth,found) in zip(fig.subplots(len(examples),1),examples):
        ax.plot(np.arange(len(curve))/fps,curve,lw=.8,color='.5')
        ax.scatter(truth/fps,curve[truth],marker='o',facecolors='none',edgecolors='green',label='Known maximum')
        ax.scatter(found/fps,curve[found],marker='x',color='red',label='Detected')
        ax.set_title(kind);ax.set_xlabel('Time (s)');ax.legend(fontsize=8,ncol=2)
    fig.savefig(output/(prefix+'_examples.png'),dpi=110);fig.clear()
    return summary


def write_report(output,rows,synth,fps):
    def table(headers,records):
        return '<table><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in records)+'</table>'
    cohort=table(['Recording','New peaks','Old N2T peaks','DV upstrokes','New / DV period (frames)','Noise stability F1','Detector ms','Review flags'],
        [(r['record'],len(r['peaks']),len(r['legacy_peaks']),len(r['dopplerview_upstrokes']),f"{r['period_frames']} / {r['old_period']}",round(r['noise_stability_f1'],3),round(r['detector_ms'],2),'; '.join(r['warnings'])) for r in rows])
    synth_table=table(['Case','Method','TP','FP','FN','F1','Matched timing error ms'],
        [(r['case'],r['method'],r['tp'],r['fp'],r['fn'],round(r['f1'],3),round(r['mean_matched_error_ms'],1) if r['mean_matched_error_ms'] is not None else 'undefined') for r in synth])
    priorities=['251031_ALA_L_1','260626_COY_choroid_6','260622_DUM_L_4','260622_DUM_L_2']
    ordered=sorted(rows,key=lambda r:(priorities.index(r['record']) if r['record'] in priorities else 99,r['record']))
    sections=''
    for r in ordered:
        name=r['record'];open_attr=' open' if name in priorities else ''
        sections+=f'<details{open_attr}><summary>{name}</summary><p>Peaks (zero-based original frames): {r["peaks"]}. Warnings: {html.escape(str(r["warnings"]))}</p><img src="{name}.png"><img src="{name}/mask_and_spatial.png"><p>Ablations / spatial sensitivity: {html.escape(str(r["variants"]))}</p><a href="{name}/peaks.json">Peak metadata</a> · <a href="{name}/peaks.csv">Peak CSV</a> · <a href="{name}/diagnostics.npz">Numerical diagnostics</a></details>'
    text=f'''<!doctype html><html><meta charset="utf-8"><title>Arterial peak detection study</title>
<style>body{{font:16px system-ui;max-width:1400px;margin:30px auto;padding:20px;color:#243747}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ccd6dd;text-align:left}}img{{width:100%}}summary{{font-size:20px;cursor:pointer;padding:16px;background:#edf5f7}}p,li{{line-height:1.6}}.note{{padding:16px;background:#fff1d6}}</style>
<h1>Arterial peak detection: {len(rows)} recordings</h1><p>Effective rate: {fps} Hz (this study: 37000 / 256). Input: doppler_signal/M0_ff (time,y,x), manual retinal artery masks. Actual dataset mask filename: retina_artery_mask.png; both spellings are supported. Source data are read-only.</p>
<p class="note">Exploratory development on this cohort, not independent validation. No ECG or user-annotated beat labels are available. Peak counts and stability are diagnostics, not accuracy scores. Synthetic tests have known labels but do not establish accuracy on real recordings. Orange regions show possible dropouts; interpolation is used only for detection, never to modify video data.</p>
<h2>Method</h2><ol><li>Stream frames in blocks of 16; average pixels inside the handmade arterial mask.</li><li>Flag brief downward deviations from a 180 ms local median using a noise- and amplitude-relative threshold; also flag narrow positive spikes (at most 60 ms) with a stricter threshold. Interpolate flagged samples for detection only.</li><li>Apply Gaussian smoothing (20 ms sigma); subtract a slow Gaussian baseline (600 ms sigma).</li><li>Estimate period from normalized autocorrelation in 0.5–2.5 Hz. Prefer the shortest recurrence reaching 70% of the best correlation and at least 0.25, to avoid choosing every second beat.</li><li>Generate permissive prominence candidates. Choose the strongest sequence with a soft interval penalty. Allow missing beats with a penalty; never insert regularly spaced peaks.</li><li>Refine maxima on the lightly smoothed intensity and separately report preceding maximum-slope upstrokes. Flag weak recurrence, irregular intervals and proximity to artifacts.</li></ol>
<p>Candidate spacing is 0.2 periods; prominence exceeds max(12% robust pulse range, estimated smoothed noise floor). Sequence transition cost is min over m=1,2,3 of 8·log(interval/(m·period))² + 2(m−1). This is a regularity prior, so arrhythmia, large rate changes and motion still need review. No statistical confidence probability is claimed.</p>
<h2>Comparison definitions</h2><p>Old N2T: five-frame moving mean, distance 15 frames, prominence 10% of robust range. Current DopplerView: 15 Hz zero-phase low-pass, existing compute_period and get_peaks definitions loaded directly from repository source. DopplerView returns upstrokes, not intensity maxima; its markers should be compared with green upstroke markers. All methods receive the same manual arterial mean.</p>
<h2>Real-recording diagnostics</h2>{cohort}<p>Noise stability: mean one-to-one peak F1 versus the unperturbed result over 20 perturbations with Gaussian noise SD equal to half the first-difference noise estimate; tolerance 80 ms. This tests sensitivity, not correctness. Spatial sensitivity uses the median of standardized 64×64 arterial-tile curves (at least 40 selected pixels each); it is not the selected production signal. Detector timings exclude HDF5 I/O and plotting.</p>
<h2>Known-signal tests</h2><p>Table below: 160 held-out-seed synthetic signals (seed 73020), evaluated after tuning on a separate seed (73019); eight conditions, 20 trials each, periods sampled between 0.58 and 1.2 s. Each has 6 s duration. This tests new random realizations of the same simulator, not independent physiological generalization. Matching tolerance: 80 ms, one prediction per true beat. N2T/new compared with true maxima; DopplerView compared with true upstrokes. Matched timing error excludes misses and false positives: read it together with FP/FN. Full development and holdout results are in synthetic.json and synthetic_holdout.json.</p>{synth_table}
<details><summary>Synthetic examples (held-out seed)</summary><img src="synthetic_holdout_examples.png"></details>
<h2>Detailed recordings — difficult cases first</h2>{sections}
<h2>Limitations and next validation</h2><p>The same cohort informed development, and the detector favors approximately recurring arterial pulses. Mark suspected peaks manually, especially near orange intervals; establish labels independently before measuring real-data precision/recall. Short recordings have little evidence for period changes. Positive motion spikes, long dropouts and ambiguous broad plateaus can still shift or remove landmarks. Detection repairs do not make affected frames valid training donors.</p></html>'''
    (output/'report.html').write_text(text,encoding='utf-8')


def extract(root, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        files = list(folder.glob('*.h5'))
        masks = [folder/'manual'/name for name in ('retinal_artery_mask.png','retina_artery_mask.png')]
        mask_path = next((p for p in masks if p.exists()), None)
        if len(files) != 1 or mask_path is None:
            rows.append(dict(record=folder.name,error='Need exactly one HDF5 and a manual artery mask'))
            continue
        target = output/(folder.name+'.npz')
        identity = dict(h5=str(files[0]), h5_size=files[0].stat().st_size,
                        h5_mtime_ns=files[0].stat().st_mtime_ns,
                        mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest())
        meta_path = target.with_suffix('.json')
        if target.exists() and meta_path.exists() and json.loads(meta_path.read_text()).get('source') == identity:
            rows.append(json.loads(meta_path.read_text())); continue
        started = time.perf_counter()
        mask_image = cv2.imdecode(np.fromfile(mask_path,np.uint8),cv2.IMREAD_UNCHANGED)
        mask = mask_image != 0 if mask_image.ndim == 2 else np.any(mask_image[...,:3]!=0,axis=2)
        if mask_image.ndim == 3 and mask_image.shape[2] == 4:
            mask &= mask_image[...,3]!=0
        if not mask.any(): raise ValueError(f'Empty mask: {mask_path}')
        with h5py.File(files[0],'r') as h5:
            dataset = h5['doppler_signal/M0_ff']
            # DopplerView saves the in-memory (time,y,x) array without transposition.
            if dataset.ndim != 3 or dataset.shape[1:] != mask.shape:
                raise ValueError(f'Mask/video mismatch: {folder.name}')
            n,h,w = dataset.shape
            curve = np.empty(n)
            # Spatial tiles provide a motion/mask sensitivity diagnostic, not extra training labels.
            tiles = []
            for y in range(0,h,64):
                for x in range(0,w,64):
                    selected = mask[y:y+64,x:x+64]
                    if selected.sum() >= 40: tiles.append((y,x,selected))
            local = np.empty((len(tiles),n))
            mean = np.zeros((h,w),np.float64)
            for start in range(0,n,16):
                block = dataset[start:start+16]
                curve[start:start+len(block)] = np.nanmean(block[:,mask],axis=1,dtype=np.float64)
                mean += np.nansum(block,axis=0,dtype=np.float64)
                for i,(y,x,selected) in enumerate(tiles):
                    local[i,start:start+len(block)] = np.nanmean(block[:,y:y+64,x:x+64][:,selected],axis=1,dtype=np.float64)
            saved = np.asarray(h5['waveform/Retina/pre_arterial_pulse']) if 'waveform/Retina/pre_arterial_pulse' in h5 else np.array([])
            np.savez_compressed(target,signal=curve,local=local,mean=mean/n,mask=mask,saved_auto_signal=saved)
            row = dict(record=folder.name,source=identity,shape=list(dataset.shape),mask_pixels=int(mask.sum()),
                       mask=str(mask_path),chunks=dataset.chunks,extraction_seconds=time.perf_counter()-started)
        meta_path.write_text(json.dumps(row,indent=2))
        rows.append(row)
        print(folder.name, row['shape'],f"{row['extraction_seconds']:.2f}s",flush=True)
    (output/'inventory.json').write_text(json.dumps(rows,indent=2))


def overview(cache, output):
    paths = sorted(cache.glob('*.npz'))
    fig = Figure(figsize=(16,2.4*len(paths)),layout='constrained'); FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(len(paths),1))
    for ax,path in zip(axes,paths):
        curve = np.load(path)['signal']
        ax.plot(curve,lw=.8); ax.set_title(path.stem); ax.set_xlabel('Frame'); ax.grid(alpha=.2)
    fig.savefig(output,dpi=110)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fps',type=float,default=37000/256)
    args = p.parse_args()
    extract(args.dataset,args.output/'cache')
    overview(args.output/'cache',args.output/'overview.png')
    rows=compare(args.output/'cache',args.output,args.fps)
    synthetic(args.output,args.fps)
    synth=synthetic(args.output,args.fps,seed=73020,prefix='synthetic_holdout')
    write_report(args.output,rows,synth,args.fps)
