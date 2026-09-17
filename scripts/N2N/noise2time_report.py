"""Regional Noise2Time evaluation and a self-contained local HTML report.

All numerical analysis uses unclipped arrays. Display clipping is explicitly
separate from analysis. No model, training, or network access is required here.
"""
from pathlib import Path
import csv
import html
import json

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import colormaps

REGIONS = ("retinal_artery", "retinal_vein", "choroidal")
LABELS = {"retinal_artery":"Retinal arteries", "retinal_vein":"Retinal veins",
          "choroidal":"Choroidal vessels", "background":"Background"}
COLORS = {"retinal_artery":"#e34a33", "retinal_vein":"#3182bd",
          "choroidal":"#a661c2", "background":"#31a354"}
EPS = 1e-12


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def derive_background(raw, roi, radius):
    """Dilate original retinal masks only, then negate their union with choroid."""
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("Background dilation radius must be a nonnegative integer")
    retinal = raw["retinal_artery"] | raw["retinal_vein"]
    if radius:
        yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
        kernel = (xx*xx + yy*yy <= radius*radius).astype(np.uint8)
        retinal = cv2.dilate(retinal.astype(np.uint8), kernel,
                             borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return roi & ~(retinal | raw["choroidal"])


def exclusive_masks(raw, roi):
    """Compute exclusions simultaneously from ORIGINAL masks, never in-place."""
    clipped = {name:np.asarray(mask, bool) & roi for name,mask in raw.items()}
    vessel_union = np.logical_or.reduce([clipped[name] for name in REGIONS])
    cleaned, counts = {}, {}
    for name in REGIONS:
        others = np.logical_or.reduce([clipped[other] for other in REGIONS if other != name])
        cleaned[name] = clipped[name] & ~others
    cleaned["background"] = clipped["background"] & ~vessel_union
    for name in (*REGIONS, "background"):
        counts[name] = dict(input_pixels=int(raw[name].sum()),
                            outside_roi_pixels=int((raw[name] & ~roi).sum()),
                            excluded_overlap_pixels=int((clipped[name] & ~cleaned[name]).sum()),
                            evaluated_pixels=int(cleaned[name].sum()))
        if not cleaned[name].any():
            raise ValueError(f"{name} has no pixels left after ROI/overlap exclusion: {counts[name]}")
    excluded = vessel_union & ~np.logical_or.reduce([cleaned[name] for name in REGIONS])
    return cleaned, counts, excluded


def ratio(a, b):
    return float(a/b) if abs(b) > EPS else None


def correlation(a, b):
    if min(np.std(a), np.std(b)) <= EPS:
        return None
    return float(np.clip(np.corrcoef(a,b)[0,1], -1, 1))


def sinusoid(curve, fps, frequency):
    t = np.arange(len(curve))/fps
    design = np.column_stack((np.ones(len(t)),np.cos(2*np.pi*frequency*t),np.sin(2*np.pi*frequency*t)))
    fit = np.linalg.lstsq(design,curve,rcond=None)[0]
    return float(np.hypot(fit[1],fit[2]))


def waveform_metrics(before, after, fps, frequency, max_lag_seconds):
    mean_before, mean_after = float(before.mean()), float(after.mean())
    a, b = sinusoid(before,fps,frequency), sinusoid(after,fps,frequency)
    limit = min(int(round(max_lag_seconds*fps)),len(before)//4)
    candidates = []
    for lag in range(-limit,limit+1):
        x,y = (before[:-lag],after[lag:]) if lag>0 else ((before[-lag:],after[:lag]) if lag<0 else (before,after))
        value = correlation(x,y)
        if value is not None:
            candidates.append((value,lag))
    best = max(candidates,key=lambda item:(item[0],-abs(item[1]))) if candidates else None
    harmonics = []
    for order in (1,2,3):
        f = order*frequency
        if f >= fps/2:
            continue
        old,new = sinusoid(before,fps,f),sinusoid(after,fps,f)
        harmonics.append(dict(order=order,frequency_hz=f,amplitude_original=old,
                              amplitude_denoised=new,amplitude_ratio=ratio(new,old)))
    return dict(temporal_correlation=correlation(before,after),
                vessel_mean_original=mean_before,vessel_mean_denoised=mean_after,
                mean_change=mean_after-mean_before,
                mean_change_percent=None if abs(mean_before)<=EPS else 100*(mean_after/mean_before-1),
                cardiac_frequency_hz=frequency,amplitude_original=a,amplitude_denoised=b,
                amplitude_ratio=ratio(b,a),
                residual_cardiac_amplitude=sinusoid(before-after,fps,frequency),
                lag_frames=best[1] if best else None,
                lag_ms=1000*best[1]/fps if best else None,
                lag_corrected_correlation=best[0] if best else None,
                lag_limit_seconds=max_lag_seconds,
                harmonics=harmonics)


def local_regions(mask, size, count):
    """Non-overlapping grid patches, ordered by original mask occupancy."""
    tiles = []
    for y in range(0,mask.shape[0],size):
        for x in range(0,mask.shape[1],size):
            n = int(mask[y:y+size,x:x+size].sum())
            if n:
                tiles.append((n,y,x))
    result = []
    for n,y,x in sorted(tiles,key=lambda z:(-z[0],z[1],z[2]))[:count]:
        selected = np.zeros_like(mask)
        selected[y:y+size,x:x+size] = mask[y:y+size,x:x+size]
        result.append((selected,dict(x=x,y=y,width=min(size,mask.shape[1]-x),
                                     height=min(size,mask.shape[0]-y),pixels=n)))
    return result


def phase_bins(n, copied_prefix, peaks):
    """Four fractional-cycle bins; discard intervals outside complete peak pairs."""
    labels = np.full(n,-1,dtype=np.int8)
    for start,end in zip(peaks[:-1],peaks[1:]):
        if end <= start:
            continue
        positions = np.arange(max(start,copied_prefix),min(end,copied_prefix+n))
        if len(positions):
            labels[positions-copied_prefix] = np.minimum(3,4*(positions-start)//(end-start))
    return labels


def accumulate(original, denoised, masks, labels, chunk=8):
    """Bounded frame buffers; accumulate pixel statistics in float64."""
    n,h,w = original.shape
    total = np.zeros((2,h,w),np.float64)
    squares = np.zeros_like(total)
    phases = np.zeros((4,h,w),np.float64)
    phase_count = np.zeros(4,np.int64)
    curves = {name:np.empty((2,n),np.float64) for name in masks}
    for start in range(0,n,chunk):
        stop = min(start+chunk,n)
        a = np.asarray(original[start:stop],dtype=np.float64)
        b = np.asarray(denoised[start:stop],dtype=np.float64)
        for i,block in enumerate((a,b)):
            # Merge centered batch moments to avoid cancellation for flat pixels.
            block_mean = block.mean(axis=0)
            delta = block_mean-total[i]
            squares[i] += np.square(block-block_mean).sum(axis=0) + delta**2*start*len(block)/stop
            total[i] += delta*len(block)/stop
            for name,mask in masks.items():
                curves[name][i,start:stop] = block[:,mask].mean(axis=1)
        for phase in range(4):
            selected = labels[start:stop] == phase
            if selected.any():
                phases[phase] += (a[selected]-b[selected]).sum(axis=0)
                phase_count[phase] += selected.sum()
    mean = total
    std = np.sqrt(np.maximum(0,squares/n))
    for phase in range(4):
        if phase_count[phase]:
            phases[phase] /= phase_count[phase]
    return mean,std,curves,phases,phase_count


def new_figure(*args,**kwargs):
    figure = Figure(*args,constrained_layout=True,**kwargs)
    FigureCanvasAgg(figure)
    return figure


def save_figure(figure, path):
    figure.savefig(path,dpi=130)
    figure.clear()


def limit(values):
    return max(float(np.percentile(np.abs(values),99)),1e-6)


def spectrum(curve,fps):
    window = np.hanning(len(curve))
    return np.fft.rfftfreq(len(curve),1/fps), 2*np.abs(np.fft.rfft((curve-curve.mean())*window))/window.sum()


def save_waveforms(folder,name,curves,fps,frequency,first,offset):
    before,after = curves
    t = np.arange(len(before))/fps
    table = np.column_stack((np.arange(len(t))+first,np.arange(len(t))+first+offset,
                            t,before,after,before-after))
    np.savetxt(folder/f"{name}_waveform.csv",table,delimiter=",",comments="",
               header="prepared_frame,original_frame,seconds_since_scored_start,original,denoised,residual")
    figure = new_figure(figsize=(11,8))
    axes = figure.subplots(3,1)
    for ax in axes[:2]:
        ax.plot(t,before,label="Original",color="#777777",lw=1)
        ax.plot(t,after,label="Denoised",color="#007c91",lw=1)
        ax.set_ylabel("Mean intensity")
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[0].set_title(LABELS.get(name,name))
    axes[1].set_xlim(0,min(t[-1],3/frequency))
    axes[1].set_title("First three nominal cardiac cycles; same intensity scale")
    axes[1].set_ylim(axes[0].get_ylim())
    axes[2].plot(t,before-after,color="#b34b35",lw=1)
    axes[2].axhline(0,color="black",lw=.5)
    axes[2].set_ylabel("Original - denoised")
    axes[2].set_xlabel("Time since first scored frame (s)")
    save_figure(figure,folder/f"{name}_waveform.png")
    figure = new_figure(figsize=(10,4))
    ax = figure.subplots()
    for curve,label,color in ((before,"Original","#777777"),(after,"Denoised","#007c91"),
                              (before-after,"Residual","#b34b35")):
        f,a = spectrum(curve,fps)
        ax.plot(f,a,label=label,color=color,lw=1)
    for k in (1,2,3):
        if k*frequency < fps/2:
            ax.axvline(k*frequency,color="#333333",ls="--",alpha=.4)
    ax.set(xlim=(0,fps/2),xlabel="Frequency (Hz)",ylabel="Amplitude (Hann window)",
           title=f"{LABELS.get(name,name)}: spectrum; dashed lines = selected f0 and harmonics")
    ax.legend(); ax.grid(alpha=.2)
    save_figure(figure,folder/f"{name}_spectrum.png")


def profile_coordinates(mask, size):
    y,x = np.nonzero(mask)
    center = np.array([x.mean(),y.mean()])
    normal = np.array([1.,0.])
    if len(x)>2:
        _,vectors = np.linalg.eigh(np.cov(np.stack((x,y))))
        normal = vectors[:,0]  # Perpendicular to major axis of selected mask patch.
    distance = np.arange(-size//2,size//2+1,dtype=float)
    points = center[:,None]+normal[:,None]*distance
    keep = (points[0]>=0)&(points[0]<=mask.shape[1]-1)&(points[1]>=0)&(points[1]<=mask.shape[0]-1)
    return distance[keep],points[:,keep]


def save_profile(folder,name,local_mask,original,denoised,fps,size,selected_frame,override=None):
    if override is None:
        distance,points = profile_coordinates(local_mask,size)
        origin = "Automatic perpendicular to mask-patch major axis; inspect placement"
    else:
        start,end = np.asarray(override["start"],float),np.asarray(override["end"],float)
        if start.shape != (2,) or end.shape != (2,) or not np.isfinite([start,end]).all():
            raise ValueError("Profile start/end must be finite [x,y] coordinates")
        length = float(np.linalg.norm(end-start))
        if length < 1 or length > 2*max(local_mask.shape):
            raise ValueError("Invalid profile length")
        points = np.linspace(start,end,int(np.ceil(length))+1).T
        if np.any(points<0) or np.any(points[0]>local_mask.shape[1]-1) or np.any(points[1]>local_mask.shape[0]-1):
            raise ValueError("Profile lies outside image")
        distance = np.linspace(0,length,points.shape[1])
        origin = "User-supplied line"
    samples = []
    for frames in (original,denoised):
        samples.append(np.stack([map_coordinates(frame,[points[1],points[0]],order=1,mode="nearest") for frame in frames]))
    a,b = samples
    np.savez(folder/f"{name}_profile.npz",distance_pixels=distance,x=points[0],y=points[1],original=a,denoised=b)
    figure = new_figure(figsize=(12,8))
    axes = figure.subplots(2,3)
    axes[0,0].imshow(original[selected_frame],cmap="gray",vmin=0,vmax=1)
    axes[0,0].plot(points[0],points[1],color="#ffb000")
    axes[0,0].set_title(f"{LABELS[name]}: profile location")
    axes[0,0].axis("off")
    for ax,curves,title in ((axes[0,1],(a[selected_frame],b[selected_frame]),"Matched original-peak frame"),
                             (axes[0,2],(a.mean(0),b.mean(0)),"Temporal mean (structural reference)")):
        ax.plot(distance,curves[0],color="#777777",label="Original")
        ax.plot(distance,curves[1],color="#007c91",label="Denoised")
        ax.set(title=title,xlabel="Distance (pixels)",ylabel="Intensity")
        ax.legend()
    extent = [distance[0],distance[-1],(len(a)-1)/fps,0]
    for ax,values,title in ((axes[1,0],a,"Original"),(axes[1,1],b,"Denoised")):
        im = ax.imshow(values,aspect="auto",extent=extent,cmap="gray",vmin=0,vmax=1)
        ax.set(title=title,xlabel="Distance (pixels)",ylabel="Time (s)")
        figure.colorbar(im,ax=ax)
    scale = limit(a-b)
    im = axes[1,2].imshow(a-b,aspect="auto",extent=extent,cmap="RdBu_r",vmin=-scale,vmax=scale)
    axes[1,2].set(title="Signed residual",xlabel="Distance (pixels)",ylabel="Time (s)")
    figure.colorbar(im,ax=axes[1,2])
    figure.suptitle(origin)
    save_figure(figure,folder/f"{name}_profile.png")
    return dict(start=points[:,0].tolist(),end=points[:,-1].tolist(),placement=origin,
                interpolation="bilinear, same coordinates for both videos",selected_scored_frame=selected_frame)


def write_comparison(path,original,denoised,fps,scale):
    h,w = original.shape[1:]
    # Pad rather than crop if externally supplied arrays have odd dimensions.
    out_h,out_w = h+h%2,3*w+(3*w)%2
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),fps,(out_w,out_h),True)
    try:
        if not writer.isOpened():
            raise RuntimeError("Cannot create comparison AVI")
        cmap = colormaps["RdBu_r"]
        for before,after in zip(original,denoised):
            gray = [cv2.cvtColor(np.rint(np.clip(frame,0,1)*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
                    for frame in (before,after)]
            residual = (cmap(np.clip((before-after)/(2*scale)+.5,0,1))[...,:3]*255).astype(np.uint8)[...,::-1]
            canvas = np.zeros((out_h,out_w,3),np.uint8)
            canvas[:h,:3*w] = np.concatenate((*gray,residual),axis=1)
            for index,label in enumerate(("Original","Denoised","Original - denoised")):
                cv2.putText(canvas,label,(index*w+5,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()


def build_report(record,restored,metadata,raw_masks,provenance,args,output):
    """Write into a new staging folder; caller publishes it on successful return."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    plots = output/"plots"; plots.mkdir()
    mask_dir = output/"masks"; mask_dir.mkdir()
    masks,counts,excluded = exclusive_masks(raw_masks,record.roi)
    for name,mask in masks.items():
        np.save(mask_dir/f"{name}.npy",mask)
        ok,encoded = cv2.imencode(".png",mask.astype(np.uint8)*255)
        if not ok: raise RuntimeError("Mask PNG encoding failed")
        encoded.tofile(mask_dir/f"{name}.png")
    first = int(metadata["copied_prefix"])
    if first < 0 or first >= len(record.frames):
        raise ValueError("Invalid copied prefix")
    original,denoised = record.frames[first:],restored[first:]
    n = len(original); fps = float(record.metadata["fps"])
    if n<4 or not np.isfinite(fps) or not 0<args.min_hz<args.max_hz<fps/2:
        raise ValueError("Need >=4 scored frames and 0 < min_hz < max_hz < Nyquist")
    if args.local_size<3 or args.local_count<1 or not np.isfinite(args.max_lag_seconds) or args.max_lag_seconds<0:
        raise ValueError("Invalid local-region size/count or lag bound")
    locals_by_region = {name:local_regions(masks[name],args.local_size,args.local_count) for name in REGIONS}
    all_masks = dict(masks)
    for name,items in locals_by_region.items():
        for i,(mask,_) in enumerate(items):
            all_masks[f"{name}_local_{i+1}"] = mask
    labels = phase_bins(n,first,record.metadata.get("peaks",[]))
    print("Computing regional waveforms and spatial statistics...",flush=True)
    mean,std,curves,phases,phase_count = accumulate(original,denoised,all_masks,labels)
    reference = sum(curves[name][0]*masks[name].sum() for name in REGIONS)/sum(masks[name].sum() for name in REGIONS)
    freq = np.fft.rfftfreq(n,1/fps)
    candidates = np.flatnonzero((freq>=args.min_hz)&(freq<=args.max_hz))
    if args.cardiac_hz is not None:
        frequency = float(args.cardiac_hz)
        if not np.isfinite(frequency) or not 0<frequency<fps/2:
            raise ValueError("--cardiac-hz must be finite, positive and below Nyquist")
        frequency_source = "User-supplied"
    else:
        if not len(candidates) or reference.std()<=EPS:
            raise ValueError("Cannot estimate cardiac frequency; supply --cardiac-hz")
        frequency = float(freq[candidates[np.argmax(np.abs(np.fft.rfft(reference-reference.mean()))[candidates])]])
        frequency_source = "Largest original pooled-vessel FFT component in search band; not independently verified"
    bg = [float(s[masks["background"]].mean()) for s in std]
    background = dict(pixels=int(masks["background"].sum()),background_std_original=bg[0],
                      background_std_denoised=bg[1],NRR=1-bg[1]/bg[0] if bg[0]>EPS else None)
    results = {}
    for name in REGIONS:
        print(f"Creating {LABELS[name]} metrics and plots...",flush=True)
        before,after = curves[name]
        metrics = waveform_metrics(before,after,fps,frequency,args.max_lag_seconds)
        metrics.update(pixels=int(masks[name].sum()),
                       local_regions=[],**{k:v for k,v in background.items() if k!="pixels"})
        for key,frame in (("original",mean[0]),("denoised",mean[1])):
            metrics[f"vessel_background_contrast_{key}"] = float(frame[masks[name]].mean()-frame[masks["background"]].mean())
        for i,(_,bounds) in enumerate(locals_by_region[name]):
            key = f"{name}_local_{i+1}"
            local = waveform_metrics(*curves[key],fps,frequency,args.max_lag_seconds)
            metrics["local_regions"].append(dict(name=key,bounds=bounds,**local))
        results[name] = metrics
        save_waveforms(plots,name,curves[name],fps,frequency,first,record.metadata["first_original_frame"])
        figure = new_figure(figsize=(11,3*len(locals_by_region[name])))
        axes = np.atleast_1d(figure.subplots(len(locals_by_region[name]),1))
        for i,ax in enumerate(axes):
            key = f"{name}_local_{i+1}"
            for curve,label,color in zip(curves[key],("Original","Denoised"),("#777777","#007c91")):
                ax.plot(np.arange(n)/fps,curve,label=label,color=color,lw=1)
            ax.set(title=f"Local {i+1}: {locals_by_region[name][i][1]}",xlabel="Time (s)",ylabel="Mean intensity")
            ax.legend()
            np.savetxt(plots/f"{key}.csv",np.column_stack((np.arange(n)/fps,*curves[key])),delimiter=",",
                       header="seconds_since_scored_start,original,denoised",comments="")
        save_figure(figure,plots/f"{name}_local.png")
    # Masks and automatic local patches, using a shared background image.
    figure = new_figure(figsize=(12,8)); axes = figure.subplots(2,3)
    for ax,name in zip(axes.flat,(*REGIONS,"background")):
        ax.imshow(mean[0],cmap="gray",vmin=0,vmax=1)
        overlay = np.zeros((*record.roi.shape,4))
        from matplotlib.colors import to_rgba
        overlay[masks[name]] = to_rgba(COLORS[name],.6)
        ax.imshow(overlay)
        for i,(_,bounds) in enumerate(locals_by_region.get(name,[])):
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((bounds["x"],bounds["y"]),bounds["width"],bounds["height"],fill=False,edgecolor="yellow"))
            ax.text(bounds["x"],bounds["y"],str(i+1),color="yellow")
        ax.set_title(f"{LABELS[name]}: {masks[name].sum()} pixels"); ax.axis("off")
    axes[1,1].imshow(excluded,cmap="gray",vmin=0,vmax=1)
    axes[1,1].set_title("Ambiguous vessel pixels excluded from all groups"); axes[1,1].axis("off")
    axes[1,2].axis("off")
    axes[1,2].text(0,1,"Automatic local regions:\nhighest mask occupancy on a grid.\nNot a random sample of vessels.\n\nInspect mask alignment before\ninterpreting the metrics.",va="top")
    save_figure(figure,plots/"masks.png")
    # Spatial metrics use full time series; color limits are for display only.
    mean_residual = mean[0]-mean[1]
    reduction = np.full(record.roi.shape,np.nan)
    usable = record.roi & (std[0]>EPS)
    reduction[usable] = 1-std[1][usable]/std[0][usable]
    np.savez(output/"spatial_maps.npz",mean_original=mean[0],mean_denoised=mean[1],mean_residual=mean_residual,
             std_original=std[0],std_denoised=std[1],std_reduction= reduction,
             phase_residual=phases,phase_frame_counts=phase_count)
    figure = new_figure(figsize=(12,8)); axes = figure.subplots(2,3)
    residual_scale = limit(mean_residual[record.roi]); std_max = max(float(std[:,record.roi].max()),1e-6)
    for ax,values,title,cmap,low,high in zip(axes.flat,
            (mean[0],mean[1],mean_residual,std[0],std[1],reduction),
            ("Original mean","Denoised mean","Mean signed residual","Original temporal SD","Denoised temporal SD","SD reduction (display clipped [-1,1])"),
            ("gray","gray","RdBu_r","magma","magma","RdBu"),
            (0,0,-residual_scale,0,0,-1),(1,1,residual_scale,std_max,std_max,1)):
        im = ax.imshow(np.ma.masked_where(~record.roi,values),cmap=cmap,vmin=low,vmax=high)
        ax.set_title(title); ax.axis("off"); figure.colorbar(im,ax=ax,shrink=.7)
    save_figure(figure,plots/"spatial_maps.png")
    phase_scale = limit(phases[:,record.roi])
    figure = new_figure(figsize=(12,4)); axes = figure.subplots(1,4)
    for i,ax in enumerate(axes):
        if phase_count[i]:
            im = ax.imshow(np.ma.masked_where(~record.roi,phases[i]),cmap="RdBu_r",vmin=-phase_scale,vmax=phase_scale)
            figure.colorbar(im,ax=ax,shrink=.65)
        else:
            ax.text(.5,.5,"No complete-cycle frames",ha="center",transform=ax.transAxes)
        ax.set_title(f"Cycle phase {i/4:.2f}-{(i+1)/4:.2f}\nn={phase_count[i]}"); ax.axis("off")
    figure.suptitle("Mean signed residual by fractional peak-to-peak phase; common symmetric scale")
    save_figure(figure,plots/"phase_residuals.png")
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8")) if args.profiles else {}
    if set(profiles)-set(REGIONS): raise ValueError("Unknown region in profiles JSON")
    for name in REGIONS:
        selected = int(np.argmax(curves[name][0]))
        results[name]["profile"] = save_profile(plots,name,locals_by_region[name][0][0],original,denoised,
                                               fps,args.local_size,selected,profiles.get(name))
    sampled = np.linspace(0,n-1,min(20,n),dtype=int)
    video_scale = limit((np.asarray(original[sampled])-np.asarray(denoised[sampled]))[:,record.roi])
    print("Writing synchronized comparison video...",flush=True)
    write_comparison(output/"comparison.avi",original,denoised,fps,video_scale)
    result = dict(schema="noise2time.regional.v1",record=record.name,frames_scored=n,excluded_prefix=first,
                  fps=fps,first_original_scored_frame=int(record.metadata["first_original_frame"]+first),
                  cardiac_frequency_hz=frequency,frequency_source=frequency_source,
                  frequency_resolution_hz=fps/n,background=background,regions=results,mask_counts=counts,
                  mask_sources=provenance,denoised_provenance=metadata,
                  phase_frame_counts=phase_count.tolist(),
                  phase_frames_excluded=int((labels<0).sum()),comparison_residual_display_limit=video_scale,
                  mask_policy="Simultaneous original-mask exclusions; vessel groups mutually exclusive; background = ROI & ~(dilated original retinal union | original choroidal); choroidal not dilated",
                  lag_convention="Positive lag means denoised waveform is delayed; integer-frame search within configured bound",
                  local_selection="Highest-occupancy disjoint mask grid tiles; inspect and do not infer small-vessel performance from these alone",
                  caveat="No clean reference. Excluding mask overlaps does not unmix retinal/choroidal signals. Metrics describe changes, not denoising accuracy.")
    save_json(output/"metrics.json",result)
    flat_keys = ("pixels","NRR","temporal_correlation","amplitude_ratio","mean_change_percent","lag_ms",
                 "vessel_background_contrast_original","vessel_background_contrast_denoised")
    with (output/"metrics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=("region",*flat_keys)); writer.writeheader()
        for name,row in results.items(): writer.writerow(dict(region=name,**{key:row[key] for key in flat_keys}))
    write_html(output,result)
    return result


def write_html(output,result):
    esc = html.escape
    def number(value): return "undefined" if value is None else f"{value:.5g}"
    rows = "".join("<tr><th>"+esc(LABELS[name])+"</th>"+"".join(f"<td>{number(row[key])}</td>" for key in
                   ("pixels","temporal_correlation","amplitude_ratio","mean_change_percent","lag_ms"))+"</tr>"
                   for name,row in result["regions"].items())
    counts = "".join(f"<tr><th>{esc(LABELS[name])}</th>"+"".join(f"<td>{row[key]}</td>" for key in
                     ("input_pixels","outside_roi_pixels","excluded_overlap_pixels","evaluated_pixels"))+"</tr>"
                     for name,row in result["mask_counts"].items())
    sections = []
    for name in REGIONS:
        sections.append(f"<h2>{LABELS[name]}</h2>"+
                        "".join(f'<img src="plots/{name}_{kind}.png" alt="{LABELS[name]} {kind}">' for kind in
                                ("waveform","spectrum","local","profile")))
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Noise2Time: {esc(result['record'])}</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1150px;margin:32px auto;padding:0 20px;color:#243747}}h1,h2{{color:#126782}}img{{width:100%;margin:12px 0}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{padding:10px;border-bottom:1px solid #ccd6dd;text-align:left}}.note{{background:#edf5f7;padding:16px;line-height:1.6}}p{{line-height:1.6}}</style>
<h1>Noise2Time evaluation: {esc(result['record'])}</h1>
<p>{result['frames_scored']} scored frames; {result['excluded_prefix']} copied prefix frames excluded. FPS: {number(result['fps'])}.
Selected frequency: {number(result['cardiac_frequency_hz'])} Hz. Resolution: {number(result['frequency_resolution_hz'])} Hz.<br>{esc(result['frequency_source'])}.</p>
<div class="note">{esc(result['caveat'])}<br>Background NRR: {number(result['background']['NRR'])} (fractional decrease in average pixel temporal SD). This background metric is shared by all three vessel groups.</div>
<table><tr><th>Region</th><th>Pixels</th><th>Correlation</th><th>Amplitude ratio</th><th>Mean change %</th><th>Lag ms</th></tr>{rows}</table>
<p>Amplitude ratios compare the same selected frequency. Positive lag means delayed denoised output; check zero-lag correlation as well. Undefined values are not perfect scores. Harmonic metrics and local measurements are in <a href="metrics.json">metrics.json</a>; summary: <a href="metrics.csv">metrics.csv</a>.</p>
<h2>Mask selection and excluded overlaps</h2><table><tr><th>Mask</th><th>Input pixels</th><th>Outside ROI</th><th>Overlap removed</th><th>Used</th></tr>{counts}</table>
<p>Retinal/choroidal and artery/vein overlaps are removed simultaneously from both vessel groups. Background is the ROI-restricted binary negation of the union of dilated original retinal masks and the original choroidal mask. Retinal disk radius: {result['mask_sources']['background']['retinal_dilation_radius_pixels']} pixels; choroidal mask is not dilated. Cleaned masks are saved in masks/. Automatic local patches favor high mask occupancy, not necessarily faint vessels.</p>
<img src="plots/masks.png" alt="Cleaned masks, local regions, excluded overlaps">
<h2>Spatial changes and phase-resolved residuals</h2><img src="plots/spatial_maps.png" alt="Spatial change maps"><img src="plots/phase_residuals.png" alt="Residuals across cardiac phase">
<p>Phase bins use fractional intervals between preprocessing brightness peaks; incomplete intervals are excluded. These are peak-to-peak phase labels, not independently measured systole/diastole. Zero mean residual does not exclude removal of pulsation. Spatial arrays: <a href="spatial_maps.npz">spatial_maps.npz</a>.</p>
<p><a href="comparison.avi">Open synchronized original / denoised / signed-residual video</a> (AVI requires a compatible player). First two panels use [0,1]; residuals use a fixed symmetric scale of ±{number(result['comparison_residual_display_limit'])}, estimated from sampled frames. Display clipping never affects metrics.</p>
{''.join(sections)}
<h2>Interpretation limits</h2><p>Inspect local curves, residual structures and profile placement before drawing conclusions. Automatic profile orientation is approximate; replace it with manually selected perpendicular lines using --profiles. Original noisy observations and temporal averages are not clean ground truth. No uncertainty estimates or independent generalization test are implied by this report.</p></html>'''
    (output/"report.html").write_text(text,encoding="utf-8")
