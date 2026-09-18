"""Run six one-factor-at-a-time experiments, then compare curves and reports.

Training uses existing preparations once, without copying the video caches.
An optional, separate evaluation dataset is prepared and scored only after training.
"""
import argparse
from dataclasses import asdict
import csv
import html
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import noise2time as api


def variants(base, validation_video):
    """Every alternative changes one conceptual factor relative to baseline."""
    if base.input_mode != 'patched' or not base.convlstm or not base.brightness_correction:
        raise ValueError('Baseline must use patched input, ConvLSTM and brightness correction')
    if base.split_mode != 'mixed' or base.validation_records:
        raise ValueError('Baseline must use mixed validation with no validation_records')
    changes = dict(baseline={}, history_only={'input_mode':'history_only'},
                   temporal_split={'split_mode':'temporal'},
                   video_validation={'split_mode':'record','validation_records':[validation_video]},
                   no_convlstm={'convlstm':False}, no_brightness={'brightness_correction':False})
    return {name: dict(asdict(base), **change) for name,change in changes.items()}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict): result.update(flatten(item,name))
        elif isinstance(item, (int,float)) or item is None: result[name] = item
    return result


def comparison(output, plan):
    """Rebuild an index and one comparison plot per logged numerical metric."""
    plots = output/'comparison'; plots.mkdir(exist_ok=True)
    all_rows, status_rows, final_rows = [], [], []
    for name in plan['variants']:
        folder = output/name
        history = folder/'runs/metrics.jsonl'
        if history.exists():
            for line in history.read_text().splitlines():
                try: row = json.loads(line)
                except json.JSONDecodeError: continue
                all_rows.append(dict(strategy=name, **flatten(row)))
        status = read_json(folder/'status.json') if (folder/'status.json').exists() else {'status':'pending'}
        links = []
        for report in sorted(folder.glob('reports/*/*/report.html')):
            relative = report.relative_to(output).as_posix()
            links.append(f'<a href="{html.escape(relative)}">{html.escape(str(report.parent.relative_to(folder/"reports")))}</a>')
            metrics=read_json(report.parent/'metrics.json')
            final_rows.append(dict(strategy=name,group=report.parent.parent.name,record=report.parent.name,
                                   **flatten(dict(background=metrics['background'],regions=metrics['regions']))))
        if (folder/'runs/metrics.png').exists():
            links.insert(0,f'<a href="{name}/runs/metrics.png">Training curves</a>')
        status_rows.append(f'<tr><td>{name}</td><td>{html.escape(status["status"])}</td><td>{" | ".join(links)}</td><td>{html.escape(status.get("error",""))}</td></tr>')
    keys = sorted({key for row in all_rows for key in row} - {'strategy','epoch'})
    final_keys=sorted({key for row in final_rows for key in row}-{'strategy','group','record'})
    with (output/'final_comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['strategy','group','record']+final_keys)
        writer.writeheader();writer.writerows(final_rows)
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['strategy','epoch']+keys);writer.writeheader();writer.writerows(all_rows)
    figures=[]
    for index,key in enumerate(keys):
        if key.endswith('frames_scored'): continue
        fig=Figure(figsize=(10,4),layout='constrained');FigureCanvasAgg(fig);ax=fig.subplots()
        for name in plan['variants']:
            rows=[row for row in all_rows if row['strategy']==name and row.get(key) is not None]
            if rows: ax.plot([r['epoch'] for r in rows],[r[key] for r in rows],'.-',label=name)
        ax.set(title=key, xlabel='Epoch');ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=8)
        else:
            ax.text(.5,.5,'Metric undefined for these data (see CSV null values)',
                    ha='center',va='center',transform=ax.transAxes)
        filename=f'metric_{index:03d}.png';fig.savefig(plots/filename,dpi=120);fig.clear()
        figures.append(f'<details><summary>{html.escape(key)}</summary><img loading="lazy" src="comparison/{filename}"></details>')
    content='''<!doctype html><meta charset="utf-8"><title>Noise2Time benchmark</title>
<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}img{max-width:100%}summary{padding:12px;cursor:pointer}code{background:#eee}</style>
<h1>Noise2Time: one-factor-at-a-time benchmark</h1>
<p>Each alternative changes one factor from baseline. Epoch diagnostics use only the first development video;
they are not independent test scores. Separate evaluation videos never select checkpoints. Reports use best.pt.</p>
<p>Lower background variability alone does not establish accuracy. Inspect vessel means, waveform preservation,
and regional reports. Validation losses across different splits use different data and are not directly comparable.</p>
<p>Temporal splitting isolates target/history/donor frames; peak timing and brightness smoothing are recomputed
independently inside each partition. The fixed prepared intensity scale and spatial masks are shared calibration.</p>
<p><a href="comparison.csv">All epoch metrics (CSV)</a> | <a href="final_comparison.csv">Final report metrics (CSV)</a> | <a href="plan.json">Exact experiment plan</a></p>
<table><tr><th>Strategy</th><th>Status</th><th>Reports</th><th>Error</th></tr>'''+''.join(status_rows)+'</table><h2>Compare every logged metric</h2>'+''.join(figures)
    temporary=output/'index.html.tmp';temporary.write_text(content,encoding='utf-8');temporary.replace(output/'index.html')


def run_child(command, log):
    """Run serially on the selected device; preserve stdout/stderr even on failure."""
    print('Running:', ' '.join(map(str,command)), flush=True)
    with log.open('a',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,*map(str,command)],stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command exited {result.returncode}; see {log}')


def prepare_evaluation(dataset, output, reference, api):
    workflow=api.load_sibling('dataset_workflow')
    target=output/'evaluation_data'
    args=['prepare','--input',str(dataset),'--output',str(target),'--skip-existing',
          '--fps',str(reference['fps'])]
    for key,value in reference['circle'].items(): args += ['--'+key,str(value)]
    for key in ('peak_min_hz','peak_max_hz'): args += ['--'+key.replace('_','-'),str(reference[key])]
    if reference['input_mode']=='avi': args.append('--avi')
    if api.main(args): raise ValueError('Evaluation preparation failed; inspect evaluation_data/preparation_summary.json')
    return [target/'prepared'/folder.name for folder in workflow.measurements(dataset)]


def final_report(record, folder, group, device):
    workflow=api.load_sibling('dataset_workflow')
    metadata=read_json(record/'metadata.json');source=Path(metadata['dataset_measure'])
    destination=folder/'reports'/group/record.name
    checkpoint=folder/'runs/best.pt'
    masks=dict(artery=workflow.manual_mask(source,'artery'),vein=workflow.manual_mask(source,'vein'),
               choroid=workflow.choroidal_masks(source)[0][0])
    signature=dict(checkpoint_sha256=api.sha256(checkpoint),frames_sha256=api.sha256(record/'frames.npy'),
                   masks={key:api.sha256(path) for key,path in masks.items()})
    if (destination/'report.html').exists():
        if not (destination/'benchmark_sources.json').exists() or read_json(destination/'benchmark_sources.json')!=signature:
            raise ValueError(f'Existing report sources changed: {destination}')
        return
    denoised_root=folder/'denoised'/group
    array=denoised_root/record.name/'denoised.npy'
    if not array.exists():
        api.main(['denoise','--record',str(record),'--checkpoint',str(checkpoint),
                  '--output',str(denoised_root),'--device',device])
    provenance=read_json(array.with_suffix('.json'))
    if provenance['checkpoint_sha256']!=api.sha256(checkpoint) or provenance['record_sha256']!=api.sha256(record/'frames.npy'):
        raise ValueError('Existing denoised output does not match checkpoint or prepared input')
    api.main(['evaluate','--record',str(record),'--denoised',str(array),'--output',str(destination),
              '--retinal-artery-mask',str(masks['artery']),
              '--retinal-vein-mask',str(masks['vein']),
              '--choroidal-masks',str(masks['choroid'])])
    api.write_json(destination/'benchmark_sources.json',signature)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',help='Existing workflow prepared/ directory (or a single prepared record)')
    parser.add_argument('--output',required=True,help='New benchmark directory; resume/report commands reuse it')
    parser.add_argument('--config',help='Baseline JSON configuration; all variants inherit it')
    parser.add_argument('--validation-video',help='Video excluded from training in the video-validation variant only')
    parser.add_argument('--measures',nargs='+',help='Select development measurements from the prepared folder')
    parser.add_argument('--evaluation-input',help='Separate dataset folder, now or later with --evaluate-only')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--dry-run',action='store_true',help='Validate all configurations and write the plan without training')
    args=parser.parse_args(argv);output=Path(args.output).resolve()
    if (output/'plan.json').exists():
        if not any((args.resume,args.evaluate_only,args.report_only)):
            raise ValueError('Benchmark exists; use --resume, --evaluate-only or --report-only')
        plan=read_json(output/'plan.json')
        if args.config or args.prepared or args.validation_video or args.measures:
            raise ValueError('Existing plan supplies config, records and validation video; omit those flags')
    else:
        if args.resume or args.evaluate_only or args.report_only:
            raise ValueError('No benchmark plan exists')
        if not args.prepared: parser.error('--prepared is required for a new benchmark')
        paths=api.resolve_training_records([args.prepared])
        if args.measures:
            missing=set(args.measures)-{path.name for path in paths}
            if missing: raise ValueError(f'Unknown prepared measurements: {sorted(missing)}')
            paths=[path for path in paths if path.name in args.measures]
        if len(paths)<2: raise ValueError('At least two development videos are required for video-disjoint validation')
        records=[api.Record(path) for path in paths]
        if len({r.name for r in records})!=len(records) or len({r.frames.shape[1:] for r in records})!=1:
            raise ValueError('Development records need unique names and identical image dimensions')
        if any(r.metadata.get('diaphragm_mask_applied') is not True for r in records):
            raise ValueError('Benchmark requires diaphragm-masked dataset preparations')
        if len({r.metadata['input_mode'] for r in records})!=1:
            raise ValueError('Do not mix raw and AVI preparations')
        base=api.Config(**read_json(Path(args.config))) if args.config else api.Config()
        base.validate()
        selected=args.validation_video or records[-1].name
        if selected not in {r.name for r in records} or selected==records[0].name:
            raise ValueError('Validation video must be present and differ from the first preview video')
        configs=variants(base,selected)
        for name,values in configs.items():
            cfg=api.Config(**values);cfg.validate()
            train,valid=api.split_samples(records,cfg)
            if cfg.samples_per_epoch<len({i for i,t in train}): raise ValueError('Increase samples_per_epoch')
            for stage,pool in (('train',train),('valid',valid)):
                for i,t in pool[:1]: api.replacement(records[i],t,cfg,np.random.default_rng(cfg.seed),stage)
            print(f'Preflight {name}: {len(train)} training targets, {len(valid)} validation targets',flush=True)
        api.load_sibling('training_monitor').Monitor(records[0],base.history,api,2)
        plan=dict(records=[str(p.resolve()) for p in paths],variants=configs,
                  preview_record=records[0].name,validation_video=selected,
                  reference=records[0].metadata,
                  prepared_hashes={r.name:api.sha256(r.path/'metadata.json') for r in records})
        output.mkdir(parents=True,exist_ok=False);api.write_json(output/'plan.json',plan)
    if args.report_only:
        comparison(output,plan);return 0
    if args.dry_run:
        comparison(output,plan);return 0
    for path in map(Path,plan['records']):
        if api.sha256(path/'metadata.json')!=plan['prepared_hashes'][path.name]:
            raise ValueError('Prepared metadata changed since benchmark planning')
    evaluation=[]
    if args.evaluation_input:
        # Check source identities before preparing or evaluating any external data.
        workflow=api.load_sibling('dataset_workflow')
        train_hashes={read_json(Path(p)/'metadata.json')['source_sha256'] for p in plan['records']}
        for source in workflow.measurements(args.evaluation_input):
            if api.sha256(workflow.single_h5(source)) in train_hashes:
                raise ValueError(f'Evaluation source also occurs in development data: {source}')
        evaluation=prepare_evaluation(args.evaluation_input,output,plan['reference'],api)
    if args.evaluate_only and not evaluation:
        parser.error('--evaluate-only requires --evaluation-input')
    failed=False
    for name,config in plan['variants'].items():
        folder=output/name;folder.mkdir(exist_ok=True)
        started=time.monotonic()
        try:
            api.write_json(folder/'status.json',dict(status='running'))
            if not args.evaluate_only and not (folder/'trained.json').exists():
                api.write_json(folder/'config.json',config)
                command=[Path(api.__file__),'train','--records',*plan['records'],
                         '--output',folder/'runs','--config',folder/'config.json','--device',args.device]
                if (folder/'runs/last.pt').exists(): command.append('--resume')
                elif (folder/'runs').exists():
                    # Preserve evidence from an interrupted run before its first checkpoint.
                    (folder/'runs').rename(folder/f'unfinished_runs_{time.time_ns()}')
                run_child(command,folder/'training.log')
                api.write_json(folder/'trained.json',dict(checkpoint_sha256=api.sha256(folder/'runs/best.pt')))
            if not (folder/'trained.json').exists(): raise ValueError('Strategy has not completed training')
            if read_json(folder/'trained.json')['checkpoint_sha256']!=api.sha256(folder/'runs/best.pt'):
                raise ValueError('Trained checkpoint changed since benchmark completion')
            final_report(Path(plan['records'][0]),folder,'development',args.device)
            for record in evaluation: final_report(record,folder,'unseen',args.device)
            api.write_json(folder/'status.json',dict(status='complete',elapsed_seconds=time.monotonic()-started,
                                                    external_evaluation=bool(evaluation)))
        except Exception as exc:
            failed=True;print(f'{name}: {exc}',flush=True)
            api.write_json(folder/'status.json',dict(status='failed',error=str(exc)))
        finally:
            comparison(output,plan)
    return int(failed)


if __name__=='__main__':
    raise SystemExit(main())
