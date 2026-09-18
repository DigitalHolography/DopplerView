"""Streaming epoch diagnostics and plots; no compressed pixels are scored."""
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def atomic_checkpoint(torch, checkpoint, path):
    """An interruption during saving leaves the previous checkpoint usable."""
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


class Monitor:
    def __init__(self, record, history, api, radius):
        workflow = api.load_sibling('dataset_workflow')
        report = api.load_sibling('noise2time_report')
        folder = Path(record.metadata['dataset_measure'])
        paths = dict(retinal_artery=workflow.manual_mask(folder, 'artery'),
                     retinal_vein=workflow.manual_mask(folder, 'vein'),
                     choroidal=workflow.choroidal_masks(folder)[0][0])
        raw = {key: workflow.strict_mask(path, record.roi.shape, api) for key, path in paths.items()}
        raw['background'] = report.derive_background(raw, record.roi, radius)
        self.masks, _, _ = report.exclusive_masks(raw, record.roi)
        self.provenance = dict(masks={key: dict(path=str(path), sha256=api.sha256(path))
                                      for key, path in paths.items()}, dilation_radius=radius)
        self.record, self.history = record, history
        self.report = report
        self.reset()

    def reset(self):
        self.count = 0
        # Welford's algorithm computes per-pixel temporal variance in bounded memory.
        size = int(self.masks['background'].sum())
        self.means = [np.zeros(size), np.zeros(size)]
        self.m2 = [np.zeros(size), np.zeros(size)]
        self.curves = {name: [[], []] for name in self.masks if name != 'background'}

    def update(self, index, prediction):
        if index < self.history:
            return  # Copied history frames cannot demonstrate denoising.
        self.count += 1
        for j, frame in enumerate((self.record.frames[index], prediction)):
            values = np.asarray(frame[self.masks['background']], dtype=np.float64)
            delta = values - self.means[j]
            self.means[j] += delta / self.count
            self.m2[j] += delta * (values - self.means[j])
            for name in self.curves:
                self.curves[name][j].append(float(frame[self.masks[name]].mean(dtype=np.float64)))

    def result(self):
        std = [float(np.sqrt(np.maximum(values / self.count, 0)).mean()) for values in self.m2]
        result = dict(frames_scored=self.count, background_std_original=std[0],
                      background_std_denoised=std[1], NRR=self.report.ratio(std[0]-std[1], std[0]))
        for name, curves in self.curves.items():
            before, after = map(np.asarray, curves)
            result[name] = dict(temporal_correlation=self.report.correlation(before, after),
                               waveform_std_ratio=self.report.ratio(after.std(), before.std()),
                               mean_original=float(before.mean()), mean_denoised=float(after.mean()))
        return result


def save_history(output, rows):
    """Rewrite complete history atomically, then refresh CSV and the loss/quality plot."""
    temporary = output / 'metrics.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows), encoding='utf-8')
    temporary.replace(output / 'metrics.jsonl')
    flat = []
    def flatten(value, prefix, target):
        for key, item in value.items():
            name = f'{prefix}.{key}' if prefix else key
            if isinstance(item, dict):
                flatten(item, name, target)
            else:
                target[name] = item
    for row in rows:
        item = {}; flatten(row, '', item); flat.append(item)
    keys = list(dict.fromkeys(key for item in flat for key in item))
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(flat)
    fig = Figure(figsize=(13, 9), layout='constrained'); FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2).ravel()
    epochs = [row['epoch'] for row in rows]
    for stage in ('train', 'valid'):
        axes[0].plot(epochs, [row[stage]['total'] for row in rows], '.-', label=stage)
    names = sorted({name for row in rows for name in row.get('diagnostics', {})})
    for name in names:
        subset = [row for row in rows if name in row.get('diagnostics', {})]
        x = [row['epoch'] for row in subset]
        values = [row['diagnostics'][name] for row in subset]
        line, = axes[1].plot(x, [v['background_std_denoised'] for v in values], '.-', label=name)
        axes[1].plot(x, [v['background_std_original'] for v in values], '.--', color=line.get_color(), alpha=.5)
        for region in ('retinal_artery', 'retinal_vein', 'choroidal'):
            label = f'{name}: {region}'
            axes[2].plot(x, [v[region]['temporal_correlation'] for v in values], '.-', label=label)
            axes[3].plot(x, [v[region]['waveform_std_ratio'] for v in values], '.-', label=label)
    for ax, title in zip(axes, ('Loss', 'Background temporal std (dashed: original)',
                              'Vessel waveform correlation', 'Vessel waveform std ratio')):
        ax.set(title=title, xlabel='Epoch'); ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=6)
    axes[3].axhline(1, color='gray', linestyle=':')
    fig.savefig(output / 'metrics.png', dpi=140); fig.clear()
