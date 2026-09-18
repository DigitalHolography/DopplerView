"""Temporal split with a held-out cycle and training donors for validation."""
from copy import copy
import numpy as np
from scipy.ndimage import gaussian_filter1d


def temporal_split(records, cfg, api):
    training, candidates = [], []
    detector = api.load_sibling('arterial_peaks')
    for index, record in enumerate(records):
        peaks = np.asarray(record.metadata.get("peaks", []), dtype=int)
        if len(peaks) < 4:
            raise ValueError(f"{record.name}: temporal split needs at least 4 peaks (3 complete cycles)")
        fps = float(record.metadata['fps'])
        guard = int(4*max(.5, .02*fps)+.5)
        valid_start, valid_stop = int(peaks[-2] + guard), int(peaks[-1])
        train_start, train_stop = 0, int(peaks[-2] - guard)
        if valid_stop-valid_start <= cfg.history or train_stop-train_start <= cfg.history:
            raise ValueError(f'{record.name}: training or held-out cycle is too short after temporal guard')
        record.split_ranges = dict(train=(train_start,train_stop), valid=(valid_start,valid_stop))
        record.stage_views = {}
        record.split_signals = {}
        full_phase = np.asarray(record.phase, dtype=int)
        full_valid = np.asarray(record.valid, dtype=bool)
        indices = np.arange(len(full_phase))
        train_donors = {int(phase): np.flatnonzero((full_phase == phase) & full_valid &
                         (indices >= train_start) & (indices < train_stop) & (record.brightness > 1e-8))
                        for phase in np.unique(full_phase) if phase >= 0}
        for stage, destination in (('train',training), ('valid',candidates)):
            lo, hi = record.split_ranges[stage]
            view = copy(record)
            view.donors = train_donors
            view.valid = np.zeros(len(record.frames),bool); view.valid[lo:hi] = full_valid[lo:hi]
            record.stage_views[stage] = view
            record.split_signals[stage] = dict(peaks=[int(p) for p in peaks if lo <= p < hi],
                                               held_out_cycle=stage=='valid', donor_source='training partition',
                                               valid_frames=int(view.valid.sum()), warnings=list(record.metadata.get('warnings', [])))
            eligible = [(index,t) for t in view.eligible(cfg.history) if t-cfg.history>=lo and t<hi]
            if not eligible:
                raise ValueError(f'{record.name}: no eligible {stage} samples after temporal separation')
            destination.extend(eligible)
    if len(candidates) < cfg.validation_samples:
        raise ValueError('Not enough disjoint validation samples; reduce validation_samples')
    # Ensure every video appears in validation, then fill without replacement.
    rng = np.random.default_rng(cfg.seed)
    chosen = [int(rng.choice([j for j,(i,t) in enumerate(candidates) if i==index]))
              for index in range(len(records))]
    if len(chosen) > cfg.validation_samples:
        raise ValueError('validation_samples must cover every video')
    rest = [j for j in range(len(candidates)) if j not in chosen]
    chosen += rng.choice(rest, cfg.validation_samples-len(chosen), replace=False).tolist()
    return training, [candidates[j] for j in sorted(chosen)]
