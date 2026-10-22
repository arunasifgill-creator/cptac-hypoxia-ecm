"""Load cBioPortal exports (flattened comma files + tab phospho) into sample x feature tables.
Merges the 174-gene set with the 132-gene extension."""
import pandas as pd, numpy as np, os
D = os.path.join(os.path.dirname(__file__), '..', 'data')
RENAME = {'COXFA4L2': 'NDUFA4L2'}

def _flat(path, n=122):
    toks = open(path).read().strip().split(',')
    assert len(toks) % (n + 1) == 0, (path, len(toks))
    rows = [toks[i:i + n + 1] for i in range(0, len(toks), n + 1)]
    df = pd.DataFrame({r[0]: r[1:] for r in rows})
    df = df.set_index('SAMPLE_ID').drop(columns='STUDY_ID')
    return df.replace('NA', np.nan).astype(float)

def _merge(a, b):
    b = b.reindex(a.index)
    b = b[[c for c in b.columns if c not in a.columns]]
    return pd.concat([a, b], axis=1).rename(columns=RENAME)

def load():
    prot = _merge(_merge(_flat(f'{D}/protein.txt'), _flat(f'{D}/protein_new.txt')), _flat(f'{D}/protein_12.txt'))
    mrna = _merge(_merge(_flat(f'{D}/mrna.txt'), _flat(f'{D}/mrna_new.txt')), _flat(f'{D}/mrna_12.txt'))
    ph = pd.read_csv(f'{D}/phospho.txt', sep='\t', header=None, index_col=0).T
    ph = ph.set_index('SAMPLE_ID').drop(columns='STUDY_ID').astype(float)
    ph.columns.name = None
    clin = pd.read_csv(f'{D}/clinical.tsv', sep='\t').set_index('Sample ID')
    return prot, mrna, ph, clin
