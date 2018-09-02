import pandas as pd
import numpy as np
import pathlib as pl
from typing import List, Dict


def get_wf_path(method: str, dataset_name: str, basepath: str) -> pl.Path:
    return pl.Path(basepath).joinpath(method).joinpath(dataset_name)


class SingleWFResultsReader(object):
    def __init__(self, path):
        assert type(path) in [pl.PosixPath, str], "path should be of type string or pl.PosixPath"
        self.pl_path = SingleWFResultsReader.inst_path(path)
        self.Mterm = self.read_mterm()
        self.Mterms = self.read_mterm(plural=True)
        self.mterm_nwavelets = self.read_mterm_nwavelets()
        self.threshold = self.read_threshold()
        self.mterm_error = self.read_mterm_err()  # type: dict
        self.error_by_forest = self.read_error_by_forest()
        self.nwavelets = self.read_nwavelets()

        self.alpha_mterm = self.read_alpha_mterm()

    @staticmethod
    def inst_path(path):
        return path if type(path) == pl.PosixPath else pl.Path(path)

    def read_threshold(self):
        return \
            pd.read_csv(self.pl_path.joinpath("threshold.txt"), names=["t"], dtype={'t': np.float64},
                        squeeze=True).values[
                0]

    def read_mterm(self, plural=False):
        _s = "Mterms.txt" if plural else "Mterm.txt"
        return pd.read_csv(self.pl_path.joinpath(_s), names=["MT"], squeeze=True).values[0]

    def read_mterm_err(self):
        _d = {"Training": None, "Testing": None, "Validating": None}
        for k in _d.keys():
            _d[k] = pd.read_csv(self.pl_path.joinpath("mTermErrorOn{}.txt".format(k)), names=['error'])
            _d[k]['nWavelets'] = self.mterm_nwavelets
        return _d

    def read_error_by_forest(self):
        _d = {"Train": None, "Test": None, "Valid": None}
        for k in _d.keys():
            _d[k] = pd.DataFrame(columns=['None', self.threshold])
            _d[k]["None"] = pd.read_csv(self.pl_path.joinpath("{}errorByForest.txt".format(k)), names=['None'],
                                        squeeze=True)
            thresh_path = \
                list(self.pl_path.glob("{}errorByForestWithThreshold{}*.txt".format(k, str(self.threshold)[:4])))[0]
            _d[k][self.threshold] = pd.read_csv(self.pl_path.joinpath(thresh_path.name), names=[self.threshold],
                                                squeeze=True)
        return _d

    def read_nwavelets(self):
        _d = pd.DataFrame(columns=['None', self.threshold])
        _d["None"] = pd.read_csv(self.pl_path.joinpath("NwaveletsInRF.txt"), names=['None'], squeeze=True)

        return _d

    def read_mterm_nwavelets(self):
        return pd.read_csv(self.pl_path.joinpath("mTermNwavelets.txt"), names=['mTermNwavelets'], squeeze=True)

    def read_alpha_mterm(self):
        return pd.read_csv(self.pl_path.joinpath("alphaMterm.txt"), names=['alphaMterm'], squeeze=True).values[0]


class WFDirectoryReader(object):
    def __init__(self, path):
        assert type(path) in [pl.PosixPath, str], "path should be of type string or pl.PosixPath"
        self.pl_path = SingleWFResultsReader.inst_path(path)
        self.rf_list = list(map(lambda x: SingleWFResultsReader(x),
                                [d for d in self.pl_path.iterdir() if d.is_dir()]))  # type: List[SingleWFResultsReader]
        self.nwavelets = self.melt_nwavelets()
        self.error_by_nwavelets = self.melt_mterm_error()
        self.error_by_tree = self.melt_error_by_tree()
        self.alpha = np.array([wf.alpha_mterm for wf in self.rf_list])

    def melt_nwavelets(self):
        def prep_table(wf, i):
            _d = wf.nwavelets.loc[:, ["None"]].reset_index().copy()
            _d['#wf'] = i
            return _d

        return pd.concat([prep_table(wf, i) for i, wf in enumerate(self.rf_list)])

    def melt_mterm_error(self):
        _d = {"Training": None, "Testing": None, "Validating": None}

        def prep_table(table, i):
            __d = table.copy()
            __d['#wf'] = i
            return __d

        for k in _d.keys():
            _d[k] = pd.concat([prep_table(wf.mterm_error[k], i) for i, wf in enumerate(self.rf_list)])
        return _d

    def melt_error_by_tree(self):
        _d = {"Train": None, "Test": None, "Valid": None}

        def prep_table(_wf, key, i):
            __d = _wf.error_by_forest[key].copy()
            __d = pd.melt(__d, value_name='Error', var_name='Threshold')
            __d['Threshold'] = __d['Threshold'].apply(lambda x: 'No threshold' if x == 'None' else 'With threshold')
            __d['#wf'] = i
            return __d

        for k in _d.keys():
            _d[k] = pd.concat([prep_table(wf, k, i) for i, wf in enumerate(self.rf_list)])
        return _d


class WFResultsReader(object):
    def __init__(self, dataset_name: str, methods: List[str], basepath: str):
        assert type(basepath) in [pl.PosixPath, str], "path should be of type string or pl.PosixPath"
        self.pl_basepath = SingleWFResultsReader.inst_path(basepath)
        self.wf_dict = {}  # type: Dict[str, WFDirectoryReader]

        for m in methods:
            self.wf_dict[m] = WFDirectoryReader(get_wf_path(m,dataset_name, basepath))

        self.error_by_nwavelets = self.melt_nwavelets()
        self.error_by_tree = self.melt_error_by_tree()
        self.alpha = self.melt_alpha()

    def melt_nwavelets(self):
        _dict = {"Training": None, "Testing": None, "Validating": None}
        for k in _dict.keys():
            _dfl = []
            for m in self.wf_dict.keys():
                _df = self.wf_dict[m].error_by_nwavelets[k].copy()
                _df['method'] = "{}".format(m)
                _dfl.append(_df.copy())

            _dict[k] = pd.concat(_dfl)
        return _dict

    def melt_alpha(self):
        _dfl = []
        for m in self.wf_dict.keys():
            _df = pd.DataFrame(self.wf_dict[m].alpha, columns=['alpha'])
            _df['method'] = m
            _dfl.append(_df.copy())
        return pd.concat(_dfl)

    def melt_error_by_tree(self):
        _dict = {"Train": None, "Test": None, "Valid": None}
        for k in _dict.keys():
            _dfl = []
            for m in self.wf_dict.keys():
                _df = self.wf_dict[m].error_by_tree[k].copy()
                _df['method'] = "{}".format(m)
                _dfl.append(_df.copy())

            _dict[k] = pd.concat(_dfl)
        return _dict

# plt.tight_layout()
