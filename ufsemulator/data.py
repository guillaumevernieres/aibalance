"""
Data preparation uti        self.use_synthetic = dcfg.get('use_synthetic_data', False)
        self.mask_mode = dcfg.get('mask_mode', 'sea_ice')
        # Atmospheric vertical level info (for interpolated files with single level)
        self.atm_level_index = dcfg.get('atm_level_index', DEFAULT_ATM_LEVEL)es for IceNet training.
Refactored to support CF-1 NetCDF inputs from separate atmosphere and ocean/ice files.
Minimized code and error handling per request.
"""

import numpy as np
import netCDF4 as nc
import torch
from typing import Tuple, Dict, Optional
from pathlib import Path
from .cf_mappings import CF_ATM, CF_OCN, DEFAULT_ATM_LEVEL


class IceDataPreparer:
    """
    Prepares training data from CF-1 NetCDF files (atmosphere + ocean/ice).
    """

    def __init__(self, config: Dict):
        self.config = config
        dcfg = config.get('domain', {})
        vcfg = config.get('variables', {})
        self.min_ice = dcfg.get('min_ice_concentration', 0.0)
        self.use_synthetic = dcfg.get('use_synthetic_data', False)
        self.mask_mode = dcfg.get('mask_mode', 'sea_ice')
        # Atmospheric vertical level info (for interpolated files with single level)
        self.atm_level_index = dcfg.get('atm_level_index', 127)  # Default: level 127 (typical nlevs-1 for 128-level model)
        # Inputs/outputs
        self.input_variables = vcfg.get('input_variables')
        self.output_variables = vcfg.get('output_variables', ['aice'])
        mxf = vcfg.get('max_input_features')
        if mxf and mxf > 0:
            self.input_variables = self.input_variables[:mxf]
        self.input_size = len(self.input_variables)
        self.output_size = len(self.output_variables)
        # CF-1 variable map (imported from cf_mappings module)
        self.cf_atm = CF_ATM
        self.cf_ocn = CF_OCN

    def _read_var(self, ds, name):
        return ds.variables[name][:]

    def read_netcdf_data_pair(self, atm_file: str, ocn_file: str) -> Dict[str, np.ndarray]:
        # Read both datasets
        da = nc.Dataset(atm_file, 'r')
        do = nc.Dataset(ocn_file, 'r')
        data: Dict[str, np.ndarray] = {}
        # Coordinates
        lat = self._read_var(da, self.cf_atm['lat']).astype(np.float32)
        lon = self._read_var(da, self.cf_atm['lon']).astype(np.float32)
        # Broadcast to 2D
        lat2 = np.repeat(lat[:, None], lon.size, axis=1)
        lon2 = np.repeat(lon[None, :], lat.size, axis=0)
        data['lat'] = lat2.flatten()
        data['lon'] = lon2.flatten()
        # Atmosphere surface vars (interpolated files contain single level at nlevs-1)
        # Check if vertical dimension exists, if so read from it, otherwise assume 2D
        tair_var = da.variables[self.cf_atm['tair']]
        if len(tair_var.shape) == 3:
            # Has vertical dimension
            nlevs = tair_var.shape[2]
            lev_idx = nlevs - 1
            data['tair'] = self._read_var(da, self.cf_atm['tair'])[:, :, lev_idx].astype(np.float32).flatten()
            data['uatm'] = self._read_var(da, self.cf_atm['uatm'])[:, :, lev_idx].astype(np.float32).flatten()
            data['vatm'] = self._read_var(da, self.cf_atm['vatm'])[:, :, lev_idx].astype(np.float32).flatten()
            data['tsfc'] = self._read_var(da, self.cf_atm['tsfc'])[:, :, lev_idx].astype(np.float32).flatten()
            data['qref'] = self._read_var(da, self.cf_atm['qref'])[:, :, lev_idx].astype(np.float32).flatten()
            atm_level = nlevs - 1
        else:
            # Interpolated file: 2D (lat, lon) - single level already extracted
            data['tair'] = self._read_var(da, self.cf_atm['tair']).astype(np.float32).flatten()
            data['uatm'] = self._read_var(da, self.cf_atm['uatm']).astype(np.float32).flatten()
            data['vatm'] = self._read_var(da, self.cf_atm['vatm']).astype(np.float32).flatten()
            data['tsfc'] = self._read_var(da, self.cf_atm['tsfc']).astype(np.float32).flatten()
            data['qref'] = self._read_var(da, self.cf_atm['qref']).astype(np.float32).flatten()
            atm_level = self.atm_level_index  # Use configured level index
        # Store atmospheric level index for metadata
        data['atm_level_index'] = atm_level
        # Ocean/ice vars
        data['sst'] = self._read_var(do, self.cf_ocn['sst'])[:, :, 0].astype(np.float32).flatten()
        data['sss'] = self._read_var(do, self.cf_ocn['sss'])[:, :, 0].astype(np.float32).flatten()
        data['aice'] = self._read_var(do, self.cf_ocn['aice'])[:, :, 0].astype(np.float32).flatten()
        data['hi'] = self._read_var(do, self.cf_ocn['hi'])[:, :, 0].astype(np.float32).flatten()
        data['hs'] = self._read_var(do, self.cf_ocn['hs'])[:, :, 0].astype(np.float32).flatten()
        data['thick'] = self._read_var(do, self.cf_ocn['thick'])[:, :, 0].astype(np.float32).flatten()
        # Mask: thickness validity (0 < thick <= 500) AND tair bounds (183.15 to 333.15 K)
        data['mask'] = ((data['thick'] > 0.0) & (data['thick'] <= 500.0) &
                       (data['tair'] >= 183.15) & (data['tair'] <= 333.15)).astype(np.int32)
        da.close(); do.close()
        return data

    def filter_data(self, data: Dict[str, np.ndarray], max_patterns: int = 400000) -> Tuple[np.ndarray, ...]:
        # Mask already applied - just select valid points based on mask_mode
        mask = data['mask'] == 1
        aice = data['aice']

        if self.mask_mode == "sea_ice":
            valid = mask & (aice > self.min_ice)
        elif self.mask_mode == "ocean":
            valid = mask & (aice < self.min_ice)
        else:  # "both"
            valid = mask

        indices = np.where(valid)[0][:max_patterns]
        n = len(indices)
        print(f"Selected {n} points from {len(data['lat'])} total")

        patterns = np.zeros((n, self.input_size), dtype=np.float32)
        targets = np.zeros((n, self.output_size), dtype=np.float32)

        for j, name in enumerate(self.input_variables):
            patterns[:, j] = data.get(name, np.zeros(len(data['lat']), dtype=np.float32))[indices]

        for j, name in enumerate(self.output_variables):
            targets[:, j] = data.get(name, np.zeros(len(data['lat']), dtype=np.float32))[indices]

        if self.use_synthetic and 'sst' in self.input_variables and 'aice' in self.output_variables:
            sst_idx = self.input_variables.index('sst')
            aice_idx = self.output_variables.index('aice')
            targets[:, aice_idx] = np.where(patterns[:, sst_idx] < -1.4, 0.9, 0.0)

        lons = data['lon'][indices]
        lats = data['lat'][indices]

        return patterns, targets, lons, lats

    def compute_normalization_stats(self, patterns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.mean(patterns, axis=0).astype(np.float32)
        std = np.std(patterns, axis=0).astype(np.float32)
        std = np.where(std > 1e-6, std, 1.0)
        return mean, std

    def thin_patterns(self, patterns, targets, lons, lats, fraction):
        if fraction < 1.0:
            n = len(targets)
            m = int(n * fraction)
            idx = np.random.choice(n, m, replace=False)
            return patterns[idx], targets[idx], lons[idx], lats[idx]
        return patterns, targets, lons, lats

    def prepare_training_data(self, atm_file: Optional[str] = None,
                              ocn_file: Optional[str] = None,
                              max_patterns: int = 400000,
                              output_file: Optional[str] = None,
                              thin_fraction: float = 1.0) -> Dict:
        data = self.read_netcdf_data_pair(atm_file, ocn_file)
        patterns, targets, lons, lats = self.filter_data(data, max_patterns)
        patterns, targets, lons, lats = self.thin_patterns(patterns, targets, lons, lats, thin_fraction)

        # Compute normalization statistics for both inputs and outputs
        input_mean, input_std = self.compute_normalization_stats(patterns)
        output_mean, output_std = self.compute_normalization_stats(targets)

        # Create CF-1 standard name mappings for DA system
        # Atmospheric variables use nlevs-1 (stored from read), ocean variables use 0
        atm_level = data.get('atm_level_index', self.atm_level_index)

        input_cf_mapping = {}
        output_cf_mapping = {}

        for var in self.input_variables:
            if var in self.cf_atm:
                input_cf_mapping[var] = {
                    'cf_name': self.cf_atm[var],
                    'source': 'atmosphere',
                    'level_index': atm_level
                }
            elif var in self.cf_ocn:
                input_cf_mapping[var] = {
                    'cf_name': self.cf_ocn[var],
                    'source': 'ocean',
                    'level_index': 0
                }

        for var in self.output_variables:
            if var in self.cf_atm:
                output_cf_mapping[var] = {
                    'cf_name': self.cf_atm[var],
                    'source': 'atmosphere',
                    'level_index': atm_level
                }
            elif var in self.cf_ocn:
                output_cf_mapping[var] = {
                    'cf_name': self.cf_ocn[var],
                    'source': 'ocean',
                    'level_index': 0
                }

        result = {
            'inputs': patterns, 'targets': targets, 'lons': lons, 'lats': lats,
            'input_mean': input_mean, 'input_std': input_std,
            'output_mean': output_mean, 'output_std': output_std,
            'metadata': {
                'n_patterns': len(patterns),
                'input_features': self.input_variables,
                'output_features': self.output_variables,
                'input_size': self.input_size,
                'output_size': self.output_size,
                'input_cf_mapping': input_cf_mapping,
                'output_cf_mapping': output_cf_mapping
            }
        }
        if output_file:
            self.save_processed_data(result, output_file)
        return result

    def save_processed_data(self, data: Dict, filename: str) -> None:
        filepath = Path(filename); filepath.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith('.npz'):
            np.savez_compressed(filename, **data)
        elif filename.endswith('.pt'):
            torch_data = {
                'inputs': torch.from_numpy(data['inputs']),
                'targets': torch.from_numpy(data['targets']),
                'lons': torch.from_numpy(data['lons']),
                'lats': torch.from_numpy(data['lats']),
                'input_mean': torch.from_numpy(data['input_mean']),
                'input_std': torch.from_numpy(data['input_std']),
                'metadata': data['metadata']
            }
            torch.save(torch_data, filename)


def create_training_data_from_netcdf(netcdf_file: str,
                                     config: Dict,
                                     output_file: str,
                                     max_patterns: int = 400000) -> str:
    # Use config-specified pair if available
    data_config = config.get('data', {})
    atm = data_config.get('atm_file')
    ocn = data_config.get('ocean_file')
    max_patterns = data_config.get('max_patterns', max_patterns)
    thin_fraction = data_config.get('thin_fraction', 1.0)
    preparer = IceDataPreparer(config)
    preparer.prepare_training_data(atm, ocn, max_patterns, output_file, thin_fraction)
    return output_file


if __name__ == "__main__":
    # Minimal CLI example (expects --atm and --ocn)
    import argparse
    p = argparse.ArgumentParser(description='Prepare training data from CF-1 NetCDF')
    p.add_argument('--atm', required=True)
    p.add_argument('--ocn', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max', type=int, default=400000)
    a = p.parse_args()
    cfg = {'domain': {}, 'data': {}}
    IceDataPreparer(cfg).prepare_training_data(a.atm, a.ocn, a.max, a.out)
