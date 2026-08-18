import io
import re
from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LinearRegression, HuberRegressor

try:
    import colour
except ImportError:  # pragma: no cover - optional dependency for colour science functions
    colour = None

CIE_CACHE_DIR = Path("__cie__cache__")
CIE_CACHE_DIR.mkdir(exist_ok=True)


def download_if_needed(url: str, target: Path) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    target.write_bytes(r.content)
    return target


# constants
BAND_RANGES = {
    "415nm_f1": (400, 430),
    "445nm_f2": (430, 460),
    "480nm_f3": (460, 500),
    "515nm_f4": (500, 540),
    "555nm_f5": (540, 575),
    "590nm_f6": (575, 610),
    "630nm_f7": (610, 650),
    "680nm_f8": (650, 700),
}

AS7341_CHANNELS = BAND_RANGES.keys()

def plot_colour_time_series(df: pd.DataFrame):
    '''
    Given a dataframe with timestamp_iso and 8 AS7341 bands,
    plot the RGB and CIE x,y time series.
    '''
    # --- time axis ---
    t = pd.to_datetime(df["timestamp_iso"])
    t0 = t.iloc[0]
    t_sec = (t - t0).dt.total_seconds().to_numpy()

    # --- NSP32m bands ---
    band_cols = ["415nm_f1","445nm_f2","480nm_f3","515nm_f4",
                "555nm_f5","590nm_f6","630nm_f7","680nm_f8"]
    wls = np.array([415,445,480,515,555,590,630,680], float)
    B = np.clip(df[band_cols].to_numpy(float), 0, None)
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    rgb = np.zeros((len(B), 3))
    xy  = np.zeros((len(B), 2))
    for i, row in enumerate(B):
        sd = colour.SpectralDistribution(dict(zip(wls, row)))
        sd = sd.interpolate(colour.SpectralShape(380, 780, 1))
        XYZ = colour.sd_to_XYZ(sd, cmfs=cmfs)     # relative XYZ
        rgb[i] = colour.XYZ_to_sRGB(XYZ / 100.0)  # absolute (no Y normalisation)
        xy[i]  = colour.XYZ_to_xy(XYZ)
    rgb = np.clip(rgb, 0, 1)
    # --- RGB time series ---
    plt.figure(figsize=(12, 4))
    plt.plot(t_sec, rgb[:,0], label="R")
    plt.plot(t_sec, rgb[:,1], label="G")
    plt.plot(t_sec, rgb[:,2], label="B")
    plt.xlabel("time since start [s]")
    plt.ylabel("sRGB (0–1)")
    plt.title("RGB time series")
    plt.legend()
    plt.show()
    # --- CIE x,y time series ---
    plt.figure(figsize=(12, 4))
    plt.plot(t_sec, xy[:,0], label="x")
    plt.plot(t_sec, xy[:,1], label="y")
    plt.xlabel("time since start [s]")
    plt.ylabel("CIE 1931 chromaticity")
    plt.title("CIE x,y time series")
    plt.legend()
    plt.show()
    pass

class apply_action_spectra:
    KM = 683.0  # lm/W (photopic luminous efficacy)

    AOPIC_COLS = {
        "S":   "s_sc(lambda)",
        "M":   "s_mc(lambda)",
        "L":   "s_lc(lambda)",
        "Rod": "s_rh(lambda)",
        "Mel": "s_mel(lambda)",
    }
    def __init__(self, irradiance, wavelengths):
        Ee = np.asarray(irradiance, dtype=float)

        # Accept either (n_wl,) or (n_rows, n_wl)
        if Ee.ndim == 1:
            Ee = Ee[None, :]  # -> (1, n_wl)
            self._single_row_input = True
        elif Ee.ndim == 2:
            self._single_row_input = False
        else:
            raise ValueError("irradiance must be 1D (n_wl,) or 2D (n_rows, n_wl).")

        self.irradiance = Ee
        n_wl = Ee.shape[-1]

        # wavelengths handling (same logic, but based on n_wl)
        if isinstance(wavelengths, tuple) and len(wavelengths) == 2:
            wavelengths = np.linspace(wavelengths[0], wavelengths[1], n_wl)
        elif isinstance(wavelengths, (list, np.ndarray)):
            wavelengths = np.asarray(wavelengths, dtype=float)
        else:
            raise ValueError("wavelengths must be a tuple (min, max) or a list/array of wavelength values.")

        self.wavelengths = np.asarray(wavelengths, dtype=float)

        if self.wavelengths.shape != (n_wl,):
            raise ValueError("wavelengths must have shape (n_wl,) matching irradiance last dimension.")
        if np.any(np.diff(self.wavelengths) <= 0):
            raise ValueError("wavelengths must be strictly increasing (ascending).")

        # Load CIE datasets
        self.photopic_df = self._read_cie_csv_with_metadata(
            "https://files.cie.co.at/CIE_sle_photopic.csv",
            "https://files.cie.co.at/CIE_sle_photopic.csv_metadata.json",
        )
        self.aopic_df = self._read_cie_csv_with_metadata(
            "https://files.cie.co.at/CIE_a-opic_action_spectra.csv",
            "https://files.cie.co.at/CIE_a-opic_action_spectra.csv_metadata.json",
        )

        self._prepare_weights()
        self._integrate()
    @staticmethod
    def _cie_cache_path(url: str) -> Path:
        filename = url.split("?")[0].rsplit("/", 1)[-1]
        return CIE_CACHE_DIR / filename

    @classmethod
    def _read_cie_csv_with_metadata(cls, csv_url: str, meta_url: str) -> pd.DataFrame:
        csv_cache = cls._cie_cache_path(csv_url)
        meta_cache = cls._cie_cache_path(meta_url)

        if not csv_cache.exists() or not meta_cache.exists():
            csv_cache.parent.mkdir(parents=True, exist_ok=True)
            try:
                meta_resp = requests.get(meta_url, timeout=60)
                meta_resp.raise_for_status()
                meta_cache.write_bytes(meta_resp.content)

                csv_resp = requests.get(csv_url, timeout=60)
                csv_resp.raise_for_status()
                csv_cache.write_bytes(csv_resp.content)
            except requests.RequestException:
                if csv_cache.exists() and meta_cache.exists():
                    pass
                else:
                    raise

        meta_json = json.loads(meta_cache.read_text(encoding="utf-8"))
        column_headers = meta_json.get("datatableInfo", {}).get("columnHeaders", [])
        titles = []
        for entry in column_headers:
            title = entry.get("title") or entry.get("columnName") or entry.get("name")
            if title:
                titles.append(title)

        if not titles:
            # Fallback for metadata files without the expected structure.
            titles = pd.read_csv(csv_cache, nrows=0).columns.tolist()

        df = pd.read_csv(csv_cache, names=titles, header=0)

        if "lambda" in df.columns:
            df = df.rename(columns={"lambda": "wavelength_nm"})
        else:
            wl_candidates = [c for c in df.columns if "lambda" in c.lower() or "wavelength" in c.lower()]
            if not wl_candidates:
                raise KeyError("Could not find wavelength column (expected 'lambda').")
            df = df.rename(columns={wl_candidates[0]: "wavelength_nm"})
        return df

    def _prepare_weights(self):
        wl_user = self.wavelengths

        if "V(lambda)" not in self.photopic_df.columns:
            raise KeyError("Photopic CSV missing 'V(lambda)' column.")
        wl_v = self.photopic_df["wavelength_nm"].to_numpy(dtype=float)
        V = self.photopic_df["V(lambda)"].to_numpy(dtype=float)
        self.V_lambda = np.interp(wl_user, wl_v, V, left=0.0, right=0.0)

        wl_a = self.aopic_df["wavelength_nm"].to_numpy(dtype=float)
        for key, col in self.AOPIC_COLS.items():
            if col not in self.aopic_df.columns:
                raise KeyError(f"Action spectra CSV missing '{col}'.")
            vals = self.aopic_df[col].to_numpy(dtype=float)
            setattr(self, f"s_{key}", np.interp(wl_user, wl_a, vals, left=0.0, right=0.0))

        self.s_S   = getattr(self, "s_S")
        self.s_M   = getattr(self, "s_M")
        self.s_L   = getattr(self, "s_L")
        self.s_Rod = getattr(self, "s_Rod")
        self.s_Mel = getattr(self, "s_Mel")
    @staticmethod
    def _trapz_nm(y, x_nm, axis=-1):
        return np.trapezoid(y, x_nm, axis=axis)

    def _integrate(self):
        Ee = self.irradiance            # (n_rows, n_wl)
        wl = self.wavelengths           # (n_wl,)

        self.photopic = self.KM * self._trapz_nm(Ee * self.V_lambda[None, :], wl, axis=-1)

        self.S_irradiance   = self._trapz_nm(Ee * self.s_S[None, :],   wl, axis=-1)
        self.M_irradiance   = self._trapz_nm(Ee * self.s_M[None, :],   wl, axis=-1)
        self.L_irradiance   = self._trapz_nm(Ee * self.s_L[None, :],   wl, axis=-1)
        self.rod_irradiance = self._trapz_nm(Ee * self.s_Rod[None, :], wl, axis=-1)
        self.mel_irradiance = self._trapz_nm(Ee * self.s_Mel[None, :], wl, axis=-1)

        self.S_lux   = self.KM * self.S_irradiance
        self.M_lux   = self.KM * self.M_irradiance
        self.L_lux   = self.KM * self.L_irradiance
        self.rod_lux = self.KM * self.rod_irradiance
        self.mEDI    = self.KM * self.mel_irradiance

        self.MP_ratio = np.divide(self.mEDI, self.photopic, out=np.zeros_like(self.mEDI), where=self.photopic != 0)

        # If original input was 1D, unwrap back to scalars
        if self._single_row_input:
            self.photopic       = float(self.photopic[0])
            self.S_irradiance   = float(self.S_irradiance[0])
            self.M_irradiance   = float(self.M_irradiance[0])
            self.L_irradiance   = float(self.L_irradiance[0])
            self.rod_irradiance = float(self.rod_irradiance[0])
            self.mel_irradiance = float(self.mel_irradiance[0])
            self.S_lux          = float(self.S_lux[0])
            self.M_lux          = float(self.M_lux[0])
            self.L_lux          = float(self.L_lux[0])
            self.rod_lux        = float(self.rod_lux[0])
            self.mEDI           = float(self.mEDI[0])
            self.MP_ratio       = float(self.MP_ratio[0])

    def summary(self) -> dict:
        return {
            "photopic_lux": self.photopic,
            "mEDI_lux_proxy": self.mEDI,
            "aopic_lux": {"S": self.S_lux, "M": self.M_lux, "L": self.L_lux, "Rod": self.rod_lux, "Mel": self.mEDI},
            "aopic_irradiance_Wm2": {
                "S": self.S_irradiance, "M": self.M_irradiance, "L": self.L_irradiance,
                "Rod": self.rod_irradiance, "Mel": self.mel_irradiance,
            },
            "MP_ratio": self.MP_ratio,
        }

class as7341_calibrator:
    '''
	    Algo:
	      1. load  as7341,  nsp32,  KM_CL_500A csv files
	        fig.1: plot: X: timeseries Y:Ev=> nsp and cl500a
	      2. collapse                 nsp32   KM_CL_500A spectral data (360-780 nm, step=1 nm) into 8 as7341 bands
	        fig2:  2x plot: X: timeseries Y:I  =>  nsp collapsed and cl500a collapsed (and unified timestep) data
             fig3:  2x plot: X: timeseries Y:Ev =>  nsp and cl500a collapsed/unified data
	      3.  per timestep for both nsp32 and KM_CL_500A references
			Fig4: gain/offset plot timeseries with nsp as reference.
			Fig4: gain/offset plot timeseries with cl500a as reference.
		  4. provide representative gain and offset (median over time.)
		  5. Apply gain/offset
			    Fig5: Plot the 8-channel irradiance post calibration on AS7341 --  nsp32 as reference
			    Fig5: Plot the 8-channel irradiance post calibration on AS7341 --  cl500a as reference
		  6. :
		    Fig6:
			    First point cluster 
					X axis AS7341 Ev post-calibration NSP32 reference
					Y axis NSP32 Ev
				2nd point cluster 
					X axis as7341 Ev post-calibration cl500a reference
					Y axis cl500a Ev
    '''
    def __init__(self, as7341_csv: str = r"./20250912AS7341calibration/as7341_log.csv",
                 nsp32_csv: str = r"./20250912AS7341calibration/nsp32_log.csv", 
                 KM_CL_500A_csv: str = r"./20250912AS7341calibration/KM_CL_500A_log.csv", 
                 plot_scatter:bool=False,
                 reference:str="nsp32",
                 verbose:bool=False) -> None:
        if reference not in ("nsp32", "km"):
            raise ValueError("reference must be either 'nsp32' or 'km'.")
        self.reference = reference
        self.as7341_csv = as7341_csv
        self.nsp32_csv = nsp32_csv
        self.KM_CL_500A_csv = KM_CL_500A_csv
        self.verbose = verbose
        self.plot_scatter = plot_scatter
        # 1. load data
        self.load(plot= 0)
        # 2. collapse to as7341 bands
        self.collapse_nsp32_to_as7341_bands(plot= False)
        self.collapse_unifyts_KM_CL_500A_to_as7341_bands(plot= False)
        # self.plot_Ev(10)
        
        # 3. gain and offset per timestep
        self.gain_offset_ts_nsp = self.gain_offset_per_timestep(reference="nsp32")
        self.gain_offset_ts_km = self.gain_offset_per_timestep(reference="km")
        
        # plot gaint and offset 
        # self._plot_gain_offsets(self.gain_offset_ts_nsp, title="Gain and Offset per timestep - reference: NSP32")
        # self._plot_gain_offsets(self.gain_offset_ts_km, title="Gain and Offset per timestep - reference: KM_CL_500A")

        self.__representative_gain_offset__()
        # 5. apply representative calibration
        self.__apply_representative_calibration__()

        if self.plot_scatter:
            # 6. scatter plot comparison of Ev
            # 6.1. calculate the Ev for the nsp32 --> !!!!! very long loop !!!!
            self.nsp32_ev = []
            self.nsp32_medi = []
            for _, row in self.nsp32_collapsed.iterrows():
                aas = apply_action_spectra(irradiance = row[list(AS7341_CHANNELS)].to_numpy(dtype=float), 
                                            wavelengths=list(np.array([ (lo+hi)/2 for lo, hi in BAND_RANGES.values() ])))
                self.nsp32_ev.append(aas.photopic)
                self.nsp32_medi.append(aas.mEDI)
            self.__scatter_plot_comparison(save_fig=True)
    def load(self, plot) -> None:
        self.as7341 = pd.read_csv(self.as7341_csv, header=0)
        self.nsp32  = pd.read_csv(self.nsp32_csv,  header=0)
        self.KM_CL_500A = pd.read_csv(self.KM_CL_500A_csv, header=0, skipinitialspace=True)

        # critical: strip whitespace in headers
        self.as7341.columns = self.as7341.columns.astype(str).str.strip()
        self.nsp32.columns  = self.nsp32.columns.astype(str).str.strip()
        self.KM_CL_500A.columns = self.KM_CL_500A.columns.astype(str).str.strip()

        # Correct KM timestamp hour offset
        self.KM_CL_500A['Date Time'] = pd.to_datetime(self.KM_CL_500A['Date Time']) - pd.Timedelta(hours=1)
        self.KM_CL_500A['Date Time'] = self.KM_CL_500A['Date Time'].dt.strftime('%m/%d/%Y %H:%M')
                
        if plot:
            np_nsp32 = pd.read_csv(self.nsp32_csv, header=0, skipinitialspace=True).loc[:, "340":"1010"]
            np_kmcl500a = pd.read_csv(self.KM_CL_500A_csv, header=0, skipinitialspace=True).loc[:, "360":"780"]

            wl_nsp = np.arange(340, 1011, 5)
            wl_km  = np.arange(360, 781, 1)

            dic = {
                "timestamp_iso_nsp32": [],
                "illuminance_nsp32": [],
                "timestamp_iso_kmcl500a": [],
                "illuminance_kmcl500a": [],
            }
            
            # NSP32 (spectral-only)
            valid_nsp = ~np_nsp32.isnull().all(axis=1)
            idx_nsp = np_nsp32.index[valid_nsp]
            dic["timestamp_iso_nsp32"] = idx_nsp.to_list()
            aas = apply_action_spectra(irradiance=np_nsp32.loc[valid_nsp].to_numpy(dtype=float),
                                        wavelengths=wl_nsp)
            dic["illuminance_nsp32"] = aas.photopic.tolist()
            
            

            # KM-CL500A (spectral-only)
            valid_km = ~np_kmcl500a.isnull().all(axis=1)
            idx_km = np_kmcl500a.index[valid_km]
            dic["timestamp_iso_kmcl500a"] = idx_km.to_list()
            aas = apply_action_spectra(irradiance=np_kmcl500a.loc[valid_km].to_numpy(dtype=float),
                                        wavelengths=wl_km)
            dic["illuminance_kmcl500a"] = aas.photopic.tolist()

            #plot
            ymin = min(np.nanmin(dic["illuminance_nsp32"]),np.nanmin(dic["illuminance_kmcl500a"]))
            ymax = max(np.nanmax(dic["illuminance_nsp32"]),np.nanmax(dic["illuminance_kmcl500a"]))
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
            axes[0].plot(dic["timestamp_iso_nsp32"], dic["illuminance_nsp32"])
            axes[0].set_ylim(ymin, ymax)
            axes[0].set_ylabel("Illuminance")
            axes[0].set_title("NSP32 photopic illuminance")
            axes[1].plot(dic["timestamp_iso_kmcl500a"], dic["illuminance_kmcl500a"])
            axes[1].set_ylim(ymin, ymax)
            axes[1].set_ylabel("Illuminance")
            axes[1].set_xlabel("Time step")
            axes[1].set_title("KM-CL500A photopic illuminance")
            
            for ax in axes:
                ax.grid(True)
                ax.yaxis.set_ticks(np.arange(ymin, ymax + 1, 100))
            plt.tight_layout()
            plt.show()     
    def collapse_nsp32_to_as7341_bands(self, plot) -> pd.DataFrame:
        """
        NSP32 -> AS7341 bands (8 channels).
        - If NSP32 columns are spec1, spec2, ... it renames them to 340..1010 (step=5).
        - Then collapses wavelength columns into BAND_RANGES using mean over [lo, hi] per band.
        Returns a DataFrame with columns = AS7341_CHANNELS and same index as df.
        """
        
        lo = 340
        hi = 1010
        step = 5
        df = self.nsp32
        rename_spec_cols = True

        # 1) Optional rename: spec* -> 340.1010
        if rename_spec_cols:
            spec_cols = [c for c in df.columns if str(c).strip().lower().startswith("spec")]
            wl_list = list(range(lo, hi + 1, step))
            if spec_cols and len(spec_cols) == len(wl_list):
                df.rename(columns={spec_cols[i]: str(wl_list[i]) for i in range(len(spec_cols))}, inplace=True)

        # 2) Collect wavelength columns (handles '340', '340.0', ' 340 ')
        wl_cols, wl_vals = [], []
        for c in df.columns:
            if str(c).strip() in ("timestamp_iso", "timestamp_unix"):
                continue
            s = str(c).strip()
            m = re.fullmatch(r"(\d+)(?:\.\d+)?", s)
            if not m:
                continue
            wl = int(m.group(1))
            if lo <= wl <= hi:
                wl_cols.append(c)
                wl_vals.append(wl)

        wl_vals = np.array(wl_vals, dtype=int)
        order = np.argsort(wl_vals)
        wl_cols = [wl_cols[i] for i in order]
        wl_vals = wl_vals[order]

        if len(wl_cols) == 0:
            raise ValueError("No NSP32 wavelength columns found after renaming/parsing headers.")

        # 3) Collapse into AS7341 bands
        out = pd.DataFrame(index=df.index, columns=list(AS7341_CHANNELS), dtype=float)
        for ch, (b_lo, b_hi) in BAND_RANGES.items():
            mask = (wl_vals >= b_lo) & (wl_vals <= b_hi)
            cols = [wl_cols[i] for i in np.where(mask)[0]]
            out[ch] = df[cols].mean(axis=1) if cols else np.nan
        # add the timestamp columns
        out.insert(0, 'timestamp_iso', df['timestamp_iso'])
        out.insert(1, 'timestamp_unix', df['timestamp_unix'])
        self.nsp32_collapsed = out
        
        if plot:
            print("Plotting collapsed NSP32 data...")
            fig, axes = plt.subplots(len(AS7341_CHANNELS), 1, figsize=(10, 15), sharex=False)
            axes[0].set_title("NSP32 collapsed to AS7341 bands")
            for i, ch in enumerate(AS7341_CHANNELS):
                axes[i].plot(self.nsp32_collapsed['timestamp_iso'], self.nsp32_collapsed[ch], label=ch)
                axes[i].set_ylabel(ch)
                axes[i].set_ylim(0, 0.02)
                # only set xlabel and ticks on the last subplot
                # turn off the x ticks:
                axes[i].xaxis.set_ticks([])
                if i == len(AS7341_CHANNELS) - 1:
                    axes[i].set_xlabel("Time step")
                    axes[i].xaxis.set_ticks(np.arange(0, len(self.nsp32_collapsed), step=10))
                    axes[i].xaxis.set_ticklabels(axes[i].get_xticks(), rotation=50, fontsize=8)
                    # axes[i].tick_params(axis='y', labelsize=8)
    def collapse_unifyts_KM_CL_500A_to_as7341_bands(self, plot) -> pd.DataFrame:
            """
            Collapse KM_CL_500A spectral data (340–1010 nm) into 8 calibration bands.
            Returns a DataFrame with columns = AS7341_CHANNELS and same index as df.
            columns are 361-780 with step 1 nm
            !!! uses fabricated seconds to create timestamp_iso !!!
            
            output is:
            - collapsed to AS7341 bands
            - unified timestamps - replaced closest timestamps from nsp32 data with those from KM_CL_500A data. 
            """
            lo = 360
            hi = 780
            df1 = self.KM_CL_500A
            wl_cols = list(range(lo, hi + 1, 1))
            wl_vals = np.array([int(c) for c in wl_cols])
            out = pd.DataFrame(index=df1.index, columns=AS7341_CHANNELS, dtype=float)

            
            # convert string headers to int
            for i in self.KM_CL_500A.columns[41:]: # columns with only the wavelengths
                # convert to int
                wl_str = str(i).strip()
                if re.fullmatch(r"(\d+)(?:\.\d+)?", wl_str):
                    wl_int = int(wl_str)
                    self.KM_CL_500A.rename(columns={i: wl_int}, inplace=True)
            
                    
            # collapse, and keep the timestamp columns
            for ch, (lo, hi) in BAND_RANGES.items():
                mask = (wl_vals >= lo) & (wl_vals <= hi)
                cols = [wl_cols[i] for i in np.where(mask)[0]]
                out[ch] = df1[cols].mean(axis=1)

            # keep illuminance from KM-CL500A in collapsed output
            out["Ev"] = df1['Ev[lx]'].to_numpy(dtype=float)
            # calculate the mEDI from the collapsed bands
            aas = apply_action_spectra(irradiance = out[list(AS7341_CHANNELS)].to_numpy(dtype=float), 
                                        wavelengths=list(np.array([ (lo+hi)/2 for lo, hi in BAND_RANGES.values() ])))
            out["mEDI"] = aas.mEDI
            # self.df1 = df1 # !!!!! temp
            #self.out = out # !!!!! temp
            


            ######## self.KM_CL_500A_collapsed #######
            # merge timestamp columns: 'Date Time'
            out.insert(0, 'Date Time', self.KM_CL_500A['Date Time'])
            # convert the Date Time to timestamp_iso original datetime does not have seconds but measured 
            # !!! fabricated seconds -- for this to work, all the reads before minute-change must have been removed!!! in 20250912AS7341calibration it corresponds to the first 19 reads
            seconds = []
            for i in range(len(out)):
                seconds.append(f"{(i % 60) + 1:02d}")
            out['Date Time'] = out['Date Time'].astype(str) + ':' + seconds 
            out['timestamp_iso'] = pd.to_datetime(out['Date Time']).dt.strftime('%Y-%m-%dT%H:%M:%S')
            
            # unify the timestamps 
            '''
            Read self.nsp32_collapsed.
            Read self.KM_CL_500A_collapsed.
            1. which one has less rows?
            2. use that one's timestamp_iso to filter the other one: find the closest timestamp_iso in the shorter one and map to the closest one in the longer one.
            3. return the union/unified dataframe -- timestamp should be from the one with less rows.
            '''
            # 1. which one has less rows? self.nsp32_collapsed or self.KM_CL_500A_collapsed
            if len(self.nsp32_collapsed) != len(out):
                shorter_df = self.nsp32_collapsed if len(self.nsp32_collapsed) < len(out) else out
                longer_df  = out if len(self.nsp32_collapsed) < len(out) else self.nsp32_collapsed
                nsp_timestamps = pd.to_datetime(self.nsp32_collapsed["timestamp_iso"]).to_numpy(dtype="datetime64[ns]")
                km_timestamps  = pd.to_datetime(out["timestamp_iso"]).to_numpy(dtype="datetime64[ns]")

                # searchsorted needs sorted km_timestamps; assume they are chronological
                idx = np.searchsorted(km_timestamps, nsp_timestamps)

                # clip to valid interior so idx-1 and idx exist
                idx = np.clip(idx, 1, len(km_timestamps) - 1)

                left  = km_timestamps[idx - 1]
                right = km_timestamps[idx]

                # choose whichever is closer
                choose_left = np.abs(nsp_timestamps - left) <= np.abs(right - nsp_timestamps)
                idx = idx - choose_left.astype(np.int64)
                
                # Now align KM rows to NSP timestamps, use nsp timestamps in the final unified output
                unified_out = out.iloc[idx].reset_index(drop=True)
                unified_out.drop(columns=["Date Time"], inplace=True)
                unified_out["timestamp_iso"] = self.nsp32_collapsed["timestamp_iso"].reset_index(drop=True)                                
                out = unified_out

            else:
                raise NotImplementedError("The case where KM_CL_500A has less rows is not implemented yet.")
            self.KM_CL_500A_collapsed = out


            ######## self.KM_CL_500A_collapsed #######
            
            
            if plot:
                #--- plot collapsed KM_CL_500A data, exactly like nsp32 ---
                print("Plotting collapsed KM_CL_500A data...")
                fig, axes = plt.subplots(len(AS7341_CHANNELS), 1, figsize=(10, 15), sharex=False)
                axes[0].set_title("KM_CL_500A collapsed to AS7341 bands")
                for i, ch in enumerate(AS7341_CHANNELS):
                    axes[i].plot(self.KM_CL_500A_collapsed['timestamp_iso'], self.KM_CL_500A_collapsed[ch], label=ch)
                    axes[i].set_ylabel(ch)
                    axes[i].set_ylim(0.00, 0.02)
                    # only set xlabel and ticks on the last subplot
                    # turn off the x ticks:
                    axes[i].xaxis.set_ticks([])
                    if i == len(AS7341_CHANNELS) - 1:
                        axes[i].set_xlabel("Time step")
                        axes[i].xaxis.set_ticks(np.arange(0, len(self.KM_CL_500A_collapsed), step=10))
                        axes[i].xaxis.set_ticklabels(axes[i].get_xticks(), rotation=50, fontsize=8)
    def plot_Ev(self,number_of_rows:Optional[int]=None) -> None:
        '''
        number_of_rows: Optional[int]: if provided, only plot the first number_of_rows rows. cause the Ev calculation can be time-consuming.
        plot Ev for both nsp32 and cl500a with 8 channels. 
        '''
        print("Plotting Ev for both NSP32 and KM_CL_500A collapsed data...")

        ev = {"nsp32":[], "kmcl500a":[]}
        print(f"Calculating Ev.")
        if number_of_rows is not None:
            for _, row in self.nsp32_collapsed.head(number_of_rows).iterrows():
                aas = apply_action_spectra(irradiance = row[list(AS7341_CHANNELS)].to_numpy(dtype=float), 
                                            wavelengths=list(np.array([ (lo+hi)/2 for lo, hi in BAND_RANGES.values() ])))
                ev["nsp32"].append(aas.photopic)
            # apply action spectra to kmcl500a collapsed data
            for _, row in self.KM_CL_500A_collapsed.head(number_of_rows).iterrows():
                aas = apply_action_spectra(irradiance=row[list(AS7341_CHANNELS)].to_numpy(dtype=float), 
                                            wavelengths=list(np.array([ (lo+hi)/2 for lo, hi in BAND_RANGES.values() ])))
                ev["kmcl500a"].append(aas.photopic)
        else:
            for _, row in self.nsp32_collapsed.iterrows():
                aas = apply_action_spectra(irradiance = row[list(AS7341_CHANNELS)].to_numpy(dtype=float), 
                                        wavelengths=list(np.array([ (lo+hi)/2 for lo, hi in BAND_RANGES.values() ])))
                ev["nsp32"].append(aas.photopic)
                
            # apply action spectra to kmcl500a collapsed data
            for _, row in self.KM_CL_500A_collapsed.iterrows():
                aas = apply_action_spectra(irradiance=row[list(AS7341_CHANNELS)].to_numpy(dtype=float), 
                                        wavelengths=list(np.array([ (lo+hi)/2 for lo, hi in BAND_RANGES.values() ])))
                ev["kmcl500a"].append(aas.photopic)
        fig, axes = plt.subplots(2, 1, figsize=(10, 15), sharex=False)
        axes[0].set_title("NSP32 collapsed to AS7341 bands - Ev")
        if number_of_rows is not None:
            axes[0].plot(self.nsp32_collapsed['timestamp_iso'].head(number_of_rows), ev["nsp32"])
        else:
            axes[0].plot(self.nsp32_collapsed['timestamp_iso'], ev["nsp32"])
        axes[0].set_ylabel("Ev")
        axes[1].set_title("KM_CL_500A collapsed to AS7341 bands - Ev")
        if number_of_rows is not None:
            axes[1].plot(self.KM_CL_500A_collapsed['timestamp_iso'].head(number_of_rows), ev["kmcl500a"])
        else:
            axes[1].plot(self.KM_CL_500A_collapsed['timestamp_iso'], ev["kmcl500a"])
        axes[1].set_ylabel("Ev")
        axes[1].set_xlabel("Time step")
        axes[0].set_ylim(0,1200)
        axes[1].set_ylim(0,1200)
        plt.tight_layout()
    def gain_offset_per_timestep(self, 
                                 reference: str = ["nsp32", "km"],
                                 eps: float = 1e-24) -> pd.DataFrame:

        as7341 = self.as7341.copy()
        nsp32 = self.nsp32_collapsed.copy()
        kmcl500a = self.KM_CL_500A_collapsed.copy()
        
        raw = as7341[list(AS7341_CHANNELS)].to_numpy(dtype=float) # AS7341 raw readings
        if reference == "nsp32":
            ref = nsp32[list(AS7341_CHANNELS)].to_numpy(dtype=float)
        elif reference == "km":
            ref = kmcl500a[list(AS7341_CHANNELS)].to_numpy(dtype=float)
            
        denom = np.where(np.abs(raw) < eps, np.nan, raw)
        gain_t = ref / denom
        offset_t = ref - gain_t * raw
        
        
        out = as7341[["timestamp_iso"]].copy()
        for i, ch in enumerate(AS7341_CHANNELS):
            out[f"gain_{ch}"] = gain_t[:, i]
            out[f"offset_{ch}"] = offset_t[:, i]
            
        return out
    def _plot_gain_offsets(self, df: pd.DataFrame, title: str = "") -> None:
        t = pd.to_datetime(df["timestamp_iso"], errors="coerce")
        fig, axes = plt.subplots(nrows=8, ncols=2, sharex=True, figsize=(14, 18))
        fig.suptitle(title)

        for r, ch in enumerate(AS7341_CHANNELS):
            ax_g = axes[r, 0]
            ax_o = axes[r, 1]

            ax_g.plot(t, df[f"gain_{ch}"].to_numpy(dtype=float))
            ax_g.set_ylabel(ch)
            ax_g.set_title("gain")

            ax_o.plot(t, df[f"offset_{ch}"].to_numpy(dtype=float))
            ax_o.set_title("offset")

        for ax in axes[-1, :]:
            ax.set_xlabel("time")
        
        # set y lim to 0 to 1e-5
        for ax in axes[:, 0]:
            ax.set_ylim(0, 1e-5)
        for ax in axes[:, 1]:
            ax.set_ylim(0, 1e-20)

        fig.tight_layout()
        plt.show()
    def __representative_gain_offset__(
        
        self,
        method: str = "huber",          # "ols" | "huber" | "median_ts"
        amin: float = 50.0,             # drop low raw counts (stabilizes fit)
        q_low: float = 0.01,            # winsorize ref/raw pairs (optional)
        q_high: float = 0.99,
        ) -> pd.DataFrame:
        """
        Returns one (gain, offset) per band

        method:
          - "huber"/"ols": fit R_b = G_b * A_b + O_b using paired samples
          - "median_ts": robust median over gain_offset_per_timestep() outputs
        """
        # --- choose reference bands
        as7341 = self.as7341.copy()
        ref_df = self.nsp32_collapsed if self.reference == "nsp32" else self.KM_CL_500A_collapsed

        # --- align by timestamp_iso (critical if any mismatch)
        keep = ["timestamp_iso"] + list(AS7341_CHANNELS)
        merged = as7341[keep].merge(ref_df[keep], on="timestamp_iso", how="inner", suffixes=("_raw", "_ref"))

        rows = []
        for ch in AS7341_CHANNELS:
            x = merged[f"{ch}_raw"].to_numpy(float)
            y = merged[f"{ch}_ref"].to_numpy(float)

            m = np.isfinite(x) & np.isfinite(y) & (x >= amin)
            x = x[m]
            y = y[m]

            if x.size < 3:
                rows.append({"band": ch, "gain": np.nan, "offset": np.nan, "n": int(x.size)})
                continue

            # optional winsorization to reduce spikes
            xlo, xhi = np.quantile(x, [q_low, q_high])
            ylo, yhi = np.quantile(y, [q_low, q_high])
            m2 = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
            x = x[m2]
            y = y[m2]

            if method == "median_ts":
                ts = self.gain_offset_per_timestep(reference=self.reference)
                g = ts[f"gain_{ch}"].to_numpy(float)
                o = ts[f"offset_{ch}"].to_numpy(float)
                rows.append({"band": ch, "gain": np.nanmedian(g), "offset": np.nanmedian(o), "n": int(np.isfinite(g).sum())})
                continue

            X = x.reshape(-1, 1)
            if method == "ols":
                model = LinearRegression()
            elif method == "huber":
                model = HuberRegressor()
            else:
                raise ValueError("method must be 'ols', 'huber', or 'median_ts'")

            model.fit(X, y)
            rows.append({"band": ch, "gain": float(model.coef_[0]), "offset": float(model.intercept_), "n": int(X.shape[0])})

        out = pd.DataFrame(rows).set_index("band")
        if self.reference == "nsp32":
            self.rep_gain_offset_nsp32 = out
        if self.reference == "km":
            self.rep_gain_offset_km = out
            
        return out
    def __apply_representative_calibration__(self):
        uncalibrated_raw = self.as7341[list(AS7341_CHANNELS)].to_numpy(dtype=float)
        self.uncalibrated_raw = uncalibrated_raw

        if self.reference == "nsp32":
            gain_nsp32 = self.rep_gain_offset_nsp32["gain"].to_numpy(dtype=float)
            offset_nsp32 = self.rep_gain_offset_nsp32["offset"].to_numpy(dtype=float)
            calibrated_nsp32 = uncalibrated_raw * gain_nsp32 + offset_nsp32
            aas = apply_action_spectra(irradiance=calibrated_nsp32, wavelengths=list(np.array([ (lo+hi)/2 for lo, hi in BAND_RANGES.values() ])))
            self.Ev_as7341_calibrated = aas.photopic
            self.mEDI_as7341_calibrated = aas.mEDI
        
        if self.reference == "km":
            gain_km = self.rep_gain_offset_km["gain"].to_numpy(dtype=float)
            offset_km = self.rep_gain_offset_km["offset"].to_numpy(dtype=float)
            calibrated_km = uncalibrated_raw * gain_km + offset_km
            aas = apply_action_spectra(irradiance=calibrated_km, wavelengths=list(np.array([ (lo+hi)/2 for lo, hi in BAND_RANGES.values() ])))
            self.Ev_as7341_calibrated = aas.photopic
            self.mEDI_as7341_calibrated = aas.mEDI
    def __scatter_plot_comparison(self,save_fig:bool=False) -> None:
        # make a scatter plot with random values
        '''6. Scatter plot comparison:
		    Fig6:
			    First point cluster 
					X axis AS7341 Ev post-calibration NSP32 reference
					Y axis NSP32 Ev
				2nd point cluster 
					X axis as7341 Ev post-calibration cl500a reference
					Y axis cl500a Ev 
        '''
        


        quant = "medi"
        if quant == "Ev":   
            if self.reference == "nsp32":
                        gt = self.nsp32_ev
            else:
                gt = self.KM_CL_500A_collapsed["Ev"].to_numpy(dtype=float)
            mbe =  np.mean(self.Ev_as7341_calibrated - gt)
            mape = np.mean(np.abs((self.Ev_as7341_calibrated - gt) / gt)) * 100
            print(f"MBE: {mbe:.4f}, MAPE: {mape:.2f}% for Ev comparison with {self.reference} reference")

            # font to times new roman
            plt.rcParams["font.family"] = "Times New Roman"
            plt.figure(figsize=(6, 4.5))
            plt.scatter(gt, self.Ev_as7341_calibrated, label="NSP32", alpha=0.09, color="black")
            plt.ylabel(r"$E_{v,\mathrm{AS7341}}, calibrated$")
            plt.xlabel(r"$E_{v,\mathrm{reference}}$")
            # set legend top, right, outside
            # plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
            # aspect ratio equal
            plt.gca().set_aspect('equal', adjustable='box')
            # plot y=x line

            lims = [650, max(plt.gca().get_xlim()[1], plt.gca().get_ylim()[1])]
            plt.plot(lims, lims, 'k--', alpha=0.5)
            plt.grid(True)
            # set x and y limits
            plt.xlim(lims)
            plt.ylim(lims)
            if save_fig:
                plt.savefig("scatter_plot_Ev_comparison.pdf", format = "pdf", bbox_inches='tight', dpi=300)
            plt.show()

        if quant == "medi":
            if self.reference == "nsp32":
                gt = self.nsp32_medi
            else:
                gt = self.KM_CL_500A_collapsed["mEDI"].to_numpy(dtype=float)
            mbe =  np.mean(self.mEDI_as7341_calibrated - gt)
            mape = np.mean(np.abs((self.mEDI_as7341_calibrated - gt) / gt)) * 100
            print(f"MBE: {mbe:.4f}, MAPE: {mape:.2f}% for mEDI comparison with {self.reference} reference")

            # font to times new roman
            plt.rcParams["font.family"] = "Times New Roman"
            plt.figure(figsize=(6, 4.5))
            plt.scatter(gt, self.mEDI_as7341_calibrated, label="NSP32", alpha=0.09, color="black")
            plt.ylabel(r"$E_{v,\mathrm{AS7341}}, calibrated$")
            plt.xlabel(r"$E_{v,\mathrm{reference}}$")
            # set legend top, right, outside
            # plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
            # aspect ratio equal
            plt.gca().set_aspect('equal', adjustable='box')
            # plot y=x line

            lims = [650, max(plt.gca().get_xlim()[1], plt.gca().get_ylim()[1])]
            plt.plot(lims, lims, 'k--', alpha=0.5)
            plt.grid(True)
            # set x and y limits
            plt.xlim(lims)
            plt.ylim(lims)
            if save_fig:
                plt.savefig("scatter_plot_mEDI_comparison.pdf", format = "pdf", bbox_inches='tight', dpi=300)
            plt.show()
            