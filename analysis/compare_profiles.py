"""Common individual-path and equal-ensemble central-profile evaluation.

No model is trained, selected or calibrated here. Calendar-paired errors of
random weather are descriptive, not forecasts or reconstruction accuracies.
"""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent.parent
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
import hashlib
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0,str(ROOT))
from scipy.stats import ks_2samp
from study_metrics import path_metrics,acf

SHA="d92748ca6625d0092b073b9bc16345da960a04c42da29926da5b125a63ad122d"
CHANNELS=("solar","wind","temperature")
YEARS=list(range(2020,2025))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path):
    if digest(path)!=SHA:
        raise ValueError("Unmatched input data")
    frame=pd.read_parquet(path)
    dates=pd.to_datetime(frame.time)
    if dates.duplicated().any() or not dates.is_monotonic_increasing or not (dates.diff().dropna()==pd.Timedelta(hours=1)).all():
        raise ValueError("Invalid source chronology")
    frame=frame.loc[~((dates.dt.month==2)&(dates.dt.day==29))]
    years=pd.to_datetime(frame.time).dt.year.to_numpy()
    if sorted(np.unique(years).tolist())!=list(range(1980,2025)) or any((years==y).sum()!=8760 for y in range(1980,2025)):
        raise ValueError("Incomplete annual data")
    raw=frame[list(CHANNELS)].to_numpy(dtype=np.float64).reshape(45,8760,3).transpose(0,2,1)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite source data")
    train=raw[:36]
    return raw,{"scales":train.std(axis=(0,2)),"night":train[:,0].max(0)<=.005,
                "low":float(np.percentile(train[:,1],10)),"high":float(np.percentile(train[:,1],90))}


def baseline_profile(genco_path,ncre_path,train,onshore_only=True):
    genco=pd.read_csv(genco_path)
    ncre=pd.read_csv(ncre_path)
    if "period" in ncre:
        ncre=ncre.sort_values("period")
    if len(ncre)!=8760:
        raise ValueError("Expected the full original 8760-hour profile")
    output=np.empty((3,8760))
    units={}
    for k,tech in ((0,"PHOT"),(1,"WTON")):
        rows=genco[(genco.System.astype(str).str.upper()=="HR") & (genco.technology.astype(str).str.upper()==tech)].copy()
        rows["cap_mw"]=pd.to_numeric(rows.cap_mw,errors="coerce").fillna(0.)
        rows=rows[(rows.cap_mw>0)&rows.unit_name.astype(str).isin(ncre.columns)]
        if k==1 and onshore_only:
            rows=rows[~rows.unit_name.astype(str).str.contains("Offshore",case=False,na=False)]
        if rows.empty:
            raise ValueError("Missing original baseline units")
        weights=rows.cap_mw.to_numpy(dtype=float)
        values=ncre[rows.unit_name.astype(str).tolist()].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite original profile; do not silently fill")
        output[k]=np.clip(values @ (weights/weights.sum()),0,1)
        units[CHANNELS[k]]=dict(zip(rows.unit_name.astype(str),weights.tolist()))
    output[2]=train.mean(0)[2]
    return output,units


def summarize_paths(name,paths,truth,reference):
    """paths maps declared sampling seeds to [5,3,8760] arrays in source units."""
    if truth.shape!=(5,3,8760) or not paths:
        raise ValueError("Expected all five reporting years")
    for values in paths.values():
        if values.shape!=truth.shape or not np.isfinite(values).all():
            raise ValueError("Invalid generated annual paths")
    rows,joint,monthly=[],[],[]
    boundaries=np.r_[0,np.cumsum([31,28,31,30,31,30,31,31,30,31,30,31])]*24
    for seed,values in paths.items():
        for i,year in enumerate(YEARS):
            result,corr=path_metrics(truth[i],values[i],reference["scales"],reference["night"],year,
                                    "prior",seed,1.,reference["low"],reference["high"])
            for k,row in enumerate(result):
                error=values[i,k].astype(np.float64)-truth[i,k]
                row.update(model=name,calendar_MAE=float(np.abs(error).mean()),
                           calendar_RMSE=float(np.sqrt(np.square(error).mean())),bias=float(error.mean()),
                           KS_D=float(ks_2samp(truth[i,k],values[i,k]).statistic),
                           p99_absolute_error=abs(row["generated_p99"]-row["real_p99"]))
            rows.extend(result)
            joint.append({"model":name,**corr})
            for k,channel in enumerate(CHANNELS):
                for month in range(12):
                    segment=slice(boundaries[month],boundaries[month+1])
                    monthly.append({"model":name,"sample_seed":seed,"year":year,"channel":channel,"month":month+1,
                                    "real_mean":float(truth[i,k,segment].mean()),
                                    "generated_mean":float(values[i,k,segment].mean())})
    samples=np.concatenate(list(paths.values()),axis=0)
    central=samples.mean(0,dtype=np.float64)
    central_rows=[]
    annual_rows=[]
    for k,channel in enumerate(CHANNELS):
        error=central[k][None]-truth[:,k]
        central_rows.append({"model":name,"channel":channel,"generated_paths":len(samples),"reporting_hours":5*8760,
                             "MAE":float(np.abs(error).mean()),"RMSE":float(np.sqrt(np.square(error).mean())),
                             "bias":float(error.mean()),"interpretation":"Central calendar-profile diagnostic; not individual-year realism"})
        observed=truth[:,k].mean(-1)
        generated=samples[:,k].mean(-1,dtype=np.float64)
        observed_sd=float(observed.std(ddof=1))
        annual_rows.append({"model":name,"channel":channel,"observed_years":5,"generated_paths":len(samples),
                            "observed_annual_mean_SD":observed_sd,"generated_annual_mean_SD":float(generated.std(ddof=1)),
                            "annual_mean_SD_ratio":float(generated.std(ddof=1)/max(observed_sd,1e-12)),
                            "observed_mean":float(observed.mean()),"generated_mean":float(generated.mean())})
    frame=pd.DataFrame(rows)
    summary={"model":name,"paths":len(samples),"score":float(frame.selection_component.mean()),"channels":{}}
    if len(paths)>1:
        spread=np.stack(list(paths.values())).std(0).mean(axis=(0,2)) / reference["scales"]
        summary["diversity_fraction_training_std"]=dict(zip(CHANNELS,spread.tolist()))
    for channel,group in frame.groupby("channel"):
        summary["channels"][channel]={key:float(group[key].mean()) for key in
            ("calendar_MAE","calendar_RMSE","bias","wasserstein","KS_D","p99_absolute_error",
             "real_std","generated_std","real_p99","generated_p99","real_ACF_1","generated_ACF_1",
             "real_ACF_24","generated_ACF_24","real_ACF_168","generated_ACF_168","selection_component")}
        # Equal-length annual paths: pool squared errors before taking the root.
        summary["channels"][channel]["pooled_calendar_RMSE"]=float(np.sqrt(np.mean(np.square(group.calendar_RMSE))))
    return summary,rows,central_rows,annual_rows,joint,monthly


def read_paths(folder,prefix="prior_seed"):
    output={}
    for seed in (101,102,103):
        with np.load(folder/f"{prefix}{seed}.npz") as saved:
            if "years" in saved and saved["years"].tolist()!=YEARS:
                raise ValueError("Frozen output years differ")
            output[seed]=saved["values"]
    return output


def main():
    out=ROOT/"results"
    out.mkdir(exist_ok=True)
    raw,reference=load_data(ROOT/"data/HR_data.parquet")
    split_rows=[]
    for split,first,last in (("train",0,36),("validation",36,40),("reporting",40,45)):
        subset=raw[first:last]
        for k,channel in enumerate(CHANNELS):
            values=subset[:,k]
            split_rows.append({"split":split,"channel":channel,"years":last-first,"mean":float(values.mean()),
                               "hourly_SD":float(values.std()),"p99":float(np.percentile(values,99)),
                               "annual_mean_SD":float(values.mean(-1).std(ddof=1)),
                               "mean_annual_ACF1":float(np.mean([acf(v,1) for v in values])),
                               "mean_annual_ACF24":float(np.mean([acf(v,24) for v in values]))})
    pd.DataFrame(split_rows).to_csv(out/"dataset_split_statistics.csv",index=False)
    artifact_dir=ROOT/"frozen_outputs" if (ROOT/"frozen_outputs").exists() else ROOT/"data"
    models={
        "selected_CVAE":read_paths(artifact_dir,"selected_ae_prior_seed"),
        "original_CVAE_100epochs":read_paths(artifact_dir,"original_ae_prior_seed")}
    if all((artifact_dir/f"laptop8h_prior_seed{seed}.npz").exists() for seed in (101,102,103)):
        models["original_consumer_CVAE8h_CPU900"]=read_paths(artifact_dir,"laptop8h_prior_seed")
    baseline,units=baseline_profile(ROOT/"data/original_genco.csv",ROOT/"data/original_ncre.csv",raw[:36],onshore_only=False)
    models["original_H2RES_as_published_allwind"]={0:np.repeat(baseline[None],5,axis=0)}
    onshore,_=baseline_profile(ROOT/"data/original_genco.csv",ROOT/"data/original_ncre.csv",raw[:36],onshore_only=True)
    models["original_H2RES_onshore_sensitivity"]={0:np.repeat(onshore[None],5,axis=0)}
    if (artifact_dir/"current_saved_ensemble_first15.npz").exists():
        with np.load(artifact_dir/"current_saved_ensemble_first15.npz") as saved:
            values=saved["values"]
        models["current_saved_pipeline_first15"]={group:values[i*5:(i+1)*5] for i,group in enumerate((-1,-2,-3))}
    for seed in (17,29,43):
        folder=ROOT/"runs"/f"hybrid_seed{seed}_faculty_cpu"
        if (folder/"report.json").exists() and json.loads((folder/"report.json").read_text())["status"]=="completed":
            models[f"hybrid_refit_CPU_seed{seed}"]=read_paths(folder)
    gpu=ROOT/"runs/hybrid_seed17_faculty_cuda"
    if (gpu/"report.json").exists() and json.loads((gpu/"report.json").read_text())["status"]=="completed":
        models["hybrid_refit_GPU_seed17"]=read_paths(gpu)
    default=ROOT/"runs/hybrid_default_seed42_faculty_cuda"
    retry=ROOT/"runs/hybrid_default_seed42_faculty_cuda_retry1"
    if (retry/"report.json").exists() and json.loads((retry/"report.json").read_text())["status"]=="completed":
        default=retry
    if (default/"report.json").exists() and json.loads((default/"report.json").read_text())["status"]=="completed":
        models["current_default_refit_GPU_seed42"]=read_paths(default)
    optimized=ROOT/"runs/hybrid_default_seed42_GPUfit_CPUpredict"
    if (optimized/"report.json").exists() and json.loads((optimized/"report.json").read_text())["status"]=="completed":
        models["current_default_GPUfit_CPUpredict_seed42"]=read_paths(optimized)
    cpu=ROOT/"runs/selected_ae_faculty_cpu900_seed17/study_test"
    if (cpu/"summary.json").exists():
        models["selected_architecture_CPU900"]=read_paths(cpu)
    laptop_hybrid=ROOT/"runs/hybrid_seed17_laptop_cpu_attempt2"
    if (laptop_hybrid/"report.json").exists() and json.loads((laptop_hybrid/"report.json").read_text())["status"]=="completed":
        models["hybrid_laptop_CPU_seed17"]=read_paths(laptop_hybrid)
    laptop_ae=ROOT/"runs/selected_ae_laptop_cpu900_seed17_attempt2/study_test"
    if (laptop_ae/"summary.json").exists():
        models["selected_architecture_laptop_CPU900"]=read_paths(laptop_ae)
    summaries=[]
    tables={key:[] for key in ("individual_path_metrics","central_profile_metrics","annual_mean_spread","cross_channel","monthly_statistics")}
    for name,paths in models.items():
        summary,individual,central,annual,joint,monthly=summarize_paths(name,paths,raw[40:],reference)
        summaries.append(summary)
        for key,value in zip(tables,(individual,central,annual,joint,monthly)):
            tables[key].extend(value)
    yield_rows=[]
    for name,paths in models.items():
        values=np.concatenate(list(paths.values()),axis=0)
        means=(values[:,0]+values[:,1]).mean(-1,dtype=np.float64)
        yield_rows.append({"model":name,"paths_or_years":len(means),"mean_renewable_yield_index":float(means.mean()),
                           "annual_yield_index_SD":float(means.std(ddof=1)),
                           "p10":float(np.percentile(means,10)),"p50":float(np.percentile(means,50)),
                           "p90":float(np.percentile(means,90)),"minimum":float(means.min()),"maximum":float(means.max())})
    for split,first,last in (("observed_train",0,36),("observed_validation",36,40),("observed_reporting",40,45)):
        means=(raw[first:last,0]+raw[first:last,1]).mean(-1)
        yield_rows.append({"model":split,"paths_or_years":len(means),"mean_renewable_yield_index":float(means.mean()),
                           "annual_yield_index_SD":float(means.std(ddof=1)),
                           "p10":float(np.percentile(means,10)),"p50":float(np.percentile(means,50)),
                           "p90":float(np.percentile(means,90)),"minimum":float(means.min()),"maximum":float(means.max())})
    pd.DataFrame(yield_rows).to_csv(out/"renewable_yield_index_spread.csv",index=False)
    for name,rows in tables.items():
        pd.DataFrame(rows).to_csv(out/(name+".csv"),index=False)
    lookup={s["model"]:s for s in summaries}
    # Stored paths were rounded to float32 when archived; the original score used float64.
    for name,expected in (("selected_CVAE",.19504503375228757),("original_CVAE_100epochs",.4978122301750438)):
        if abs(lookup[name]["score"]-expected)>2e-7:
            raise AssertionError("Frozen-path score mismatch: "+name)
    if "original_consumer_CVAE8h_CPU900" in lookup and abs(lookup["original_consumer_CVAE8h_CPU900"]["score"]-.17970732620622296)>2e-7:
        raise AssertionError("Original consumer checkpoint archived-path mismatch")
    result={"status":"complete" if all(f"hybrid_refit_CPU_seed{s}" in lookup for s in (17,29,43)) else "partial_awaiting_new_fits",
            "data_sha256":SHA,"reporting_years":YEARS,"model_selection_performed":False,
            "central_and_individual_metrics_separated":True,"full_reporting_hours":43800,
            "frozen_path_score_reproduction_tolerance":2e-7,"models":summaries,
            "baseline_units":units,"temperature_baseline_is_training_climatology_proxy":True}
    result["stored_ensemble_negative_sample_ids_are_bookkeeping_groups_not_rng_seeds"]=True
    from audit import input_provenance
    result.update(input_provenance(json.loads((ROOT/"data_provenance.json").read_text())))
    (out/"comparison_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (out/"input_hashes.json").write_text(json.dumps({name:digest(ROOT/"data"/name) for name in
        ("HR_data.parquet","original_genco.csv","original_ncre.csv")},indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"scores":{s["model"]:s["score"] for s in summaries}},indent=2))


if __name__=="__main__":
    main()
