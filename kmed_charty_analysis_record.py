"""
kmed_charty_analysis_record.py

VoiceEMR 실증 데이터 - 음성 녹음 및 의무기록 작성 시간 분석
    - KMed_time    : STT 소요 시간 (분)
    - elapsed2_kmed: STT 완료 후 두 번째 녹음까지 경과 시간 (분)

입력 파일: smc_records.xlsx
    - 시트 "all"       : 전체 케이스 레코드
    - 시트 "exception" : 제외 대상 A_ID 목록 (기록 부적절 판단)

출력:
    1. 성별 / 경력 / 나이대별 KMed_time 분석
        (Mann-Whitney + HL 추정치+CI + Mixed-effects log-scale Ratio)
    2. 성별 데이터 유효성 검사
    3. 직군별 elapsed2_kmed 분석
        (Mann-Whitney + HL 추정치+CI + Mixed-effects log-scale Ratio)
    4. 성별 / 경력 / 나이대별 elapsed2_kmed 분석
        (Mann-Whitney + HL 추정치+CI + Mixed-effects log-scale Ratio)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

# ──────────────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────────────
INPUT_FILE = "smc_records.xlsx"
CUTOFF = pd.Timestamp("2025-05-26")

ALL_C = [
    "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08",
    "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18",
]

# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────────────────────────
all_data = pd.read_excel(INPUT_FILE, sheet_name="all")
exception_data = pd.read_excel(INPUT_FILE, sheet_name="exception")

exid1 = list(exception_data["A_ID"])
source = all_data.to_dict(orient="records")

# ──────────────────────────────────────────────────────────────────────────────
# KMed_time(분) 계산 — recording_time 불필요, 더 많은 케이스 포함
# ──────────────────────────────────────────────────────────────────────────────
result_time = []
for d in source:
    try:
        t = d["KMed_time"]
        d_new = {**d, "KMed_time": (t.hour * 3600 + t.minute * 60 + t.second) / 60}
        result_time.append(d_new)
    except Exception:
        pass

result_filtered1_time = [d for d in result_time if d["A_ID"] not in exid1]

# ──────────────────────────────────────────────────────────────────────────────
# elapsed2_kmed(분) 계산 — recording_time 필요, KMed_time 계산 가능 케이스의 부분집합
# ──────────────────────────────────────────────────────────────────────────────
result_elapsed = []
for d in source:
    try:
        kmed_dt = datetime.strptime(d["KMed_date"], "%Y/%m/%d_%H:%M")
        t = d["KMed_time"]
        kmed_dur = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)

        try:
            rec2 = datetime.strptime(d["recording_time_2"], "%Y/%m/%d_%H:%M")
            elapsed2 = rec2 - kmed_dt - kmed_dur
        except Exception:
            rec1 = datetime.strptime(d["recording_time_1"], "%Y/%m/%d_%H:%M")
            elapsed2 = rec1 - kmed_dt - kmed_dur

        d_new = {
            **d,
            "KMed_time":     (t.hour * 3600 + t.minute * 60 + t.second) / 60,
            "elapsed2_kmed": int(elapsed2.total_seconds()) / 60,
        }
        result_elapsed.append(d_new)
    except Exception:
        pass

result_filtered1_elapsed = [d for d in result_elapsed if d["A_ID"] not in exid1]

# ──────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────────────────────────────────────
def med_iqr(vals):
    if not vals:
        return float("nan"), float("nan"), float("nan")
    a = np.array(vals, dtype=float)
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))


def hodges_lehmann_ci(x, y, alpha=0.05):
    x, y  = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    diffs = (y[:, None] - x[None, :]).ravel()
    hl    = np.median(diffs)
    N     = len(x) * len(y)
    c     = stats.norm.ppf(1 - alpha / 2) * np.sqrt(len(x) * len(y) * (len(x) + len(y) + 1) / 12)
    ds    = np.sort(diffs)
    lo    = ds[max(0,   int(np.floor((N - c) / 2)))]
    hi    = ds[min(N-1, int(np.ceil( (N + c) / 2)))]
    return hl, lo, hi


def rank_biserial(x, y):
    nx, ny = len(x), len(y)
    u, _   = stats.mannwhitneyu(y, x, alternative="two-sided")
    return 1 - 2 * u / (nx * ny)


def lme_log(vals_pre, vals_post, cids_pre, cids_post):
    """log(y) ~ period + (1|C_ID) → Ratio = exp(coef), 군집 보정"""
    recs = (
        [{"C_ID": c, "post_int": 0, "log_y": np.log(v)}
            for c, v in zip(cids_pre,  vals_pre)  if v > 0] +
        [{"C_ID": c, "post_int": 1, "log_y": np.log(v)}
            for c, v in zip(cids_post, vals_post) if v > 0]
    )
    sub = pd.DataFrame(recs).dropna()
    if sub["C_ID"].nunique() < 2:
        return None, None, None, None
    try:
        result = MixedLM.from_formula("log_y ~ post_int", groups="C_ID", data=sub).fit(
            reml=True, method="lbfgs"
        )
        coef  = float(result.params["post_int"])
        ci    = result.conf_int()
        ci_lo = float(ci.loc["post_int", 0])
        ci_hi = float(ci.loc["post_int", 1])
        pval  = float(result.pvalues["post_int"])
        return round(np.exp(coef), 3), round(np.exp(ci_lo), 3), round(np.exp(ci_hi), 3), round(pval, 4)
    except Exception:
        return None, None, None, None


def build_row(title, cids_group, bef_records, aft_records, y_col):
    bef_vals = [d[y_col] for d in bef_records]
    aft_vals = [d[y_col] for d in aft_records]

    mean_c = float(np.mean(bef_vals)) if bef_vals else float("nan")
    std_c  = float(np.std(bef_vals))  if bef_vals else float("nan")
    mean_t = float(np.mean(aft_vals)) if aft_vals else float("nan")
    std_t  = float(np.std(aft_vals))  if aft_vals else float("nan")
    diff   = mean_t - mean_c

    p_mwu = (stats.mannwhitneyu(bef_vals, aft_vals, alternative="two-sided")[1]
                if bef_vals and aft_vals else float("nan"))

    try:
        hl, hl_lo, hl_hi = hodges_lehmann_ci(bef_vals, aft_vals)
        r_rb = rank_biserial(bef_vals, aft_vals)
    except Exception:
        hl = hl_lo = hl_hi = r_rb = float("nan")

    # LME 원값 (군집 보정)
    recs = [{"C_ID": d["C_ID"],
                "period": "post" if d["date"] >= CUTOFF else "pre",
                "y": float(d[y_col])}
            for d in (bef_records + aft_records)]
    sub = pd.DataFrame(recs)
    sub["period"] = pd.Categorical(sub["period"], categories=["pre", "post"])
    n_clin = sub["C_ID"].nunique()
    if n_clin >= 2:
        try:
            mf = smf.mixedlm("y ~ period", sub, groups=sub["C_ID"]).fit(reml=True, method="lbfgs")
            p_mixed = round(float(mf.pvalues["period[T.post]"]), 4)
        except Exception:
            p_mixed = float("nan")
    else:
        p_mixed = float("nan")

    # LME log-scale (군집 보정) → Ratio
    bef_cids = [d["C_ID"] for d in bef_records]
    aft_cids = [d["C_ID"] for d in aft_records]
    ratio, ratio_lo, ratio_hi, p_lme_log = lme_log(bef_vals, aft_vals, bef_cids, aft_cids)

    med_c, q1_c, q3_c = med_iqr(bef_vals)
    med_t, q1_t, q3_t = med_iqr(aft_vals)

    return {
        "title":           title,
        "n_clin":          n_clin,
        "n_pre":           len(bef_vals),
        "n_post":          len(aft_vals),
        "mean_pre":        round(mean_c, 2),
        "std_pre":         round(std_c,  2),
        "median_pre":      f"{med_c:.2f} [{q1_c:.2f}–{q3_c:.2f}]",
        "mean_post":       round(mean_t, 2),
        "std_post":        round(std_t,  2),
        "median_post":     f"{med_t:.2f} [{q1_t:.2f}–{q3_t:.2f}]",
        "mean_diff":       round(diff,          2),
        "median_diff":     round(med_t - med_c, 2),
        "direction":       "↑ 증가" if diff > 0 else "↓ 감소",
        "p_mwu":           round(p_mwu, 4),
        "HL_estimate":     round(hl,    2) if hl    == hl    else float("nan"),
        "HL_CI95":         f"[{hl_lo:.2f}, {hl_hi:.2f}]" if hl == hl else "N/A",
        "rank_biserial_r": round(r_rb,  3) if r_rb  == r_rb  else float("nan"),
        "p_mixed":         p_mixed,
        "Ratio":           ratio,
        "Ratio_CI95":      f"[{ratio_lo:.3f}, {ratio_hi:.3f}]" if ratio is not None else "N/A",
        "p_lme_log":       p_lme_log,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 그룹 정의
# ──────────────────────────────────────────────────────────────────────────────
groups_role = {
    "전문간호사": ["C01", "C02", "C03", "C04", "C05"],
    "전공의":     ["C06", "C07", "C08"],
    "전문의+교수": ["C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18"],
    "전체":       ALL_C,
}

groups_demo_time = {
    "여성":               ["C01", "C02", "C03", "C04", "C05", "C06", "C07",
                            "C09", "C11", "C13", "C15", "C16"],
    "남성":               ["C08", "C10", "C12", "C14", "C17", "C18"],
    "5년 미만":           ["C06", "C07", "C08"],
    "5년 이상 10년 미만":  ["C05", "C09", "C10", "C11"],
    "10년 이상 15년 미만": ["C01", "C02", "C03", "C04", "C13", "C16", "C17"],
    "15년 이상":          ["C12", "C14", "C15", "C18"],
    "20대":               ["C06", "C08"],
    "30대":               ["C01", "C02", "C04", "C05", "C07", "C09", "C11",
                            "C13", "C16"],
    "40대":               ["C03", "C10", "C14", "C15", "C17"],
    "50대":               ["C12", "C18"],
    "전체":               ALL_C,
}

groups_demo_elapsed = {
    "여성":               ["C01", "C02", "C03", "C04", "C05", "C06", "C07",
                            "C09", "C11", "C13", "C15", "C16"],
    "남성":               ["C08", "C10", "C12", "C14", "C17", "18"],  # 원본 코드 유지
    "5년 미만":           ["C06", "C07", "C08"],
    "5년 이상 10년 미만":  ["C05", "C09", "C10", "C11"],
    "10년 이상 15년 미만": ["C01", "C02", "C03", "C04", "C13", "C16", "C17"],
    "15년 이상":          ["C12", "C14", "C15", "C18"],
    "20대":               ["C06", "C08"],
    "30대":               ["C01", "C02", "C04", "C05", "C07", "C09", "C11",
                            "C13", "C16"],
    "40대":               ["C03", "C10", "C14", "C15", "C17"],
    "50대":               ["C12", "C18"],
    "전체":               ALL_C,
}


# ──────────────────────────────────────────────────────────────────────────────
# [셀 6] 성별 / 경력 / 나이대별 KMed_time 분석
# ──────────────────────────────────────────────────────────────────────────────
rows = []
for title, cids_group in groups_demo_time.items():
    bef = [d for d in result_filtered1_time if d["C_ID"] in cids_group and d["date"] <  CUTOFF]
    aft = [d for d in result_filtered1_time if d["C_ID"] in cids_group and d["date"] >= CUTOFF]
    rows.append(build_row(title, cids_group, bef, aft, "KMed_time"))

print("[KMed_time] 성별 / 경력 / 나이대별")
print(pd.DataFrame(rows).to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# [셀 7] 성별 데이터 유효성 검사
# ──────────────────────────────────────────────────────────────────────────────
for i, d in enumerate(result_filtered1_time):
    if d["gender"] not in ["남자", "여자"]:
        print(i)


# ──────────────────────────────────────────────────────────────────────────────
# [셀 8] 직군별 elapsed2_kmed 분석
# ──────────────────────────────────────────────────────────────────────────────
rows = []
for title, cids_group in groups_role.items():
    bef = [d for d in result_filtered1_elapsed
            if d["C_ID"] in cids_group and d["date"] <  CUTOFF and d["elapsed2_kmed"] > 0]
    aft = [d for d in result_filtered1_elapsed
            if d["C_ID"] in cids_group and d["date"] >= CUTOFF and d["elapsed2_kmed"] > 0]
    rows.append(build_row(title, cids_group, bef, aft, "elapsed2_kmed"))

print("\n[elapsed2_kmed] 직군별")
print(pd.DataFrame(rows).to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# [셀 9] 성별 / 경력 / 나이대별 elapsed2_kmed 분석
# ──────────────────────────────────────────────────────────────────────────────
rows = []
for title, cids_group in groups_demo_elapsed.items():
    bef = [d for d in result_filtered1_elapsed
            if d["C_ID"] in cids_group and d["date"] <  CUTOFF and d["elapsed2_kmed"] > 0]
    aft = [d for d in result_filtered1_elapsed
            if d["C_ID"] in cids_group and d["date"] >= CUTOFF and d["elapsed2_kmed"] > 0]
    rows.append(build_row(title, cids_group, bef, aft, "elapsed2_kmed"))

print("\n[elapsed2_kmed] 성별 / 경력 / 나이대별")
print(pd.DataFrame(rows).to_string(index=False))
