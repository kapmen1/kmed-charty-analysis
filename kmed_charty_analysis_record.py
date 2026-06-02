"""
kmed_charty_analysis_record.py

VoiceEMR 실증 데이터 - 음성 녹음 및 의무기록 작성 시간 분석
  - KMed_time    : STT 소요 시간 (분)
  - elapsed2_kmed: STT 완료 후 두 번째 녹음까지 경과 시간 (분)

입력 파일: smc_records.xlsx
  - 시트 "all"       : 전체 케이스 레코드
  - 시트 "exception" : 제외 대상 A_ID 목록 (기록 부적절 판단)

출력:
  1. 성별 / 경력 / 나이대별 KMed_time 분석 (Mann-Whitney + Mixed-effects)
  2. 성별 데이터 유효성 검사
  3. 직군별 elapsed2_kmed 분석 (Mann-Whitney)
  4. 성별 / 경력 / 나이대별 elapsed2_kmed 분석 (Mann-Whitney)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.formula.api as smf

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
# Mixed-effects 모형 p-value 헬퍼 함수
# ──────────────────────────────────────────────────────────────────────────────
def mixed_p(cids_group, y_col="KMed_time", data=None):
    """의료진(C_ID)을 랜덤 효과로 보정한 mixed-effects 모형의 period p-value 반환."""
    if data is None:
        data = result_filtered1_time
    recs = [
        {
            "C_ID": d["C_ID"],
            "period": "post" if d["date"] >= CUTOFF else "pre",
            "y": float(d[y_col]),
        }
        for d in data
        if d["C_ID"] in cids_group
    ]
    sub = pd.DataFrame(recs)
    if sub.empty or sub["C_ID"].nunique() < 2:
        return float("nan"), 0
    sub["period"] = pd.Categorical(sub["period"], categories=["pre", "post"])
    n_clin = sub["C_ID"].nunique()
    try:
        mf = smf.mixedlm("y ~ period", sub, groups=sub["C_ID"]).fit(
            reml=True, method="lbfgs"
        )
        p = float(mf.pvalues["period[T.post]"])
    except Exception:
        p = float("nan")
    return p, n_clin


# ──────────────────────────────────────────────────────────────────────────────
# 그룹 정의
# ──────────────────────────────────────────────────────────────────────────────
groups_role = {
    "전문간호사": ["C01", "C02", "C03", "C04", "C05"],
    "전공의":     ["C06", "C07", "C08"],
    "전문의+교수": ["C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18"],
    "전체":       ALL_C,
}

# 셀 6: 성별/경력/나이대 그룹 (KMed_time 분석용)
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

# 셀 9: 성별/경력/나이대 그룹 (elapsed2_kmed 분석용; 원본 코드 그대로 유지)
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
#   Mann-Whitney U test + Linear Mixed-Effects Model (의료진 군집 보정)
# ──────────────────────────────────────────────────────────────────────────────
rows = []
for title, cids_group in groups_demo_time.items():
    bef_vals = [
        d["KMed_time"]
        for d in result_filtered1_time
        if d["C_ID"] in cids_group and d["date"] < CUTOFF
    ]
    aft_vals = [
        d["KMed_time"]
        for d in result_filtered1_time
        if d["C_ID"] in cids_group and d["date"] >= CUTOFF
    ]

    mean_control = float(np.mean(bef_vals)) if bef_vals else float("nan")
    std_control  = float(np.std(bef_vals))  if bef_vals else float("nan")
    mean_case    = float(np.mean(aft_vals))  if aft_vals  else float("nan")
    std_case     = float(np.std(aft_vals))   if aft_vals  else float("nan")
    diff         = mean_case - mean_control

    p_mwu = (
        stats.mannwhitneyu(bef_vals, aft_vals, alternative="two-sided")[1]
        if bef_vals and aft_vals
        else float("nan")
    )
    p_mixed, n_clin = mixed_p(cids_group, "KMed_time", result_filtered1_time)

    rows.append({
        "title":        title,
        "n_clin":       n_clin,
        "n_control":    len(bef_vals),
        "n_case":       len(aft_vals),
        "mean_control": round(mean_control, 2),
        "std_control":  round(std_control,  2),
        "mean_case":    round(mean_case,    2),
        "std_case":     round(std_case,     2),
        "diff":         round(diff,         2),
        "direction":    "↑ 증가" if diff > 0 else "↓ 감소",
        "p_mwu":        round(p_mwu,   4),
        "p_mixed":      round(p_mixed, 4) if p_mixed == p_mixed else float("nan"),
    })

print(pd.DataFrame(rows).to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# [셀 7] 성별 데이터 유효성 검사
# ──────────────────────────────────────────────────────────────────────────────
for i in range(len(result_filtered1_time)):
    if result_filtered1_time[i]["gender"] not in ["남자", "여자"]:
        print(i)


# ──────────────────────────────────────────────────────────────────────────────
# [셀 8] 직군별 elapsed2_kmed 분석
#   Mann-Whitney U test
# ──────────────────────────────────────────────────────────────────────────────
rows = []
for title, cids_group in groups_role.items():
    bef_vals = [
        d["elapsed2_kmed"]
        for d in result_filtered1_elapsed
        if d["C_ID"] in cids_group and d["date"] < CUTOFF and d["elapsed2_kmed"] > 0
    ]
    aft_vals = [
        d["elapsed2_kmed"]
        for d in result_filtered1_elapsed
        if d["C_ID"] in cids_group and d["date"] >= CUTOFF and d["elapsed2_kmed"] > 0
    ]

    mean_control = float(np.mean(bef_vals)) if bef_vals else float("nan")
    std_control  = float(np.std(bef_vals))  if bef_vals else float("nan")
    mean_case    = float(np.mean(aft_vals))  if aft_vals  else float("nan")
    std_case     = float(np.std(aft_vals))   if aft_vals  else float("nan")
    diff         = mean_case - mean_control

    p_mwu = (
        stats.mannwhitneyu(bef_vals, aft_vals, alternative="two-sided")[1]
        if bef_vals and aft_vals
        else float("nan")
    )

    rows.append({
        "title":              title,
        "n_control":          len(bef_vals),
        "n_case":             len(aft_vals),
        "mean_control":       round(mean_control, 2),
        "std_control":        round(std_control,  2),
        "mean_case":          round(mean_case,    2),
        "std_case":           round(std_case,     2),
        "diff(case-control)": round(diff,         2),
        "direction":          "↑ 증가" if diff > 0 else "↓ 감소",
        "p_value":            round(p_mwu, 4),
    })

print(pd.DataFrame(rows).to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# [셀 9] 성별 / 경력 / 나이대별 elapsed2_kmed 분석
#   Mann-Whitney U test
# ──────────────────────────────────────────────────────────────────────────────
rows = []
for title, cids_group in groups_demo_elapsed.items():
    bef_vals = [
        d["elapsed2_kmed"]
        for d in result_filtered1_elapsed
        if d["C_ID"] in cids_group and d["date"] < CUTOFF and d["elapsed2_kmed"] > 0
    ]
    aft_vals = [
        d["elapsed2_kmed"]
        for d in result_filtered1_elapsed
        if d["C_ID"] in cids_group and d["date"] >= CUTOFF and d["elapsed2_kmed"] > 0
    ]

    mean_control = float(np.mean(bef_vals)) if bef_vals else float("nan")
    mean_case    = float(np.mean(aft_vals))  if aft_vals  else float("nan")
    diff         = mean_case - mean_control

    p_mwu = (
        stats.mannwhitneyu(bef_vals, aft_vals, alternative="two-sided")[1]
        if bef_vals and aft_vals
        else float("nan")
    )

    rows.append({
        "title":              title,
        "n_control":          len(bef_vals),
        "n_case":             len(aft_vals),
        "mean_control":       round(mean_control, 2),
        "mean_case":          round(mean_case,    2),
        "diff(case-control)": round(diff,         2),
        "direction":          "↑ 증가" if diff > 0 else "↓ 감소",
        "p_value":            round(p_mwu, 4),
    })

print(pd.DataFrame(rows).to_string(index=False))
