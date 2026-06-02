"""
kmed_charty_roc_evaluation.py

VoiceEMR 실증 데이터 - 구조화 필드 (과거력/계통문진/신체검진) 정확도 평가
  - KMed_emr : AI 생성 의무기록 (KMed_Charty 출력)
  - 과거력 / 계통문진 / 신체검진 : 전문가 수정 의무기록 (label)

입력 파일: smc_records.xlsx
  - 시트 "all"       : A_ID, KMed_emr, 과거력, 계통문진, 신체검진,
                       과거력_수정여부, 계통문진_수정여부, 신체검진_수정여부 컬럼
  - 시트 "exception" : 제외 대상 A_ID 목록

출력:
  1. 비교 데이터 shape 및 head (블록 5)
  2. 카테고리별 Macro/Micro Precision·Recall·Value Accuracy (블록 6)
  3. ':' 없는 label 항목 점검 (블록 7)
"""

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────────────
INPUT_FILE = "smc_records.xlsx"
EMR_CATS   = ["주호소", "현병력", "과거력", "개인력 및 사회력",
              "계통문진", "신체검진", "진단명", "진료 계획"]
EVAL_CATS  = ["과거력", "계통문진", "신체검진"]

# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────────────────────────
all_data       = pd.read_excel(INPUT_FILE, sheet_name="all")
exception_data = pd.read_excel(INPUT_FILE, sheet_name="exception")

samples_except = set(exception_data["A_ID"].tolist())

# ──────────────────────────────────────────────────────────────────────────────
# KMed_emr 파싱 → 카테고리별 구조화 (블록 3)
# ──────────────────────────────────────────────────────────────────────────────
all_data["VoiceEMR_record"] = None
for idx, emr in enumerate(all_data["KMed_emr"]):
    loc  = None
    temp = {}
    lines = str(emr).split("\n")
    lines = [x for x in lines if len(x.lstrip("'").lstrip("- ").strip()) > 2]
    for line in lines:
        if line.strip() in EMR_CATS:
            temp[line.strip()] = []
            loc = line.strip()
        else:
            if loc is not None:
                temp[loc].append(line)
    all_data.at[idx, "VoiceEMR_record"] = temp

# ──────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수 정의 (블록 4)
# ──────────────────────────────────────────────────────────────────────────────
def normalize_value(v):
    """value 문자열에서 +/-를 찾아 정규화. 없으면 원문 유지."""
    if "+" in v:
        return "+"
    elif "-" in v:
        return "-"
    return v.strip()


def parse_items(data):
    """카테고리 데이터를 {key: value} dict로 파싱. 리스트/문자열/NaN 모두 처리."""
    if data is None or isinstance(data, float):
        return {}
    lines = data if isinstance(data, list) else [
        x.strip() for x in str(data).split("\n") if x.strip()
    ]
    result = {}
    for line in lines:
        line = line.lstrip("'").lstrip("- ").strip()
        if ":" in line:
            k, v = line.split(":", 1)
            result[k.strip().lower()] = normalize_value(v)
    return result


def compare_category(label_data, result_data):
    """
    key 기준으로 비교 (소문자 통일):
      TF      : label과 result 모두에 key가 있음
      FN      : label에 있으나 result에 없음
      TN      : result에 있으나 label에 없음
      correct : TF 중 정규화된 value가 일치
    """
    label_dict  = parse_items(label_data)
    result_dict = parse_items(result_data)

    label_keys  = set(label_dict.keys())
    result_keys = set(result_dict.keys())
    both        = label_keys & result_keys

    return {
        "TF":      len(both),
        "FN":      len(label_keys - result_keys),
        "TN":      len(result_keys - label_keys),
        "correct": sum(1 for k in both if label_dict[k] == result_dict[k]),
    }


def safe_div(a, b):
    return a / b if b > 0 else None


# ──────────────────────────────────────────────────────────────────────────────
# 비교 실행 (블록 5)
# ──────────────────────────────────────────────────────────────────────────────
data = all_data[~all_data["A_ID"].isin(samples_except)].reset_index(drop=True)

comparison = []
for _, row in data.iterrows():
    voice_emr_re = row["VoiceEMR_record"]
    record = {"구분자": row["A_ID"]}

    for cat in EVAL_CATS:
        label_dict  = parse_items(row[cat])
        result_dict = parse_items(voice_emr_re.get(cat, []) if isinstance(voice_emr_re, dict) else [])
        stat        = compare_category(row[cat], voice_emr_re.get(cat, []) if isinstance(voice_emr_re, dict) else [])

        record[f"{cat}_label_len"]  = len(label_dict)
        record[f"{cat}_result_len"] = len(result_dict)
        record[f"{cat}_TF"]         = stat["TF"]
        record[f"{cat}_FN"]         = stat["FN"]
        record[f"{cat}_TN"]         = stat["TN"]
        record[f"{cat}_correct"]    = stat["correct"]
        record[f"{cat}_수정여부"]    = row[f"{cat}_수정여부"]

    comparison.append(record)

comparison_df = pd.DataFrame(comparison)
print(comparison_df.shape)
print(comparison_df.head())


# ──────────────────────────────────────────────────────────────────────────────
# 카테고리별 Precision / Recall / Value Accuracy 계산 (블록 6)
# ──────────────────────────────────────────────────────────────────────────────
rows = []
for cat in EVAL_CATS:
    precisions, recalls, val_accs = [], [], []
    sum_TF, sum_result, sum_label, sum_correct = 0, 0, 0, 0

    for _, r in comparison_df.iterrows():
        if r[f"{cat}_label_len"] == 0 and r[f"{cat}_result_len"] == 0:
            continue

        p  = safe_div(r[f"{cat}_TF"], r[f"{cat}_result_len"])
        re = safe_div(r[f"{cat}_TF"], r[f"{cat}_label_len"])
        va = safe_div(r[f"{cat}_correct"], r[f"{cat}_TF"])

        if p  is not None: precisions.append(p)
        if re is not None: recalls.append(re)
        if va is not None: val_accs.append(va)

        sum_TF      += r[f"{cat}_TF"]
        sum_result  += r[f"{cat}_result_len"]
        sum_label   += r[f"{cat}_label_len"]
        sum_correct += r[f"{cat}_correct"]

    rows.append({
        "카테고리":            cat,
        "label_항목수":        int(comparison_df[f"{cat}_label_len"].sum()),
        "n_precision":         len(precisions),
        "n_recall":            len(recalls),
        "macro_precision":     round(np.mean(precisions), 2) if precisions else None,
        "macro_precision_std": round(np.std(precisions),  2) if precisions else None,
        "macro_recall":        round(np.mean(recalls),    2) if recalls    else None,
        "macro_recall_std":    round(np.std(recalls),     2) if recalls    else None,
        "micro_precision":     round(safe_div(sum_TF, sum_result), 2) if safe_div(sum_TF, sum_result) is not None else None,
        "micro_recall":        round(safe_div(sum_TF, sum_label),  2) if safe_div(sum_TF, sum_label)  is not None else None,
        "macro_value_acc":     round(np.mean(val_accs), 2) if val_accs else None,
        "macro_value_acc_std": round(np.std(val_accs),  2) if val_accs else None,
        "micro_value_acc":     round(safe_div(sum_correct, sum_TF), 2) if safe_div(sum_correct, sum_TF) is not None else None,
    })

# 전체 (카테고리 합산)
precisions_all, recalls_all, val_accs_all = [], [], []
sum_TF_all, sum_result_all, sum_label_all, sum_correct_all = 0, 0, 0, 0

for _, r in comparison_df.iterrows():
    total_label   = sum(r[f"{cat}_label_len"]  for cat in EVAL_CATS)
    total_result  = sum(r[f"{cat}_result_len"] for cat in EVAL_CATS)
    total_TF      = sum(r[f"{cat}_TF"]         for cat in EVAL_CATS)
    total_correct = sum(r[f"{cat}_correct"]    for cat in EVAL_CATS)
    if total_label == 0 and total_result == 0:
        continue

    p  = safe_div(total_TF, total_result)
    re = safe_div(total_TF, total_label)
    va = safe_div(total_correct, total_TF)
    if p  is not None: precisions_all.append(p)
    if re is not None: recalls_all.append(re)
    if va is not None: val_accs_all.append(va)

    sum_TF_all      += total_TF
    sum_result_all  += total_result
    sum_label_all   += total_label
    sum_correct_all += total_correct

rows.append({
    "카테고리":            "전체",
    "label_항목수":        int(sum(comparison_df[f"{cat}_label_len"].sum() for cat in EVAL_CATS)),
    "n_precision":         len(precisions_all),
    "n_recall":            len(recalls_all),
    "macro_precision":     round(np.mean(precisions_all), 2) if precisions_all else None,
    "macro_precision_std": round(np.std(precisions_all),  2) if precisions_all else None,
    "macro_recall":        round(np.mean(recalls_all),    2) if recalls_all    else None,
    "macro_recall_std":    round(np.std(recalls_all),     2) if recalls_all    else None,
    "micro_precision":     round(safe_div(sum_TF_all, sum_result_all), 2) if safe_div(sum_TF_all, sum_result_all) is not None else None,
    "micro_recall":        round(safe_div(sum_TF_all, sum_label_all),  2) if safe_div(sum_TF_all, sum_label_all)  is not None else None,
    "macro_value_acc":     round(np.mean(val_accs_all), 2) if val_accs_all else None,
    "macro_value_acc_std": round(np.std(val_accs_all),  2) if val_accs_all else None,
    "micro_value_acc":     round(safe_div(sum_correct_all, sum_TF_all), 2) if safe_div(sum_correct_all, sum_TF_all) is not None else None,
})

stats_df = pd.DataFrame(rows).set_index("카테고리")
print(stats_df.to_string())


# ──────────────────────────────────────────────────────────────────────────────
# ':' 없는 label 항목 점검 (블록 7)
# ──────────────────────────────────────────────────────────────────────────────
pd.set_option("display.max_rows", None)
no_colon = []
for _, row in data.iterrows():
    for cat in EVAL_CATS:
        cell_data = row[cat]
        if not cell_data or isinstance(cell_data, float):
            continue
        lines = [x.strip() for x in str(cell_data).split("\n") if x.strip()]
        for line in lines:
            line_clean = line.lstrip("- ").strip()
            if line_clean and ":" not in line_clean:
                no_colon.append({
                    "구분자":   row["A_ID"],
                    "카테고리": cat,
                    "원문":     line,
                })

no_colon_df = pd.DataFrame(no_colon)
print(f"총 {len(no_colon_df)}건 / {len(data)}")
print(no_colon_df.to_string())
