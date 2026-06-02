"""
EMR 오류 라벨링 분석 (L1 / L2 / L3)

AI 생성 의무기록(KMed_emr)과 전문가 수정 의무기록(Revised_emr)을 LLM으로 비교하여
오류 수준(L1 / L2 / L3)을 라벨링하고, 도입 전(bef) / 후(aft) 그룹별 통계를 출력합니다.

입력 데이터 (smc_records.xlsx)
- all       시트: 전체 케이스 (A_ID, C_ID, date, Transcription_revised, KMed_emr, Revised_emr)
- exception 시트: 제외할 케이스의 A_ID 목록
"""

import json
import time

import numpy as np
import pandas as pd
import requests
from json_repair import repair_json

# ── 설정 ──────────────────────────────────────────────────────────────────────
EXCEL_PATH = "smc_records.xlsx"
API_URL    = "YOUR_PLATFORM_URL"
API_KEY    = "YOUR_API_KEY"   # API 키 입력
MODEL      = "gpt-oss-120b"
CUTOFF     = "2025/05/26"     # 시스템 도입 전/후 기준일

# 의료진 그룹 정의
GROUPS = {
    "전문간호사": [f"C{i:02d}" for i in range(1,  6)],  # C01-C05
    "전공의":     [f"C{i:02d}" for i in range(6,  9)],  # C06-C08
    "전문의":     [f"C{i:02d}" for i in range(9, 19)], # C12-C18
}

SYSTEM_PROMPT = """당신은 응급실 의무기록 전문 평가자입니다.
AI가 생성한 의무기록(KMed_emr)과 전문가가 검토·수정한 의무기록(revised_emr)을 비교하여,
수정된 각 내용에 대해 아래 기준으로 라벨링하고 근거를 작성하세요.

[라벨 기준]
- L3 (Potential Clinical Harm): 진료 시 환자에게 위해를 끼칠 수 있는 오류 (정보 누락, 틀린 정보 기입 등)
- L2 (Informational Incompleteness): 틀린 정보는 없으나 진단 정밀도를 높이는 맥락적 정보가 누락된 경우
- L1 (Structural Refinement): 임상 정보는 정확하나 용어·약어로 정제하여 다시 작성한 경우

예시 1)
KMed_emr - 한 달 전부터 어지러움이 있음
Revised_emr - 약 1개월 전부터 몸이 붓는 증상이 있음
Label 결과:
L3, L1
Labeling 근거:
L3: 몸이 붓는 증상인데 어지러움으로 작성됨
L1: 한 달과 1개월 동일한 의미이나 다르게 쓰임

[중요 원칙]
- 라벨 우선순위: L3 > L2 > L1 (동일 수정에 복수 라벨 가능)
- KMed_emr에는 있으나 revised_emr에서 삭제된 내용, 또는 revised_emr에 새로 추가된 내용 모두 분석 대상
- Transcription(원본 음성 전사)을 참고하여 KMed_emr의 오류 여부를 판단하세요
- KMed_emr과 revised_emr이 동일하면 "수정 없음"으로 반환하세요

[출력 형식 - 반드시 JSON으로만 반환]
{
  "has_changes": true 또는 false,
  "labels": ["L3", "L2", "L1"] 중 해당하는 것,
  "highest_label": "L3" 또는 "L2" 또는 "L1" 또는 "없음",
  "details": [
    {
      "label": "L3",
      "modified_content": "수정된 내용 (구체적으로)",
      "reason": "해당 라벨을 부여한 근거"
    }
  ]
}"""


# ── 데이터 로드 ────────────────────────────────────────────────────────────────
def load_data(excel_path):
    df_all       = pd.read_excel(excel_path, sheet_name="all")
    df_exception = pd.read_excel(excel_path, sheet_name="exception")

    samples_except = set(df_exception["A_ID"].tolist())

    data = df_all.to_dict(orient="records")
    data = [d for d in data if d["A_ID"] not in samples_except]

    print(f"총 {len(data)}건 로드 완료")
    return data


# ── LLM 호출 ──────────────────────────────────────────────────────────────────
def build_user_prompt(transcription, kmed_emr, revised_emr):
    trans_section = (
        f"[Transcription - 원본 음성 전사]\n{transcription}\n\n"
        if pd.notna(transcription) and str(transcription).strip()
        else ""
    )
    return (
        f"{trans_section}"
        f"[KMed_emr - AI 생성 의무기록]\n{kmed_emr}\n\n"
        f"[revised_emr - 전문가 수정 의무기록]\n{revised_emr}\n\n"
        f"위 두 의무기록을 비교하여 수정된 내용에 대해 라벨링하고 JSON 형식으로만 반환하세요."
    )


def call_llm(transcription, kmed_emr, revised_emr, max_retries=5):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(transcription, kmed_emr, revised_emr)},
        ],
        "max_tokens": 8192,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            content  = response.json()["choices"][0]["message"]["content"].strip()
            try:
                return json.loads(content)
            except Exception:
                return json.loads(repair_json(content))
        except Exception as e:
            print(f"  재시도 {attempt + 1}/{max_retries}: {e}")
            time.sleep(5)
    return None


# ── LLM 평가 실행 ─────────────────────────────────────────────────────────────
def run_evaluation(data):
    for i, row in enumerate(data):
        print(f"[{i + 1:03d}/{len(data)}] {row['A_ID']} 처리 중...", end=" ")
        result = call_llm(
            row.get("Transcription_revised"),
            row["KMed_emr"],
            row["Revised_emr"],
        )
        if result:
            details = result.get("details", [])
            row["evaluation"] = result
            row["risk_score"] = {
                "L1": len([x for x in details if "L1" in x.get("label", "")]),
                "L2": len([x for x in details if "L2" in x.get("label", "")]),
                "L3": len([x for x in details if "L3" in x.get("label", "")]),
            }
        else:
            row["evaluation"] = None
            row["risk_score"] = {"L1": 0, "L2": 0, "L3": 0}
        print("완료")
        time.sleep(0.3)

    print(f"\n평가 완료: {len(data)}건")
    return data


# ── 통계 계산 ─────────────────────────────────────────────────────────────────
def calc_stats(records):
    if not records:
        return dict(
            n=0,
            L1=np.nan, L1_std=np.nan,
            L2=np.nan, L2_std=np.nan,
            L3=np.nan, L3_std=np.nan,
            score_mean=np.nan, score_std=np.nan,
        )
    return dict(
        n          = len(records),
        L1         = round(np.mean([d["risk_score"].get("L1", 0) for d in records]), 2),
        L1_std     = round(np.std( [d["risk_score"].get("L1", 0) for d in records]), 2),
        L2         = round(np.mean([d["risk_score"].get("L2", 0) for d in records]), 2),
        L2_std     = round(np.std( [d["risk_score"].get("L2", 0) for d in records]), 2),
        L3         = round(np.mean([d["risk_score"].get("L3", 0) for d in records]), 2),
        L3_std     = round(np.std( [d["risk_score"].get("L3", 0) for d in records]), 2),
        score_mean = round(np.mean([d["score"] for d in records]), 2),
        score_std  = round(np.std( [d["score"] for d in records]), 2),
    )


def print_stats(data):
    # period 태깅 및 score 계산 (score = L1×1 + L2×2 + L3×3)
    for d in data:
        rs = d.get("risk_score", {"L1": 0, "L2": 0, "L3": 0})
        d["score"]  = rs.get("L1", 0) * 1 + rs.get("L2", 0) * 2 + rs.get("L3", 0) * 3
        d["period"] = "aft" if d["date"] >= CUTOFF else "bef"

    # 사람별 통계
    person_rows = []
    for cid in sorted(set(d["C_ID"] for d in data)):
        for period in ("bef", "aft", "total"):
            recs = [d for d in data if d["C_ID"] == cid and (period == "total" or d["period"] == period)]
            person_rows.append({"C_ID": cid, "period": period, **calc_stats(recs)})

    person_df = pd.DataFrame(
        person_rows,
        columns=["C_ID", "period", "n", "L1", "L1_std", "L2", "L2_std", "L3", "L3_std", "score_mean", "score_std"],
    )

    # 그룹별 통계
    group_rows = []
    for g_name, cids_group in GROUPS.items():
        for period in ("bef", "aft", "total"):
            recs = [d for d in data if d["C_ID"] in cids_group and (period == "total" or d["period"] == period)]
            group_rows.append({"group": g_name, "period": period, **calc_stats(recs)})

    group_df = pd.DataFrame(
        group_rows,
        columns=["group", "period", "n", "L1", "L1_std", "L2", "L2_std", "L3", "L3_std", "score_mean", "score_std"],
    )

    # period별 최종 출력
    COLS = ["C_ID", "n", "L1", "L1_std", "L2", "L2_std", "L3", "L3_std", "score_mean", "score_std"]

    for period in ("total", "bef", "aft"):
        p_slice = (
            person_df[person_df["period"] == period]
            .drop(columns="period")
            .reset_index(drop=True)
        )
        g_slice = (
            group_df[group_df["period"] == period]
            .drop(columns="period")
            .rename(columns={"group": "C_ID"})
        )
        recs_all = [d for d in data if period == "total" or d["period"] == period]
        overall  = pd.DataFrame([{"C_ID": "전체", **calc_stats(recs_all)}])

        combined = pd.concat([p_slice, g_slice, overall], ignore_index=True)

        print(f"\n{'=' * 72}")
        print(f"[{period.upper()}]  score = L1×1 + L2×2 + L3×3")
        print(f"{'=' * 72}")
        print(combined[COLS].to_string(index=False))


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_data(EXCEL_PATH)
    data = run_evaluation(data)
    print_stats(data)
