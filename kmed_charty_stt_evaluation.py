"""
KMed Charty STT 평가 (CER / WER)

KMed_Charty STT 결과(STT)와 전문가 전사본(Transcription)을 비교하여
CER(Character Error Rate)과 WER(Word Error Rate)을 계산하고 통계를 출력합니다.

입력 데이터 (smc_records.xlsx)
- all       시트: 전체 케이스 (A_ID, C_ID, Transcription, STT)
- exception 시트: 제외할 케이스의 A_ID 목록
"""

import numpy as np
import pandas as pd
import Levenshtein as Lev

# ── 설정 ──────────────────────────────────────────────────────────────────────
EXCEL_PATH = "smc_records.xlsx"


# ── 전처리 ────────────────────────────────────────────────────────────────────
def preprocess(text, for_cer=True):
    text = str(text)
    text = text.replace("네", "").replace(".", "").replace(",", "").replace("?", "").replace("~", "")
    if for_cer:
        text = text.replace(" ", "")
    else:
        text = text.replace("\n", " ").replace("(", "").replace(")", "")
        while "  " in text:
            text = text.replace("  ", " ")
        text = text.strip()
    return text


# ── CER (Character Error Rate) ────────────────────────────────────────────────
def cer(ref, hyp):
    ref = preprocess(ref, for_cer=True)
    hyp = preprocess(hyp, for_cer=True)
    dist   = Lev.distance(hyp, ref)
    length = len(ref)
    return dist, length, dist / length if length > 0 else 0.0


# ── WER (Word Error Rate) ─────────────────────────────────────────────────────
def wer(ref, hyp):
    ref = preprocess(ref, for_cer=False)
    hyp = preprocess(hyp, for_cer=False)
    r = ref.split()
    h = hyp.split()
    if not r:
        return 0, 0, 0, len(h), float("nan")

    costs    = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    backtrace = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]

    OP_OK, OP_SUB, OP_INS, OP_DEL = 0, 1, 2, 3

    for i in range(1, len(r) + 1):
        costs[i][0]    = i
        backtrace[i][0] = OP_DEL
    for j in range(1, len(h) + 1):
        costs[0][j]    = j
        backtrace[0][j] = OP_INS

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                costs[i][j]    = costs[i - 1][j - 1]
                backtrace[i][j] = OP_OK
            else:
                sub_cost = costs[i - 1][j - 1] + 1
                ins_cost = costs[i][j - 1]     + 1
                del_cost = costs[i - 1][j]     + 1
                costs[i][j] = min(sub_cost, ins_cost, del_cost)
                if costs[i][j] == sub_cost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == ins_cost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    i, j = len(r), len(h)
    num_cor = num_sub = num_del = num_ins = 0
    while i > 0 or j > 0:
        op = backtrace[i][j]
        if op == OP_OK:
            num_cor += 1; i -= 1; j -= 1
        elif op == OP_SUB:
            num_sub += 1; i -= 1; j -= 1
        elif op == OP_INS:
            num_ins += 1; j -= 1
        else:
            num_del += 1; i -= 1

    score = (num_sub + num_del + num_ins) / len(r)
    return num_cor, num_sub, num_del, num_ins, score


# ── 데이터 로드 ────────────────────────────────────────────────────────────────
def load_data(excel_path):
    df_all       = pd.read_excel(excel_path, sheet_name="all")
    df_exception = pd.read_excel(excel_path, sheet_name="exception")

    samples_except = set(df_exception["A_ID"].tolist())

    data = df_all.to_dict(orient="records")
    data = [d for d in data if d["A_ID"] not in samples_except]

    print(f"총 {len(data)}건 로드 완료")
    return data


# ── 평가 및 출력 ───────────────────────────────────────────────────────────────
def evaluate(data):
    cer_results = []
    wer_results = []
    wer_failed  = []

    for row in data:
        label = row["Transcription"]
        stt   = row["STT"]

        try:
            cer_results.append(cer(label, stt))
        except Exception as e:
            print(f"  CER 실패 [{row['A_ID']}]: {e}")

        try:
            wer_results.append(wer(label, stt))
        except Exception as e:
            wer_failed.append(row["A_ID"])

    if wer_failed:
        print(f"WER 계산 실패: {wer_failed}")

    # ── CER 결과 출력 ──────────────────────────────────────────────────────────
    cer_scores = [x[2] for x in cer_results]
    print(f"\n{'=' * 40}")
    print(f"[CER]  n={len(cer_scores)}")
    print(f"{'=' * 40}")
    print(f"  mean : {np.mean(cer_scores):.4f}")
    print(f"  std  : {np.std(cer_scores):.4f}")

    # ── WER 결과 출력 ──────────────────────────────────────────────────────────
    wer_scores = [x[4] for x in wer_results]
    print(f"\n{'=' * 40}")
    print(f"[WER]  n={len(wer_scores)}")
    print(f"{'=' * 40}")
    print(f"  mean : {np.mean(wer_scores):.4f}")
    print(f"  std  : {np.std(wer_scores):.4f}")


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_data(EXCEL_PATH)
    evaluate(data)
