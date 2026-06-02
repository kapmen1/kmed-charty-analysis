# KMed Charty — Analysis Code

응급실 AI 의무기록 시스템(KMed Charty) 실증 연구를 위한 분석 코드입니다.

---

## 파일 구성

| 파일 | 설명 |
|---|---|
| `kmed_charty_analysis_record.py` | STT 소요 시간 및 의무기록 작성 시간 분석 |
| `kmed_charty_llm_eval.py` | LLM 기반 EMR 오류 라벨링 (L1 / L2 / L3) |
| `kmed_charty_roc_evaluation.py` | 구조화 필드 Precision / Recall / Value Accuracy 평가 |
| `kmed_charty_stt_evaluation.py` | STT CER / WER 평가 |
| `requirements.txt` | 의존 패키지 목록 |

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 입력 데이터

모든 스크립트는 **`smc_records.xlsx`** 단일 파일을 입력으로 사용합니다.

### 시트 구성

#### `exception` 시트 (공통)
제외할 케이스의 A_ID 목록입니다.

| 컬럼 | 설명 |
|---|---|
| `A_ID` | 제외 대상 케이스 번호 |

#### `all` 시트
스크립트별로 필요한 컬럼이 다릅니다.

| 컬럼 | 설명 | analysis_record | llm_eval | roc_evaluation | stt_evaluation |
|---|---|:---:|:---:|:---:|:---:|
| `A_ID` | 케이스 번호 | ✓ | ✓ | ✓ | ✓ |
| `C_ID` | 의료진 ID (C01–C18) | ✓ | ✓ | | |
| `date` | 진료 날짜 | ✓ | ✓ | | |
| `KMed_date` | KMed STT 시작 일시 (`YYYY/MM/DD_HH:MM`) | ✓ | | | |
| `KMed_time` | KMed STT 소요 시간 (time 형식) | ✓ | | | |
| `recording_time_1` | 첫 번째 녹음 일시 (`YYYY/MM/DD_HH:MM`) | ✓ | | | |
| `recording_time_2` | 두 번째 녹음 일시 (`YYYY/MM/DD_HH:MM`) | ✓ | | | |
| `gender` | 의료진 성별 (`남자` / `여자`) | ✓ | | | |
| `KMed_emr` | KMed Charty AI 생성 의무기록 | | ✓ | ✓ | |
| `Transcription_revised` | 전문가 수정 전사본 | | ✓ | | |
| `Revised_emr` | 전문가 수정 의무기록 (완성 문자열) | | ✓ | | |
| `과거력` | 전문가 수정 과거력 (key: value 형식) | | | ✓ | |
| `계통문진` | 전문가 수정 계통문진 (key: value 형식) | | | ✓ | |
| `신체검진` | 전문가 수정 신체검진 (key: value 형식) | | | ✓ | |
| `과거력_수정여부` | 과거력 수정 여부 (`Y` / `N`) | | | ✓ | |
| `계통문진_수정여부` | 계통문진 수정 여부 (`Y` / `N`) | | | ✓ | |
| `신체검진_수정여부` | 신체검진 수정 여부 (`Y` / `N`) | | | ✓ | |
| `Transcription` | 정답 전사본 (reference) | | | | ✓ |
| `STT` | KMed STT 출력 전사본 (hypothesis) | | | | ✓ |

---

## 실행 방법

각 스크립트는 독립적으로 실행합니다. `smc_records.xlsx`가 같은 디렉토리에 있어야 합니다.

```bash
# STT 소요 시간 및 의무기록 작성 시간 분석
python kmed_charty_analysis_record.py

# LLM 기반 EMR 오류 라벨링
python kmed_charty_llm_eval.py

# 구조화 필드 정확도 평가
python kmed_charty_roc_evaluation.py

# STT CER / WER 평가
python kmed_charty_stt_evaluation.py
```

> `kmed_charty_llm_eval.py` 실행 전, 스크립트 상단의 `API_KEY`와 `API_URL`을 실제 값으로 교체하세요.

---

## 출력 설명

### `kmed_charty_analysis_record.py`
- **표 1**: 성별 / 경력 / 나이대별 KMed_time (STT 소요 시간) — Mann-Whitney U + Mixed-effects p-value
- **표 2**: 직군별 elapsed2_kmed (의무기록 작성 소요 시간) — Mann-Whitney U p-value
- **표 3**: 성별 / 경력 / 나이대별 elapsed2_kmed — Mann-Whitney U p-value

### `kmed_charty_llm_eval.py`
- **[TOTAL / BEF / AFT] 표**: 의료진별 · 직군별 L1 / L2 / L3 오류 수 평균·표준편차 및 종합 score
  - score = L1×1 + L2×2 + L3×3
  - BEF: 시스템 도입 전 (2025-05-26 이전), AFT: 도입 후

### `kmed_charty_roc_evaluation.py`
- **비교 데이터 shape 및 head**
- **카테고리별 정확도 표**: 과거력 / 계통문진 / 신체검진 / 전체
  - Macro · Micro Precision, Recall, Value Accuracy
- **`:` 없는 label 항목 점검 결과**

### `kmed_charty_stt_evaluation.py`
- **[CER]**: Character Error Rate 평균 · 표준편차
- **[WER]**: Word Error Rate 평균 · 표준편차
