# STT — m4a 회의 녹음 한국어 전사

`.m4a` 회의/미팅 녹음을 [OpenAI Whisper](https://github.com/openai/whisper)로 한국어 STT 하여 텍스트 파일로 저장하는 스크립트.

- `input/` 폴더에 `.m4a` 를 넣고 실행하면, 전부 전사해서 `output/<원본이름>.txt` 로 저장
- 처리 끝난 원본 `.m4a` 는 `input/finish/` 로 자동 이동 (다음 실행 때 중복 처리 안 함)
- 실행 시 컴퓨터 사양(GPU / VRAM / RAM)을 자동 감지해서 Whisper 모델 크기를 자동 선택

## 디렉터리 구조

```
STT/
├─ transcribe.py        # 메인 스크립트
├─ requirements.txt      # 파이썬 의존성
├─ input/                # 여기에 .m4a 넣기
│  └─ finish/            # 처리 끝난 .m4a 가 여기로 이동됨
├─ output/               # 전사 결과 .txt
└─ ffmpeg/               # (직접 받은 ffmpeg, git 에는 포함 안 됨)
```

## 설치

### 1. 파이썬 의존성

```bash
pip install -r requirements.txt
```

`requirements.txt` 는 `openai-whisper`, `psutil` 을 설치하며 `torch`(CPU 빌드)가 함께 깔린다.

### 2. NVIDIA GPU 가속 (선택, 강력 추천)

CPU 만으로도 동작하지만 긴 회의 녹음은 매우 느리다. NVIDIA GPU가 있으면 CUDA 빌드 `torch` 로 교체한다.

```bash
pip uninstall -y torch
# CUDA 12.x 빌드 (그래픽 드라이버에 맞는 버전 선택)
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

> Python 3.14 처럼 최신 버전이면 정식 채널에 CUDA 휠이 아직 없을 수 있다. 그 경우 nightly 채널을 쓴다:
> ```bash
> pip install torch --pre --index-url https://download.pytorch.org/whl/nightly/cu128
> ```

설치 후 확인:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. ffmpeg

Whisper 는 오디오 디코딩에 `ffmpeg` 를 사용한다.

- 패키지 매니저: `winget install Gyan.FFmpeg` (winget 있는 경우)
- 수동 설치: <https://www.gyan.dev/ffmpeg/builds/> 에서 `ffmpeg-release-essentials.zip` 다운로드 → 압축 풀어서 `bin` 폴더를 PATH 에 추가
  - 또는 압축 푼 폴더를 이 저장소의 `ffmpeg/` 로 두면(`ffmpeg/bin/ffmpeg.exe` 형태) `transcribe.py` 가 PATH 설정 없이도 자동 인식한다.

확인: `ffmpeg -version`

## 사용법

```bash
# input/ 의 모든 .m4a 처리
python transcribe.py

# 구간 타임스탬프 포함  ([00:01:23 - 00:01:30] 형태)
python transcribe.py --timestamps

# 모델 직접 지정 (tiny / base / small / medium / large-v3)
python transcribe.py --model large-v3

# 디바이스 강제 지정
python transcribe.py --device cuda
python transcribe.py --device cpu

# 이미 output txt 가 있어도 다시 처리
python transcribe.py --overwrite
```

## 모델 자동 선택 기준

| 환경 | 조건 | 선택 모델 |
|---|---|---|
| GPU (CUDA) | 사용 가능 VRAM ≥ ~10 GB | `large-v3` |
| GPU (CUDA) | ~5 GB 이상 | `medium` |
| GPU (CUDA) | ~2 GB 이상 | `small` |
| CPU | RAM ≥ 16 GB | `medium` |
| CPU | RAM ≥ 8 GB | `small` |
| CPU | RAM ≥ 4 GB | `base` |

(VRAM 은 시스템/런타임 여유로 1 GB 를 빼고 계산한다. `--model` 로 언제든 덮어쓸 수 있다.)

최초 실행 시 선택된 모델 가중치를 자동으로 다운로드한다 (`large-v3` 는 약 2.9 GB, 사용자 캐시 디렉터리에 저장됨).

## 트러블슈팅

### `winget : The term 'winget' is not recognized`
`winget`(앱 설치 관리자)이 없거나 PATH 에 없는 환경. ffmpeg 를 수동 설치(위 "3. ffmpeg" 참고)하거나, 압축 푼 폴더를 저장소의 `ffmpeg/` 로 두면 된다.

### `RuntimeError: ... ffmpeg ...` / `FileNotFoundError: [WinError 2]` (전사 시작 직후)
ffmpeg 가 PATH 에 없음. `ffmpeg -version` 으로 확인하고, 새 터미널을 열어 PATH 가 반영됐는지 확인. 또는 `ffmpeg/bin/ffmpeg.exe` 형태로 저장소 안에 두면 스크립트가 자동으로 잡는다.

### `torch.cuda.is_available()` 가 `False` 인데 GPU 가 있음
- CPU 빌드 `torch` 가 깔려 있음 → 위 "2. GPU 가속" 대로 CUDA 빌드로 재설치
- `pip install torch --index-url ...` 가 `Could not find a version that satisfies the requirement torch` 로 실패 → 파이썬 버전이 너무 최신이라 해당 CUDA 채널에 휠이 없음. nightly 채널(`/whl/nightly/cu128`)을 시도
- NVIDIA 그래픽 드라이버가 오래됨 → 드라이버 업데이트

### CUDA `out of memory`
- 더 작은 모델 사용: `python transcribe.py --model medium`
- 다른 GPU 사용 프로그램(브라우저 GPU 가속, 게임 등) 종료
- 또는 `--device cpu` 로 CPU 실행

### CPU 에서 너무 느림
3시간 녹음 기준 CPU(medium)는 수 시간 걸릴 수 있다. GPU 가속을 쓰거나, 정확도를 양보하고 `--model small` / `--model base` 로 낮춘다.

### 전사 결과가 부정확하거나 영어로 나옴
- 더 큰 모델 사용 (`--model large-v3`)
- 녹음 음질이 나쁘면(소음, 멀리서 녹음) 한계가 있음
- 언어는 한국어(`ko`)로 고정되어 있다. 다른 언어가 섞여 있으면 코드의 `language="ko"` 부분을 조정

### `.txt` 가 너무 짧게 나옴
무음 구간이 길거나 말이 띄엄띄엄한 녹음이면 정상. 내용 확인 후 필요하면 `--timestamps` 로 구간을 같이 보면 어디가 비었는지 알 수 있다.

### 같은 파일을 다시 전사하고 싶음
`output/` 에서 해당 `.txt` 를 지우거나 `--overwrite` 옵션을 쓴다. 단, 원본 `.m4a` 는 `input/finish/` 로 이동돼 있으니 `input/` 으로 다시 옮겨야 한다.

## 동작 요약 (transcribe.py)

1. `input/*.m4a` 목록을 모은다 (`input/finish/` 는 제외)
2. 이미 `output/<이름>.txt` 가 있으면 건너뜀 (`--overwrite` 면 무시)
3. GPU/VRAM/RAM 을 감지해 모델·디바이스 결정 (`--model`, `--device` 로 덮어쓰기 가능)
4. Whisper 모델 로드 (최초 1회 다운로드)
5. 각 파일을 `language="ko"` 로 전사 → `output/<이름>.txt` 저장 → 원본을 `input/finish/` 로 이동
