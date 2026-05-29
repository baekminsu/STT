"""input/ 폴더의 .m4a 파일들을 한국어로 STT 하여 output/ 폴더에 txt 로 저장.

- input/ 안의 모든 .m4a 를 처리한다.
- 결과는 output/<원본이름>.txt 로 저장한다.
- 처리 끝난 원본 .m4a 는 input/finish/ 로 옮긴다 (다음 실행 때 다시 처리 안 함).
- 컴퓨터 사양(GPU/VRAM/RAM)을 실행 시 자동 파악해서 whisper 모델 크기를 자동 선정한다.
  (필요하면 --model 로 직접 지정)

사용법:
    python transcribe.py                       # input/ 의 모든 .m4a 처리
    python transcribe.py --timestamps          # 구간 타임스탬프 포함
    python transcribe.py --model large-v3      # 모델 직접 지정
    python transcribe.py --overwrite           # 이미 txt 가 있어도 다시 처리

준비물:
    pip install -r requirements.txt
    ffmpeg 설치 필요 (whisper 가 m4a 디코딩에 사용).
        Windows: winget install Gyan.FFmpeg   또는  https://ffmpeg.org/download.html
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import whisper

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
FINISH_DIR = INPUT_DIR / "finish"  # 처리 끝난 .m4a 를 옮겨두는 곳

# 같이 받아둔 ffmpeg(C:\STT\ffmpeg\bin)이 있으면 PATH 에 추가 (시스템 PATH 설정 안돼 있어도 동작)
_FFMPEG_BIN = BASE_DIR / "ffmpeg" / "bin"
if _FFMPEG_BIN.is_dir():
    os.environ["PATH"] = str(_FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")

# verbose 실시간 출력에 콘솔 기본 인코딩(예: 한글 Windows 의 cp949)으로 표현 못 하는
# 글자가 섞여도 UnicodeEncodeError 로 죽지 않도록 에러 처리만 'replace' 로 완화한다.
# (인코딩은 그대로 두므로 한글은 정상 표시되고, txt 결과는 항상 utf-8 로 저장됨)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

# 모델별 대략적인 GPU VRAM 요구량(GB) — 자동 선정 기준
MODEL_VRAM_GB = {
    "tiny": 1,
    "base": 1,
    "small": 2,
    "medium": 5,
    "large-v3": 10,
}


def detect_specs() -> dict:
    specs = {
        "cuda": torch.cuda.is_available(),
        "gpu_name": None,
        "vram_gb": 0.0,
        "ram_gb": 0.0,
        "cpu_count": os.cpu_count() or 1,
    }
    if specs["cuda"]:
        specs["gpu_name"] = torch.cuda.get_device_name(0)
        specs["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    try:
        import psutil
        specs["ram_gb"] = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            specs["ram_gb"] = stat.ullTotalPhys / (1024 ** 3)
        except Exception:
            specs["ram_gb"] = 0.0
    return specs


def choose_model(specs: dict) -> tuple[str, str]:
    """사양에 따라 (모델명, 디바이스) 반환."""
    if specs["cuda"]:
        usable = max(specs["vram_gb"] - 1.0, 0)  # 시스템/런타임 여유 1GB
        for model in ("large-v3", "medium", "small", "base", "tiny"):
            if usable >= MODEL_VRAM_GB[model]:
                return model, "cuda"
        return "tiny", "cuda"

    ram = specs["ram_gb"]
    if ram >= 16:
        return "medium", "cpu"
    if ram >= 8:
        return "small", "cpu"
    if ram >= 4:
        return "base", "cpu"
    return "tiny", "cpu"


def fmt_ts(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe_file(model, src: Path, dst: Path, with_timestamps: bool, use_fp16: bool) -> None:
    print(f"\n[처리 중] {src.name} (변환되는 구간이 아래에 실시간으로 표시됩니다)")
    result = model.transcribe(str(src), language="ko", verbose=True, fp16=use_fp16)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        f.write(f"# {src.name}\n\n")
        if with_timestamps:
            for seg in result["segments"]:
                f.write(f"[{fmt_ts(seg['start'])} - {fmt_ts(seg['end'])}] {seg['text'].strip()}\n")
        else:
            f.write(result["text"].strip() + "\n")
    print(f"[완료] {dst}")


def move_to_finish(src: Path) -> Path:
    FINISH_DIR.mkdir(parents=True, exist_ok=True)
    target = FINISH_DIR / src.name
    if target.exists():
        i = 1
        while True:
            cand = FINISH_DIR / f"{src.stem} ({i}){src.suffix}"
            if not cand.exists():
                target = cand
                break
            i += 1
    src.rename(target)
    print(f"[이동] {src.name} -> input/finish/{target.name}")
    return target


def main():
    parser = argparse.ArgumentParser(description="input/*.m4a -> output/*.txt 한국어 STT (사양 자동 감지)")
    parser.add_argument("--model", default=None,
                        help="whisper 모델 직접 지정 (tiny/base/small/medium/large-v3). 미지정 시 사양 기반 자동 선정")
    parser.add_argument("--device", default=None, help="cpu 또는 cuda 강제 지정 (기본: 자동)")
    parser.add_argument("--timestamps", action="store_true", help="구간 타임스탬프 포함")
    parser.add_argument("--overwrite", action="store_true", help="이미 output txt 가 있어도 다시 처리")
    args = parser.parse_args()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = sorted(INPUT_DIR.glob("*.m4a"))
    if not sources:
        sys.exit(f"input 폴더에 .m4a 파일이 없습니다: {INPUT_DIR}")

    # 처리할 대상 추리기
    todo = []
    for src in sources:
        dst = OUTPUT_DIR / (src.stem + ".txt")
        if dst.exists() and not args.overwrite:
            print(f"[건너뜀] {src.name} -> 이미 {dst.name} 있음 (--overwrite 로 재처리)")
            continue
        todo.append((src, dst))

    if not todo:
        print("새로 처리할 파일이 없습니다.")
        return

    specs = detect_specs()
    print("=== 컴퓨터 사양 ===")
    if specs["cuda"]:
        print(f"GPU: {specs['gpu_name']} (VRAM {specs['vram_gb']:.1f} GB)")
    else:
        print("GPU: 사용 불가 (CPU 로 실행)")
    print(f"RAM: {specs['ram_gb']:.1f} GB / CPU 코어: {specs['cpu_count']}")

    auto_model, auto_device = choose_model(specs)
    model_name = args.model or auto_model
    device = args.device or auto_device
    use_fp16 = device == "cuda"
    print(f"=> 모델: {model_name} ({'사용자 지정' if args.model else '자동 선정'}), 디바이스: {device}")

    print(f"\n모델 로딩 중: {model_name} (최초 실행 시 다운로드)")
    model = whisper.load_model(model_name, device=device)

    done = 0
    for src, dst in todo:
        try:
            transcribe_file(model, src, dst, args.timestamps, use_fp16)
            move_to_finish(src)
            done += 1
        except Exception as e:
            print(f"[실패] {src.name}: {e}")

    print(f"\n총 {done}/{len(todo)}개 파일 변환 완료. 결과 폴더: {OUTPUT_DIR}")
    print(f"처리 끝난 원본은 {FINISH_DIR} 로 이동됨.")


if __name__ == "__main__":
    main()
