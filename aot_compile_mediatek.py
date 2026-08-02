#!/usr/bin/env python3
"""AOT-compile a TFLite model for MediaTek NPU (MT6989 = Dimensity 9300).

Usage: python3 aot_compile_mediatek.py <model.tflite> [<soc>]
"""
import sys
import traceback

from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.vendors.mediatek import target as mtk_target


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: aot_compile_mediatek.py <model.tflite> [soc]", file=sys.stderr)
        return 2
    model_path = sys.argv[1]
    soc = sys.argv[2] if len(sys.argv) > 2 else "MT6989"
    target = mtk_target.Target(getattr(mtk_target.SocModel, soc))
    print(f"Target: {target} | NP: {target.recommended_np_version}")
    try:
        compiled = aot_lib.aot_compile(model_path, target=target, keep_going=True)
        print("AOT COMPILE OK")
        print("Outputs:", compiled)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
