import re
import os

files_to_patch = [
    '/usr/local/cuda-12.8/include/crt/math_functions.h',
    '/usr/local/cuda-12.8/include/crt/math_functions.hpp'
]

funcs = ['rsqrt', 'rsqrtf', 'sinpi', 'sinpif', 'cospi', 'cospif']

for path in files_to_patch:
    if not os.path.exists(path): continue
    with open(path, 'r') as f:
        content = f.read()

    for func in funcs:
        pattern = r"((?:double|float)\s+" + func + r"\s*\([^)]+\))\s*(?!noexcept)(;|{)"
        content = re.sub(pattern, r"\1 noexcept(true)\2", content)

    # Patch abs/labs/llabs __THROW declarations
    pattern_abs = r"(extern\s+__DEVICE_FUNCTIONS_DECL__\s+__device_builtin__\s+__cudart_builtin__\s+(?:int|long\s+int|long\s+long\s+int)\s+__cdecl\s+(?:abs|labs|llabs)\s*\([^)]+\)\s+__THROW\s*;)"
    content = re.sub(pattern_abs, r"/* \1 */", content)

    with open(path, 'w') as f:
        f.write(content)

print("Patch applied to headers!")
